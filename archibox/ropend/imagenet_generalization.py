import argparse
import math
from pathlib import Path

import nvidia.dali as dali
import torch
import torch.nn.functional as F
import tqdm
import wandb
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

from archibox.ropend.imagenet_classification import (
    Config,
    EncoderSelfAttention,
    ViTClassifier,
)
from archibox.ropend.positional_embeddings import (
    AbsolutePE,
    AbsolutePEConfig,
    AxialRoPE,
    AxialRoPEConfig,
    FixedSinCosPE,
    FixedSinCosPEConfig,
    UniformRoPE,
    UniformRoPEConfig,
)

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32


@dali.pipeline_def
def imagenet_valid_pipeline(resize: int, crop: int, num_shards, shard_id):
    images, labels = dali.fn.readers.file(
        file_root="/srv/datasets/ImageNet/val",
        num_shards=num_shards,
        shard_id=shard_id,
        stick_to_shard=True,
        name="imagenet_valid_reader",
    )
    images = dali.fn.decoders.image(images, output_type=dali.types.RGB, device="mixed")
    images = dali.fn.resize(images, resize_shorter=resize)
    images = dali.fn.crop_mirror_normalize(
        images.gpu(),
        dtype=dali.types.FLOAT,
        output_layout="CHW",
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return images, labels.gpu()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str)
    parser.add_argument("resolution", type=int)
    parser.add_argument("--temp", type=float)
    args = parser.parse_args()

    matches = sorted(
        Path("archibox/ropend/runs/imagenet_class/wandb/").glob(
            f"run-????????_??????-{args.run_id}*/"
        )
    )
    if len(matches) != 1:
        print(f"error: found {len(matches)} != 1 matches for run_id={args.run_id}")
        if len(matches) > 0:
            print(f"{matches=}")
        quit()
    rundir = matches[0] / "files"
    assert rundir.is_dir()

    ckpt = torch.load(rundir / "latest.ckpt")

    api = wandb.Api()
    run = api.run(f"archibox/ropend-imagenet-class/{matches[0].name.split('-')[-1]}")
    cfg = Config.model_validate(run.config)

    valid_pipe = imagenet_valid_pipeline(
        batch_size=256,
        num_threads=4,
        device_id=0,
        resize=math.floor(args.resolution / 0.875),
        crop=args.resolution,
        num_shards=1,
        shard_id=0,
    )
    valid_loader = DALIClassificationIterator(
        valid_pipe,
        reader_name="imagenet_valid_reader",
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True,
    )

    nh, nw = 224 // cfg.patch_size, 224 // cfg.patch_size
    if cfg.pos_emb == "absolute":
        pos_emb = AbsolutePE(
            AbsolutePEConfig(dim=cfg.dim, init_std=cfg.ape_init_std), nh, nw
        )
    elif cfg.pos_emb == "fixed":
        pos_emb = FixedSinCosPE(
            FixedSinCosPEConfig(
                dim=cfg.dim, min_freq=cfg.min_freq, max_freq=cfg.max_freq
            ),
            nh,
            nw,
        )
    elif cfg.pos_emb == "axial_rotary":
        pos_emb = AxialRoPE(
            AxialRoPEConfig(
                head_dim=cfg.head_dim,
                min_freq=cfg.min_freq,
                max_freq=cfg.max_freq,
                n_zero_freqs=cfg.n_zero_freqs,
            ),
            nh,
            nw,
        )
    else:
        assert cfg.pos_emb == "uniform_rotary"
        pos_emb = UniformRoPE(
            UniformRoPEConfig(
                n_heads=cfg.dim // cfg.head_dim if cfg.sep_rope_heads else 1,
                head_dim=cfg.head_dim,
                min_freq=cfg.min_freq,
                max_freq=cfg.max_freq,
                n_zero_freqs=cfg.n_zero_freqs,
                direction_spacing=cfg.direction_spacing,
                learnable=cfg.learnable_rope,
            ),
            nh,
            nw,
        )
    model = ViTClassifier(
        image_size=(224, 224),
        patch_size=cfg.patch_size,
        dim=cfg.dim,
        mlp_dim=cfg.mlp_dim,
        head_dim=cfg.head_dim,
        depth=cfg.depth,
        n_classes=1000,
        pos_emb=pos_emb,
        pooling=cfg.pooling,
    ).cuda()
    model.load_state_dict(ckpt["model"])

    model.image_size = (args.resolution, args.resolution)
    print(f"{type(pos_emb)=}")
    pos_emb.resize_to(args.resolution, orig_size=224)

    if args.temp is not None:
        print("scaling temperature...")
        for block in model.blocks:
            if isinstance(block, EncoderSelfAttention):
                block.scale_temperature = True
                block.temperature.fill_(args.temp)

    # if model.blocks[0].rotary is not None and not model.blocks[0].rotary.cfg.learnable:
    #     print(f"{model.blocks[0].rotary.cos_HWhF.shape=}")

    n = 0
    mean_nll = 0
    mean_acc = 0
    for batch in tqdm.tqdm(valid_loader, desc="validation", mininterval=5.0):
        images_NCHW = batch[0]["data"]
        labels_N = batch[0]["label"].squeeze(-1).long()

        logits_ND = model(images_NCHW.type(DTYPE))
        nll = F.cross_entropy(logits_ND.float(), labels_N)
        acc = (torch.argmax(logits_ND, dim=-1) == labels_N).float().mean()
        n += 1
        mean_nll += (nll - mean_nll) / n
        mean_acc += (acc - mean_acc) / n
    print(f"{mean_nll=:.6f}, {mean_acc=:.4f}")


if __name__ == "__main__":
    main()
