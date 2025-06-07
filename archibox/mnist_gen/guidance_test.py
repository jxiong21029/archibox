import argparse
from pathlib import Path

import torch
import torch.nn as nn
import tqdm
from einops import rearrange
from PIL import Image

from archibox.components import make_embedding
from archibox.mnist_gen.dataloading import mnist_loader
from archibox.mnist_gen.vector_flow import FlatTrigflow
from archibox.muon import Muon, split_muon_adamw_params


class MnistFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = make_embedding(11, 1024)
        self.flow = FlatTrigflow(
            1024,
            2048,
            data_dim=28 * 28,
            depth=4,
            min_freq=2 * torch.pi,
            max_freq=200 * torch.pi,
            freq_dim=512,
        )


def train():
    model = MnistFlow()
    model.cuda()

    muon_params, scalar_params, embeds_params, output_params = split_muon_adamw_params(
        model, verbose=True
    )

    if len(muon_params) > 0:
        muon = Muon(muon_params, lr=0.02, momentum=0.9, weight_decay=0.01)
    adamw = torch.optim.AdamW(
        scalar_params + embeds_params + output_params,
        lr=0.003,
        betas=[0.9, 0.99],
        weight_decay=0.01,
    )

    loader = mnist_loader(train=True, batch_size=8192)
    for _ in tqdm.trange(1000):
        images, labels = next(loader)
        N = images.size(0)

        x = images.type(torch.bfloat16).flatten(1, -1) / 255.0
        x = 2 * x - 1

        labels = (labels + 1) * (torch.rand(N, device="cuda") > 0.8)
        emb = model.embed(labels)

        loss = model.flow.loss(x, emb).mean()
        loss.backward()
        muon.step()
        adamw.step()
        muon.zero_grad()
        adamw.zero_grad()

    (Path(__file__).parent / "data/guidance").mkdir(exist_ok=True)
    torch.save(model.state_dict(), Path(__file__).parent / "data/guidance/model.pt")


def generate():
    ckpt = torch.load(Path(__file__).parent / "data/guidance/model.pt")
    model = MnistFlow()
    model.cuda()
    model.load_state_dict(ckpt)

    labels = torch.arange(50, device="cuda") % 10
    c = model.embed(labels + 1)
    u = model.embed(torch.zeros_like(labels))

    kwargs = dict(
        noise_ratio=0.1,
        n_steps=32,
        minval=-1.0,
        maxval=1.0,
        use_heun=False,
        analytic_step=True,
    )

    imgs = model.flow.sample(c, **kwargs)
    imgs = (imgs + 1) / 2
    imgs = torch.round(imgs.clamp(0, 1) * 255).type(torch.uint8)
    imgs = imgs.cpu().numpy()
    imgs = rearrange(imgs, "(nh nw) (H W) -> (nh H) (nw W)", nh=5, nw=10, H=28, W=28)
    Image.fromarray(imgs).save(Path(__file__).parent / "data/guidance/lam=0.png")

    for lam in (1, 2, 4, 8, 16, 32):
        imgs = model.flow.sample(c, u=u, lam=float(lam), **kwargs)
        imgs = (imgs + 1) / 2
        imgs = torch.round(imgs.clamp(0, 1) * 255).type(torch.uint8)
        imgs = imgs.cpu().numpy()
        imgs = rearrange(
            imgs, "(nh nw) (H W) -> (nh H) (nw W)", nh=5, nw=10, H=28, W=28
        )
        Image.fromarray(imgs).save(
            Path(__file__).parent / f"data/guidance/lam={lam}.png"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
    else:
        generate()
