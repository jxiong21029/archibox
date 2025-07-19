import nvidia.dali as dali
from nvidia.dali.auto_aug import rand_augment


@dali.pipeline_def(enable_conditionals=True, enable_checkpointing=True, seed=0)
def imagenet_train_pipeline(num_shards, shard_id, randaug_n: int, randaug_m: int):
    images, labels = dali.fn.readers.file(
        file_root="/srv/datasets/ImageNet/train",
        random_shuffle=True,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    images = dali.fn.decoders.image_random_crop(
        images,
        output_type=dali.types.RGB,
        random_aspect_ratio=[0.75, 1.33],
        random_area=[0.08, 1.0],
        device="mixed",
    )
    images = dali.fn.resize(images, resize_x=224, resize_y=224)
    images = rand_augment.rand_augment(
        images,
        n=randaug_n,
        m=randaug_m,
        fill_value=None,  # edge pad
    )
    images = dali.fn.crop_mirror_normalize(
        images,
        dtype=dali.types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mirror=dali.fn.random.coin_flip(),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return images, labels.gpu()


@dali.pipeline_def(seed=0)
def imagenet_valid_pipeline(num_shards, shard_id):
    images, labels = dali.fn.readers.file(
        file_root="/srv/datasets/ImageNet/val", num_shards=num_shards, shard_id=shard_id
    )
    images = dali.fn.decoders.image(images, output_type=dali.types.RGB, device="mixed")
    images = dali.fn.resize(images, resize_shorter=256)
    images = dali.fn.crop_mirror_normalize(
        images.gpu(),
        dtype=dali.types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return images, labels.gpu()


def main():
    import torch
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    train_pipe = imagenet_train_pipeline(
        batch_size=1024,
        num_threads=4,
        device_id=0,
        num_shards=1,
        shard_id=0,
        randaug_n=2,
        randaug_m=10,
    )
    train_pipe.build()
    valid_pipe = imagenet_valid_pipeline(
        batch_size=1024, num_threads=4, device_id=0, num_shards=1, shard_id=0
    )
    valid_pipe.build()

    train_loader = DALIClassificationIterator(
        train_pipe, last_batch_policy=LastBatchPolicy.DROP, auto_reset=True
    )
    valid_loader = DALIClassificationIterator(
        valid_pipe, last_batch_policy=LastBatchPolicy.DROP, auto_reset=True
    )
    for batch in train_loader:
        images, labels = batch[0]["data"], batch[0]["label"].squeeze(-1)
        print(images.device, images.dtype, images.shape)
        print(labels.device, labels.dtype, labels.shape)
        break

    for batch in valid_loader:
        images, labels = batch[0]["data"], batch[0]["label"].squeeze(-1)
        print(images.device, images.dtype, images.shape)
        print(labels.device, labels.dtype, labels.shape)
        break


if __name__ == "__main__":
    main()
