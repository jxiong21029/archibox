import logging
import random
from pathlib import Path

from torchvision.datasets import MNIST

log = logging.getLogger(__name__)


def mnist_loader(
    train: bool, batch_size: int, epochs: int | None = None, device="cuda"
):
    dataset_path = Path(__file__).parent / "data"
    dataset_path.mkdir(exist_ok=True)
    dataset = MNIST(dataset_path, train=True, download=True)
    images = dataset.data.to(device)
    labels = dataset.targets.to(device)

    idxs = list(range(len(images)))
    rng = random.Random(42)
    rng.shuffle(idxs)
    ptr = 0
    epoch = 0
    while True:
        if ptr + batch_size >= len(idxs):
            ptr = 0
            rng.shuffle(idxs)
            epoch += 1
            if epochs is not None and epoch >= epochs:
                break
            log.debug(f"starting {epoch=}")

        idx = idxs[ptr : ptr + batch_size]
        yield images[idx], labels[idx]
        ptr += batch_size
