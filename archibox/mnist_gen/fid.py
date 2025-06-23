import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

from archibox.components import FusedLinear, RMSNorm
from archibox.mnist_gen.dataloading import mnist_loader

log = logging.getLogger(__name__)


class MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.k = FusedLinear(dim, mlp_dim)
        self.act = nn.SiLU()
        self.v = FusedLinear(mlp_dim, dim, scale=True, zero_init=True)

    def forward(self, x):
        scores = self.k(self.norm(x))
        scores = self.act(scores)
        return self.v(scores)


class MLPClassifier(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, depth: int, n_classes: int = 10):
        super().__init__()
        self.in_proj = FusedLinear(28 * 28, dim)
        self.blocks = nn.ModuleList([MLPBlock(dim, mlp_dim) for _ in range(depth)])
        self.out_norm = RMSNorm(dim, affine=False)
        self.out_head = FusedLinear(dim, n_classes, zero_init=True)
        self.out_head.weight._is_output = True

    def encode(self, images_ND):
        x_ND = self.in_proj(images_ND)
        for block in self.blocks:
            x_ND = x_ND + block(x_ND)
        x_ND = self.out_norm(x_ND)
        return x_ND

    def loss(self, images_ND, labels_ND):
        features = self.encode(images_ND)
        logits = self.out_head(features)
        return F.cross_entropy(logits, labels_ND)


class FID:
    def __init__(self, dim: int, mlp_dim: int, depth: int):
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.depth = depth

        self.model = MLPClassifier(dim, mlp_dim, depth)

    @classmethod
    def from_pretrained(cls, path="archibox/mnist_gen/data/classifier.pt"):
        ckpt = torch.load(path)
        fid = cls(**ckpt["config"])
        fid.model.load_state_dict(ckpt["classifier"])
        return fid

    def save(self, path):
        ckpt = dict(config=dict(dim=self.dim, mlp_dim=self.mlp_dim, depth=self.depth))
        ckpt["classifier"] = self.model.state_dict()
        torch.save(ckpt, path)

    @torch.no_grad
    def compute_fid(self, samples_ND, eps=1e-6):
        N, D = samples_ND.shape
        assert torch.is_floating_point(samples_ND)
        assert D == 28 * 28
        if samples_ND.amax() > 1.0:
            log.warning(
                "samples seem improperly normalized; should be normalized to [-1, 1], "
                "but found values larger than 1.0"
            )
        elif samples_ND.amin() < -1.0:
            log.warning(
                "samples seem improperly normalized; should be normalized to [-1, 1], "
                "but found values less than -1.0"
            )
        elif samples_ND.amin() >= 0:
            log.warning(
                "samples seem improperly normalized; should be normalized to [-1, 1], "
                "but found no values less than 0.0"
            )

        feats_fake = []
        for i in range(len(samples_ND) // 1024):
            x_ND = samples_ND[i : i + 1024].bfloat16()
            feats_fake.append(self.model.encode(x_ND))
        feats_fake_ND = torch.cat(feats_fake).float().cpu().numpy()

        feats_real = []
        for images, _ in mnist_loader(train=True, batch_size=1024, epochs=1):
            x_ND = images.bfloat16().flatten(1, -1) / 255.0 * 2.0 - 1.0
            feats_real.append(self.model.encode(x_ND))
        feats_real_ND = torch.cat(feats_real).float().cpu().numpy()

        mu1 = np.mean(feats_fake_ND, axis=0)
        mu2 = np.mean(feats_real_ND, axis=0)
        cov1 = np.cov(feats_fake_ND, rowvar=False)
        cov2 = np.cov(feats_real_ND, rowvar=False)

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
        if not np.isfinite(covmean).all():
            print(
                f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
            )
            offset = np.eye(cov1.shape[0]) * eps
            covmean = linalg.sqrtm((cov1.numpy() + offset).dot(cov2.numpy() + offset))
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        return diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean)


def main():
    """train and save MLP classifier"""

    N_STEPS = 1500
    VALID_EVERY = 250

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    fid = FID(dim=256, mlp_dim=1024, depth=4)
    fid.model.to("cuda")

    optim = torch.optim.AdamW(fid.model.parameters(), lr=1e-3)

    loss_sum = 0
    count = 0
    for step, (images, labels) in enumerate(mnist_loader(train=True, batch_size=1024)):
        if step == N_STEPS:
            break

        dx = torch.randint(-2, 3, ())
        dy = torch.randint(-2, 3, ())
        images = F.pad(images, (dx, -dx, dy, -dy))
        x_ND = images.type(torch.bfloat16).flatten(1, -1) / 255.0 * 2.0 - 1.0
        loss = fid.model.loss(x_ND, labels)

        loss_sum += loss.detach()
        count += 1

        loss.backward()
        optim.step()
        optim.zero_grad()

        if (step + 1) % VALID_EVERY == 0:
            valid_loss_sum = 0
            valid_count = 0
            with torch.no_grad():
                for images, labels in mnist_loader(
                    train=False, batch_size=1024, epochs=1
                ):
                    x_ND = (
                        images.type(torch.bfloat16).flatten(1, -1) / 255.0 * 2.0 - 1.0
                    )
                    loss = fid.model.loss(x_ND, labels)
                    valid_loss_sum += loss
                    valid_count += 1

            print(
                f"[{step + 1: >4}/{N_STEPS}] train_loss={loss_sum / count:.4f}, "
                f"valid_loss={valid_loss_sum / valid_count:.4f}"
            )
            loss_sum = 0
            count = 0

    savepath = Path("archibox/mnist_gen/data/classifier.pt")
    fid.save(savepath)


if __name__ == "__main__":
    main()
