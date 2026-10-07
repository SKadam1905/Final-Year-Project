"""PyTorch dataset over processed samples, with crowd-counting augmentation.

Training crops are aligned to the 8-pixel grid, so cropping and flipping the 1/8-resolution
density map stays exact and the crop's head count is simply the sum of its density.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.csrnet_model import OUTPUT_STRIDE, normalize_image
from app.image_io import read_image
from ml.dataset_index import Sample

PAD_BGR = (104, 116, 124)  # ImageNet mean colour: padding normalises to ~0


def _randint(high: int) -> int:
    """Uniform int in [0, high), drawn from torch's RNG (seeded per DataLoader worker)."""
    return int(torch.randint(0, high, ()).item())


def random_crop(image: np.ndarray, density: np.ndarray, crop: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid-aligned random crop; smaller images are padded to crop x crop."""
    h, w = image.shape[:2]
    ch, cw = min(crop, h), min(crop, w)
    y0 = OUTPUT_STRIDE * _randint((h - ch) // OUTPUT_STRIDE + 1)
    x0 = OUTPUT_STRIDE * _randint((w - cw) // OUTPUT_STRIDE + 1)
    image = image[y0 : y0 + ch, x0 : x0 + cw]
    s = OUTPUT_STRIDE
    density = density[y0 // s : (y0 + ch) // s, x0 // s : (x0 + cw) // s]
    if (ch, cw) != (crop, crop):
        image = cv2.copyMakeBorder(image, 0, crop - ch, 0, crop - cw, cv2.BORDER_CONSTANT, value=PAD_BGR)
        density = np.pad(density, ((0, (crop - ch) // s), (0, (crop - cw) // s)))
    return image, density


class CrowdDataset(Dataset):
    """Yields (image [3,H,W], density [1,H/8,W/8], count).

    `count` is the annotated head count for evaluation, or the crop's count in training.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        train: bool,
        crop_size: int = 384,
        hflip: bool = True,
        color_jitter: bool = True,
        grayscale_p: float = 0.1,
    ) -> None:
        if crop_size % OUTPUT_STRIDE:
            raise ValueError(f"crop_size must be a multiple of {OUTPUT_STRIDE}")
        self.samples = list(samples)
        self.train = train
        self.crop_size = crop_size
        self.hflip = hflip
        self.color_jitter = color_jitter
        self.grayscale_p = grayscale_p

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = read_image(sample.image)
        density = np.load(sample.density)
        h, w = image.shape[:2]
        if density.shape != (h // OUTPUT_STRIDE, w // OUTPUT_STRIDE) or h % OUTPUT_STRIDE or w % OUTPUT_STRIDE:
            raise ValueError(f"{sample.id}: image {w}x{h} and density {density.shape} disagree; "
                             "re-run ml.prepare_dataset")
        count = sample.count
        if self.train:
            image, density = self._augment(image, density)
            count = float(density.sum())
        return (
            torch.from_numpy(normalize_image(image)),
            torch.from_numpy(np.ascontiguousarray(density, dtype=np.float32))[None],
            torch.tensor(count, dtype=torch.float32),
        )

    def _augment(self, image: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, density = random_crop(image, density, self.crop_size)
        if self.hflip and torch.rand(()) < 0.5:
            image, density = image[:, ::-1], density[:, ::-1]
        if self.color_jitter:
            contrast = 0.8 + 0.4 * float(torch.rand(()))
            brightness = 25.0 * (2 * float(torch.rand(())) - 1)
            image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
        if self.grayscale_p and torch.rand(()) < self.grayscale_p:
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(image), np.ascontiguousarray(density)
