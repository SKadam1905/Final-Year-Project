"""CSRNet crowd-density network (Li, Zhang & Chen, CVPR 2018) and an inference wrapper.

Front end: the first 10 conv layers of VGG-16 (ImageNet weights), output stride 8.
Back end:  6 dilated 3x3 convs (dilation 2), then a 1x1 conv to a 1-channel density map.
The predicted count is the sum of the density map.

Layer names match the authors' code (github.com/leeyeehoo/CSRNet-pytorch), so their released
ShanghaiTech checkpoints load with `load_weights` and give a baseline before your own
training finishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

FRONTEND_CFG = (64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512)
BACKEND_CFG = (512, 512, 512, 256, 128, 64)
OUTPUT_STRIDE = 8
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_layers(cfg, in_channels: int = 3, dilation: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=dilation, dilation=dilation),
                       nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, pretrained_frontend: bool = True) -> None:
        super().__init__()
        self.frontend = make_layers(FRONTEND_CFG)
        self.backend = make_layers(BACKEND_CFG, in_channels=512, dilation=2)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        for module in [*self.backend.modules(), self.output_layer]:
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
        if pretrained_frontend:
            from torchvision.models import VGG16_Weights, vgg16

            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.frontend.load_state_dict(vgg.features[: len(self.frontend)].state_dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.backend(self.frontend(x)))

    def set_frontend_trainable(self, trainable: bool) -> None:
        for param in self.frontend.parameters():
            param.requires_grad_(trainable)


def extract_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Model weights from our checkpoints ("model"), the authors' ("state_dict") or a bare dict."""
    state = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_weights(model: CSRNet, path: str | Path, allow_pickle: bool = False) -> dict:
    """Load weights into `model`; returns the full checkpoint dict.

    `allow_pickle=True` is needed for some old checkpoints (e.g. PyTorch 0.4 era files) that
    torch's safe loader rejects. Only use it for files from a source you trust.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=not allow_pickle)
    model.load_state_dict(extract_state_dict(checkpoint))
    return checkpoint


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def normalize_image(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (H, W, 3) -> ImageNet-normalised RGB float32 (3, H, W)."""
    rgb = image_bgr[..., ::-1].astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()


def stride_aligned_size(width: int, height: int, max_width: int | None) -> tuple[int, int]:
    """Input size (w, h): downscaled to `max_width` if wider, both sides multiples of 8."""
    scale = min(1.0, max_width / width) if max_width else 1.0

    def align(v: int) -> int:
        return max(OUTPUT_STRIDE, round(v * scale / OUTPUT_STRIDE) * OUTPUT_STRIDE)

    return align(width), align(height)


@dataclass(frozen=True)
class DensityResult:
    density: np.ndarray  # (h/8, w/8) float32 over the network input; sums to `count`
    count: float
    input_size: tuple[int, int]  # (w, h) fed to the network
    frame_size: tuple[int, int]  # (w, h) of the original frame


class DensityEstimator:
    """Runs CSRNet on single BGR frames (used by the AI worker and offline tools)."""

    def __init__(
        self,
        weights: str | Path | None,
        device: str = "auto",
        max_width: int | None = 1280,
        amp: bool = True,
        allow_pickle: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.model = CSRNet(pretrained_frontend=False)
        if weights is not None:
            load_weights(self.model, weights, allow_pickle=allow_pickle)
        self.model.to(self.device).eval()
        self.max_width = max_width
        self.amp = amp and self.device.type == "cuda"

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        h, w = frame_bgr.shape[:2]
        size = stride_aligned_size(w, h, self.max_width)
        if size != (w, h):
            interp = cv2.INTER_AREA if size[0] < w else cv2.INTER_LINEAR
            frame_bgr = cv2.resize(frame_bgr, size, interpolation=interp)
        return torch.from_numpy(normalize_image(frame_bgr)).unsqueeze(0)

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> DensityResult:
        x = self.preprocess(frame_bgr).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            out = self.model(x)
        density = out.float()[0, 0].cpu().numpy()
        h, w = frame_bgr.shape[:2]
        return DensityResult(density, float(density.sum()), (x.shape[3], x.shape[2]), (w, h))
