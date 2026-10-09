"""Grad-CAM for CSRNet (Selvaraju et al., ICCV 2017): which image regions drove the count.

Target = the predicted count (sum of the density map); explained layer = the last conv of
CSRNet's back end. Written by hand because the pytorch-grad-cam package pulls in
opencv-python-headless, which replaces the GUI OpenCV the annotator window needs.
"""

from __future__ import annotations

import copy
import threading

import numpy as np
import torch
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module) -> None:
        # A private copy, so the hooks never run on the model serving live inference, and so
        # its in-place ReLUs (which break backward hooks) can be switched off.
        self.model = copy.deepcopy(model).eval()
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        for param in self.model.parameters():
            param.requires_grad_(False)  # gradients are needed for activations, not weights
        layer = [m for m in self.model.backend if isinstance(m, nn.Conv2d)][-1]
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)
        self._lock = threading.Lock()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _save_activation(self, module, inputs, output) -> None:
        self._activation = output.detach()

    def _save_gradient(self, module, grad_inputs, grad_outputs) -> None:
        self._gradient = grad_outputs[0].detach()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Heat map in [0, 1] at the density map's resolution (1/8 of the input) for one image."""
        with self._lock, torch.enable_grad():
            self.model(x.clone().requires_grad_(True)).sum().backward()
            weights = self._gradient.mean(dim=(2, 3), keepdim=True)  # importance of each channel
            cam = torch.relu((weights * self._activation).sum(dim=1))[0]
            peak = float(cam.max())
            return (cam / peak if peak > 0 else cam).cpu().numpy()
