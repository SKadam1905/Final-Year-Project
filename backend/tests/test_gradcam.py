import copy

import numpy as np
import torch
from torch import nn

from app.csrnet_model import CSRNet, to_input
from ml.gradcam import GradCAM
from tests.conftest import texture


def model():
    torch.manual_seed(0)
    return CSRNet(pretrained_frontend=False).eval()


def test_heat_map_is_normalised_at_density_resolution():
    cam = GradCAM(model())(to_input(np.dstack([texture(64, 96)] * 3), None))
    assert cam.shape == (8, 12)
    assert cam.min() >= 0.0 and cam.max() == 1.0


def test_matches_grad_cam_computed_directly():
    """The hooks must capture the last back-end conv's output and its gradient exactly."""
    live = model()
    x = torch.randn(1, 3, 32, 48)
    cam = GradCAM(live)(x)

    ref = copy.deepcopy(live)
    for m in ref.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    last_conv = max(i for i, m in enumerate(ref.backend) if isinstance(m, nn.Conv2d))
    h = ref.frontend(x)
    for i, layer in enumerate(ref.backend):
        h = layer(h)
        if i == last_conv:
            activation = h
            activation.retain_grad()
    ref.output_layer(h).sum().backward()
    weights = activation.grad.mean(dim=(2, 3), keepdim=True)
    expected = torch.relu((weights * activation).sum(dim=1))[0].detach()
    np.testing.assert_allclose(cam, (expected / expected.max()).numpy(), rtol=1e-4, atol=1e-6)


def test_the_live_model_is_left_untouched():
    live = model()
    before = {k: v.clone() for k, v in live.state_dict().items()}
    GradCAM(live)(torch.randn(1, 3, 32, 48))
    assert all(m.inplace for m in live.modules() if isinstance(m, nn.ReLU))
    assert all(p.grad is None for p in live.parameters())
    assert not any(m._forward_hooks or m._backward_hooks for m in live.modules())
    assert all(torch.equal(before[k], v) for k, v in live.state_dict().items())


def test_repeated_calls_are_deterministic():
    gradcam = GradCAM(model())
    x = torch.randn(1, 3, 32, 48)
    np.testing.assert_allclose(gradcam(x), gradcam(x))
