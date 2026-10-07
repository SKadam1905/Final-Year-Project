import numpy as np
import pytest

from ml.density_maps import adaptive_sigmas, density_map, sum_pool


def test_density_sums_to_head_count(rng):
    points = rng.uniform([0, 0], [120, 100], (300, 2))
    assert density_map(points, (100, 120)).sum() == pytest.approx(300, abs=1e-2)


def test_heads_on_or_past_the_border_still_count_once():
    points = [[0, 0], [119.99, 99.99], [125, 50], [-3, 10]]
    assert density_map(points, (100, 120)).sum() == pytest.approx(4, abs=1e-4)


def test_empty_annotation_gives_empty_map():
    dmap = density_map(np.zeros((0, 2)), (16, 24))
    assert dmap.shape == (16, 24) and not dmap.any()


def test_crowded_heads_get_narrower_kernels():
    cluster = [[50 + i, 50] for i in range(0, 8, 2)]
    isolated = [[5, 5], [95, 5], [5, 95]]
    sigmas = adaptive_sigmas(np.array(cluster + isolated, float))
    assert sigmas[:4].max() < sigmas[4:].min()


def test_single_head_uses_fallback_sigma():
    assert adaptive_sigmas(np.array([[10.0, 10.0]]), fallback=15.0).tolist() == [15.0]


def test_fixed_sigma():
    # Pixel (row 20, col 20) covers [20, 21) x [20, 21); its centre is (20.5, 20.5).
    dmap = density_map([[20.5, 20.5]], (40, 40), sigma=4.0)
    assert dmap.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.unravel_index(dmap.argmax(), dmap.shape) == (20, 20)
    with pytest.raises(ValueError):
        density_map([[20.5, 20.5]], (40, 40), sigma=0.0)


def test_sum_pool_preserves_total_and_shape(rng):
    dmap = density_map(rng.uniform(0, 64, (50, 2)), (64, 96))
    pooled = sum_pool(dmap, 8)
    assert pooled.shape == (8, 12)
    assert pooled.sum() == pytest.approx(dmap.sum(), abs=1e-4)
    with pytest.raises(ValueError):
        sum_pool(np.zeros((10, 16)), 8)
