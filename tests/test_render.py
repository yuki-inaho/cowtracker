"""AllTracker-style splatting of point tracks onto frames."""

import numpy as np
import pytest

from cowtracker.toolkit.render import splat_tracks

T, H, W = 2, 20, 30
RED = np.array([[255, 0, 0]], dtype=np.uint8)


def _frames(value=100):
    return np.full((T, H, W, 3), value, dtype=np.uint8)


def _one_point(visible_at):
    tracks = np.array([[[10.0, 5.0], [20.0, 12.0]]], dtype=np.float32)  # (x, y) at t = 0, 1
    visible = np.array([visible_at], dtype=bool)
    return tracks, visible


def test_splat_draws_visible_points_only():
    tracks, visible = _one_point([True, False])
    frames = splat_tracks(_frames(), tracks, visible, RED, rate=2, bkg_opacity=1.0)

    assert frames.shape == (T, H, W, 3) and frames.dtype == np.uint8
    np.testing.assert_allclose(frames[0, 5, 10], [255, 0, 0], atol=1)  # accum / (weight + 1e-6) truncates
    assert frames[1, 12, 20].tolist() == [100, 100, 100]  # hidden at t = 1


@pytest.mark.parametrize(("opacity", "expected"), [(1.0, 100), (0.5, 50)])
def test_background_opacity_scales_the_frame(opacity, expected):
    tracks, visible = _one_point([False, False])
    frames = splat_tracks(_frames(), tracks, visible, RED, rate=2, bkg_opacity=opacity)

    assert frames[0, 0, 0].tolist() == [expected] * 3


def test_zero_opacity_blacks_out_and_boosts_saturation():
    tracks, visible = _one_point([True, True])
    colors = np.array([[200, 100, 100]], dtype=np.uint8)  # HSV saturation 127
    frames = splat_tracks(_frames(), tracks, visible, colors, rate=2, bkg_opacity=0.0)

    assert frames[0, 0, 0].tolist() == [0, 0, 0]
    red, green, _ = frames[0, 5, 10].tolist()
    assert abs(red - 200) <= 1 and green < 100  # saturation x1.5 as in AllTracker's demo
