"""Geometry and capture replay checks for the dual-camera viewer."""

import numpy as np
import pytest

from occlusion.visualizer import load_capture, project_cloud
from occlusion.depth import DepthResult
from occlusion.outputs import save_result


def test_projection_filters_scales_and_transforms():
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.array([[1, 0, np.nan], [2, np.inf, -1]])
    pose = np.eye(4)
    pose[:3, 3] = [10, 20, 30]
    points, colors = project_cloud(rgb, depth, np.eye(3), pose, scale=2, stride=1)
    np.testing.assert_allclose(points, [[10, 20, 32], [10, 24, 34]])
    np.testing.assert_array_equal(colors, rgb[[0, 1], [0, 0]])
    clipped, _ = project_cloud(rgb, depth, np.eye(3), pose, stride=1, max_depth=1)
    assert clipped.shape == (1, 3)


def test_clouds_overlay_with_known_camera_translation():
    rgb = np.zeros((1, 2, 3), dtype=np.uint8)
    pose = np.eye(4)
    pose[0, 3] = -1
    first, _ = project_cloud(rgb, np.array([[1, 0]]), np.eye(3), np.eye(4), stride=1)
    second, _ = project_cloud(rgb, np.array([[0, 1]]), np.eye(3), pose, stride=1)
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize('kind', ['intrinsics', 'pose', 'alignment', 'scale'])
def test_invalid_geometry_rejected(kind):
    rgb, depth, k, pose = np.zeros((2, 2, 3)), np.ones((2, 2)), np.eye(3), np.eye(4)
    if kind == 'intrinsics':
        k[0, 0] = 0
    if kind == 'pose':
        pose[0, 0] = 2
    if kind == 'alignment':
        rgb = rgb[:1]
    with pytest.raises(ValueError):
        project_cloud(rgb, depth, k, pose, scale=0 if kind == 'scale' else 1)


def test_replay_uses_processed_rgb(tmp_path):
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    result = DepthResult(original[:2, :2], np.ones((2, 2)), None, np.eye(3), None, 0)
    save_result(tmp_path, original, result, {})
    loaded = load_capture(tmp_path)
    np.testing.assert_array_equal(loaded.rgb, result.rgb)
    np.testing.assert_array_equal(loaded.depth, result.depth)
    np.testing.assert_array_equal(loaded.intrinsics, result.intrinsics)
