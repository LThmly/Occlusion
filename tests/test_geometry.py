import cv2
import numpy as np
import pytest

from occlusion.geometry import backproject, resize_intrinsics, transform_points


def test_backprojection_known_plane_and_invalid_pixels():
    depth = np.array([[2, 2, np.nan], [2, 0, -1]], dtype=float)
    matrix = np.array([[2, 0, 1], [0, 2, 0], [0, 0, 1]], dtype=float)
    points, pixels = backproject(depth, matrix)
    np.testing.assert_array_equal(pixels, [[0, 0], [0, 1], [1, 0]])
    np.testing.assert_allclose(points, [[-1, 0, 2], [0, 0, 2], [-1, 1, 2]])


def test_resized_intrinsics_preserve_axial_rays():
    matrix = np.array([[400, 0, 320], [0, 420, 240], [0, 0, 1]], dtype=float)
    resized = resize_intrinsics(matrix, (640, 480), (320, 240))
    np.testing.assert_allclose(np.linalg.solve(matrix, [120, 200, 1]),
                               np.linalg.solve(resized, [60, 100, 1]))


def test_distorted_pixels_backproject_to_correct_rays():
    matrix = np.array([[300, 0, 320], [0, 310, 240], [0, 0, 1]], dtype=float)
    distortion = np.array([0.15, -0.05, 0.001, 0.002, 0.01])
    depth = np.full((480, 640), 2.0)
    points, pixels = backproject(depth, matrix, distortion, stride=40)
    projected, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, distortion)
    np.testing.assert_allclose(projected.reshape(-1, 2), pixels[:, ::-1], atol=0.02)
    np.testing.assert_allclose(points[:, 2], 2)


def test_rigid_transform_composition_and_inverse():
    points = np.array([[1, 2, 3], [-2, 0, 4]], dtype=float)
    first = np.eye(4)
    first[:3, :3] = cv2.Rodrigues(np.array([0.2, -0.1, 0.4]))[0]
    first[:3, 3] = [1, 0, 2]
    second = np.eye(4)
    second[:3, 3] = [0, 2, -1]
    transformed = transform_points(points, first)
    np.testing.assert_allclose(transform_points(transformed, np.linalg.inv(first)), points, atol=1e-12)
    np.testing.assert_allclose(transform_points(transformed, second),
                               transform_points(points, second @ first))
    first[0, 0] *= 2
    with pytest.raises(ValueError, match="rigid"):
        transform_points(points, first)


def test_masks_empty_depth_and_stride():
    depth = np.ones((3, 3))
    mask = np.ones_like(depth, dtype=bool)
    mask[0, 0] = False
    points, pixels = backproject(depth, np.eye(3), stride=2, mask=mask)
    assert len(points) == 3
    assert [0, 0] not in pixels.tolist()
    assert backproject(np.zeros((3, 3)), np.eye(3))[0].shape == (0, 3)
    with pytest.raises(ValueError, match="Stride"):
        backproject(depth, np.eye(3), stride=0)
