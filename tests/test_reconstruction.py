import numpy as np
import pytest

from occlusion.reconstruction import fit_plane


def test_ransac_recovers_tilted_plane_with_outliers():
    rng = np.random.default_rng(3)
    xy = rng.uniform(-1, 1, (300, 2))
    z = 2 + 0.2 * xy[:, 0] - 0.1 * xy[:, 1] + rng.normal(0, 0.001, 300)
    table = np.column_stack((xy, z))
    outliers = rng.uniform([-1, -1, 0.2], [1, 1, 1], (100, 3))
    points = np.concatenate((table, outliers))
    plane = fit_plane(points, 0.005)
    expected = np.array([0.2, -0.1, -1])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(plane.normal, expected, atol=0.001)
    assert plane.offset > 0
    assert plane.inliers[:300].mean() > 0.99
    assert not plane.inliers[300:].any()
    assert plane.signed_distance([[0, 0, 1]])[0] > 0


def test_normalized_relief_is_invariant_to_multiplicative_scale():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, (100, 2))
    points = np.column_stack((xy, np.full(100, 2)))
    plane = fit_plane(points, 0.01)
    scaled = fit_plane(points * 7, 0.07)
    test_points = np.array([[0.2, 0.1, 1.5], [0, 0, 2.1]])
    np.testing.assert_allclose(plane.signed_distance(test_points) / plane.offset,
                               scaled.signed_distance(test_points * 7) / scaled.offset, atol=1e-12)


@pytest.mark.parametrize("points", [np.zeros((5, 3)), np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3]])])
def test_degenerate_plane_input_is_rejected(points):
    with pytest.raises(ValueError, match="collinear"):
        fit_plane(points, 0.01)


def test_plane_rejects_nonfinite_points_and_bad_threshold():
    with pytest.raises(ValueError, match="finite"):
        fit_plane([[0, 0, 1], [1, 0, np.nan], [0, 1, 1]], 0.01)
    with pytest.raises(ValueError, match="threshold"):
        fit_plane([[0, 0, 1], [1, 0, 1], [0, 1, 1]], 0)
