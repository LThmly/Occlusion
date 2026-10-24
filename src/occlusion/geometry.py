"""Camera geometry: OpenCV axes (+x right, +y down, +z forward).

Depth is optical-axis Z, not Euclidean ray distance. Units follow input depth.
"""

import cv2
import numpy as np


def validate_intrinsics(intrinsics):
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.allclose([matrix[0, 1], matrix[1, 0]], 0)):
        raise ValueError("Expected finite zero-skew pinhole intrinsics with positive focal lengths")
    return matrix


def resize_intrinsics(intrinsics, source_size, target_size):
    """Sizes are (width, height); use DA3's scale-only pixel convention.

    Covers resize without crop. It must not be used for a different camera
    mode, digital zoom, or an unknown crop.
    """
    sizes = np.asarray([source_size, target_size], dtype=float)
    if sizes.shape != (2, 2) or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError("Image sizes must contain positive width and height")
    matrix = validate_intrinsics(intrinsics).copy()
    matrix[:2] *= (sizes[1] / sizes[0])[:, None]
    return matrix


def backproject(depth, intrinsics, distortion=None, stride=1, mask=None):
    """Return valid Nx3 camera points and their (row, column) pixel indices.

    For distorted RGB/depth, undistort pixel rays rather than resampling depth.
    Distortion coefficients retain their OpenCV normalized-coordinate meaning
    after resize. Color and depth must describe the same processed pixel grid.
    """
    depth = np.asarray(depth)
    matrix = validate_intrinsics(intrinsics)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise ValueError("Depth must be a numeric HxW array")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("Stride must be a positive integer")
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        if np.asarray(mask).shape != depth.shape:
            raise ValueError("Mask shape must match depth")
        valid &= np.asarray(mask, dtype=bool)
    rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    selected = valid[rows, cols]
    pixels = np.column_stack((rows[selected], cols[selected]))
    if not len(pixels):
        return np.empty((0, 3)), pixels
    uv = pixels[:, ::-1].astype(np.float64).reshape(-1, 1, 2)
    rays_xy = cv2.undistortPoints(uv, matrix, distortion).reshape(-1, 2)
    rays = np.column_stack((rays_xy, np.ones(len(pixels))))
    points = rays * depth[pixels[:, 0], pixels[:, 1], None]
    return points, pixels


def transform_points(points, transform):
    """Apply T_A_B to Nx3 points in B, returning points in A; no scale change."""
    points = np.asarray(points, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be finite Nx3 coordinates")
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1)):
        raise ValueError("Transform must be a rigid 4x4 matrix with a proper rotation")
    return points @ transform[:3, :3].T + transform[:3, 3]
