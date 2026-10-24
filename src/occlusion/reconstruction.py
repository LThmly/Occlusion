"""Small geometric primitives; multi-view fusion is not implemented yet."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    normal: np.ndarray  # Unit vector, oriented toward the camera origin.
    offset: float  # normal @ point + offset = 0; units follow input points.
    inliers: np.ndarray

    def signed_distance(self, points):
        """Positive on the camera-facing side of the plane."""
        return np.asarray(points) @ self.normal + self.offset


def fit_plane(points, distance_threshold, iterations=128, seed=0):
    """Seeded three-point RANSAC, followed by least-squares inlier refinement.

    Fit only a user-selected table region; this does not discover the table or
    certify free space. The threshold has the same units as the input points.
    """
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
            or not np.isfinite(points).all()):
        raise ValueError("Plane fitting needs at least three finite 3D points")
    if not np.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("Plane distance threshold must be positive and finite")
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("RANSAC iterations must be a positive integer")
    rng = np.random.default_rng(seed)
    best = None
    best_score = (-1, -np.inf)
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        edge1, edge2 = b - a, c - a
        normal = np.cross(edge1, edge2)
        length = np.linalg.norm(normal)
        if length <= 1e-10 * np.linalg.norm(edge1) * np.linalg.norm(edge2):
            continue
        normal /= length
        distances = np.abs((points - a) @ normal)
        inliers = distances <= distance_threshold
        score = (int(inliers.sum()), -float(np.median(distances[inliers])))
        if score > best_score:
            best, best_score = inliers, score
    if best is None or best.sum() < 3:
        raise ValueError("Points are coincident or collinear; no plane can be fitted")
    for _ in range(3):
        selected = points[best]
        center = selected.mean(axis=0)
        _, singular, vectors = np.linalg.svd(selected - center, full_matrices=False)
        if singular[1] <= 1e-10 * singular[0]:
            raise ValueError("Plane inliers are collinear")
        normal = vectors[-1]
        offset = -float(center @ normal)
        updated = np.abs(points @ normal + offset) <= distance_threshold
        if updated.sum() < 3:
            raise ValueError("Too few plane inliers after refinement")
        if np.array_equal(best, updated):
            break
        best = updated
    # Orient the normal so the camera origin lies on the positive side.
    if offset < 0:
        normal, offset = -normal, -offset
    return Plane(normal, offset, np.abs(points @ normal + offset) <= distance_threshold)
