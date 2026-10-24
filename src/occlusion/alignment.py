"""Provisional two-camera alignment from manually matched RGB/depth pixels.

No physical motion. Fit wrist points into overhead-camera relative units using
q = scale * R @ p + t. This is not metric camera calibration or time sync.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import validate_intrinsics
from .visualizer import CAMERAS, load_capture


def fit_similarity(source, target):
    """Umeyama least-squares positive scale and proper (nonreflecting) rotation.

    Row arrays satisfy target ~= scale * source @ rotation.T + translation.
    Noncollinear planar points are supported; nearly collinear sets are rejected.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (source.ndim != 2 or source.shape[1] != 3 or source.shape != target.shape
            or len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Similarity fitting requires matching finite Nx3 arrays with at least three points")
    x, y = source - source.mean(axis=0), target - target.mean(axis=0)
    for centered in (x, y):
        singular = np.linalg.svd(centered, compute_uv=False)
        if singular[0] == 0 or singular[1] <= 0.01 * singular[0]:
            raise ValueError("Matched points are coincident or nearly collinear; spread points over the scene")
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.ones(3)
    signs[-1] = 1 if np.linalg.det(u @ vt) > 0 else -1
    rotation = (u * signs) @ vt
    scale = float((singular @ signs) / np.mean(np.sum(x*x, axis=1)))
    translation = target.mean(axis=0) - scale * rotation @ source.mean(axis=0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("No positive similarity scale could be fitted")
    return scale, rotation, translation


def align_correspondences(source, target, inlier_fraction=0.05, validation_fraction=0.10):
    """Fit only training pairs; reserve every third pair for independent checks.

    Thresholds are fractions of the training target's maximum pairwise distance.
    They are provisional overlay tolerances, not centimeter accuracy claims.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3
            or len(source) < 9 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Select at least nine matching finite 3D point pairs (twelve recommended)")
    if not (0 < inlier_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Alignment error fractions must lie between zero and one")
    validation = np.arange(len(source))[::3]
    training = np.array([i for i in range(len(source)) if i not in validation])
    sx, ty = source[training], target[training]
    fit_similarity(sx, ty)  # Reject degenerate overall geometry before RANSAC.
    extent = float(np.max(np.linalg.norm(ty[:, None] - ty[None, :], axis=2)))
    rng = np.random.default_rng(0)
    best, best_score = None, (-1, -np.inf)
    for _ in range(256):
        selected = rng.choice(len(training), 3, replace=False)
        try:
            scale, rotation, translation = fit_similarity(sx[selected], ty[selected])
        except ValueError:
            continue
        residuals = np.linalg.norm(scale * sx @ rotation.T + translation - ty, axis=1)
        inliers = residuals <= inlier_fraction * extent
        if inliers.any():
            score = (int(inliers.sum()), -float(np.median(residuals[inliers])))
            if score > best_score:
                best, best_score = inliers, score
    required = max(6, int(np.ceil(0.75 * len(training))))
    if best is None or best.sum() < 3:
        raise ValueError("No nondegenerate correspondence subset supports a diagnostic fit")
    scale, rotation, translation = fit_similarity(sx[best], ty[best])
    residuals = np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1)
    training_inliers = training[best]
    training_passed = bool(best.sum() >= required
                           and np.max(residuals[training_inliers]) <= inlier_fraction * extent)
    validation_passed = bool(np.max(residuals[validation]) <= validation_fraction * extent)
    accepted = training_passed and validation_passed
    failures = []
    if not training_passed:
        failures.append(f"Training support failed: {int(best.sum())}/{len(training)} consensus points; "
                        f"{required} required, with all fitted inliers inside the threshold")
    if not validation_passed:
        failures.append("Held-out alignment checks failed")
    warnings = []
    for label, points in (("wrist", sx[best]), ("overhead", ty[best])):
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if singular[-1] < 0.02 * singular[0]:
            warnings.append(f"{label}: matches are nearly planar; check additional shared features at different heights")
    return scale, rotation, translation, {
        "accepted": accepted, "training_indices_zero_based": training.tolist(),
        "training_support_passed": training_passed, "validation_passed": validation_passed,
        "required_training_inliers": required, "failure_reasons": failures,
        "training_inlier_indices_zero_based": training_inliers.tolist(),
        "validation_indices_zero_based": validation.tolist(),
        "per_pair_errors_relative": residuals.tolist(),
        "per_pair_errors_normalized": (residuals / extent).tolist(),
        "training_target_extent_relative": extent,
        "training_rms_normalized": float(np.sqrt(np.mean((residuals[training_inliers] / extent)**2))),
        "validation_max_error_normalized": float(np.max(residuals[validation]) / extent),
        "inlier_threshold_fraction": inlier_fraction,
        "validation_threshold_fraction": validation_fraction,
        "warnings": warnings,
        "note": "Fractions of matched training-point span; not metric accuracy or exposure synchronization",
    }


def pair_identity(pair):
    """Bind clicked pixels to exact saved RGB, depth, and predicted intrinsics."""
    return {name: {filename: hashlib.sha256((Path(pair) / name / filename).read_bytes()).hexdigest()
                   for filename in ("rgb_processed.png", "depth.npy", "prediction.npz")}
            for name in CAMERAS}


def load_pair(pair):
    results = {name: load_capture(Path(pair) / name) for name in CAMERAS}
    for name, result in results.items():
        validate_intrinsics(result.intrinsics)
        if result.depth.ndim != 2 or result.rgb.shape != (*result.depth.shape, 3):
            raise ValueError(f"{name}: processed RGB and depth are not aligned")
        metadata = json.loads((Path(pair) / name / "metadata.json").read_text())
        if metadata.get("depth_units") != "relative":
            raise ValueError("This alignment workflow currently expects relative-depth captures")
    return results


def sample_point(result, pixel):
    """Use one exact processed pixel: no smoothing across depth boundaries."""
    uv = np.asarray(pixel, dtype=float)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError("Pixel must contain finite [column, row] coordinates")
    if not (0 <= uv[0] < result.depth.shape[1] and 0 <= uv[1] < result.depth.shape[0]):
        raise ValueError("Pixel lies outside the processed depth image")
    u, v = np.floor(uv).astype(int)
    z = float(result.depth[v, u])
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Selected pixel has invalid depth; choose a nearby identifiable surface point")
    return np.linalg.solve(validate_intrinsics(result.intrinsics), [u, v, 1]) * z


def fit_pair(pair, matches_path, output, inlier_fraction=0.05, validation_fraction=0.10):
    pair, output = Path(pair), Path(output)
    report_path = output.with_suffix(".report.json")
    if output.suffix != ".json" or output.exists() or report_path.exists():
        raise ValueError("Choose a new .json output; existing geometry/reports are not overwritten")
    results = load_pair(pair)
    matches = json.loads(Path(matches_path).read_text())
    if matches.get("schema_version") != 1 or matches.get("capture_identity") != pair_identity(pair):
        raise ValueError("Matches do not belong to this exact saved pair; select points again")
    coordinates = {}
    for name in CAMERAS:
        pixels = [item[name] for item in matches["pairs"]]
        coordinates[name] = np.array([sample_point(results[name], pixel) for pixel in pixels])
        if len({tuple(np.floor(p).astype(int)) for p in pixels}) != len(pixels):
            raise ValueError(f"{name}: the same pixel was selected more than once")
    scale, rotation, translation, report = align_correspondences(
        coordinates["wrist"], coordinates["overhead"], inlier_fraction, validation_fraction)
    report.update(pair=str(pair), matches=str(matches_path), capture_identity=matches["capture_identity"],
                  mode="manual_similarity_uncalibrated", scale_validated=False,
                  units="overhead_observation_relative_units", wrist_depth_scale=scale,
                  wrist_rotation=rotation.tolist(), wrist_translation=translation.tolist())
    # This secondary image-space check helps diagnose whether clicks are
    # consistent with one planar face. It does not participate in 3D acceptance.
    pixels = {name: np.array([item[name] for item in matches["pairs"]], dtype=float) for name in CAMERAS}
    homography, inliers = cv2.findHomography(pixels["wrist"], pixels["overhead"], cv2.RANSAC, 3)
    if homography is not None:
        projected = cv2.perspectiveTransform(pixels["wrist"][None], homography)[0]
        errors = np.linalg.norm(projected - pixels["overhead"], axis=1)
        report["planar_image_diagnostic"] = {
            "median_overhead_pixel_error": float(np.median(errors)),
            "per_pair_overhead_pixel_errors": errors.tolist(),
            "inlier_indices_zero_based": np.flatnonzero(inliers.ravel()).tolist(),
            "note": "All-match planar image fit for diagnosis only; not camera calibration or 3D validation",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["accepted"]:
        reasons = "; ".join(report["failure_reasons"])
        raise ValueError(f"{reasons}; inspect {report_path}. No geometry file written")
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation
    geometry = {name: {
        "intrinsics": np.asarray(results[name].intrinsics).tolist(),
        "image_size": [results[name].depth.shape[1], results[name].depth.shape[0]],
        "world_from_camera": (pose if name == "wrist" else np.eye(4)).tolist(),
        "depth_scale": scale if name == "wrist" else 1.0,
    } for name in CAMERAS}
    geometry["_alignment"] = {
        "mode": "manual_similarity_uncalibrated", "reference_frame": "overhead_camera",
        "units": "relative_not_meters", "report": str(report_path),
        "capture_identity": matches["capture_identity"], "scale_validated": False,
        "note": "Validated only on the saved stationary pair; wrist movement or new depth scale invalidates alignment",
    }
    output.write_text(json.dumps(geometry, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Save one stationary pair before sequential depth inference")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--wrist-camera", default="/dev/video2")
    capture.add_argument("--overhead-camera", default="/dev/video0")
    capture.add_argument("--resolution", type=int, default=504)
    capture.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    capture.add_argument("--warmup-frames", type=int, default=30)
    pick = commands.add_parser("pick", help="Click matching points in saved, processed RGB images")
    pick.add_argument("--pair", type=Path, required=True)
    pick.add_argument("--output", type=Path, help="Default: PAIR/matches.json")
    fit = commands.add_parser("fit", help="Fit wrist-to-overhead similarity and validate held-out points")
    fit.add_argument("--pair", type=Path, required=True)
    fit.add_argument("--matches", type=Path, help="Default: PAIR/matches.json")
    fit.add_argument("--output", type=Path, default=Path("configs/geometry.json"))
    fit.add_argument("--inlier-fraction", type=float, default=0.05)
    fit.add_argument("--validation-fraction", type=float, default=0.10)
    args = parser.parse_args()
    try:
        if args.command == "capture":
            if args.warmup_frames < 1 or args.resolution < 28 or args.resolution % 14:
                raise ValueError("Warmup must be positive; resolution must be a multiple of 14 >=28")
            from .alignment_ui import capture_pair
            capture_pair(args)
        elif args.command == "pick":
            from .alignment_ui import pick_matches
            pick_matches(args.pair, args.output or args.pair / "matches.json")
        else:
            report = fit_pair(args.pair, args.matches or args.pair / "matches.json", args.output,
                              args.inlier_fraction, args.validation_fraction)
            print(f"Saved {args.output}; wrist scale {report['wrist_depth_scale']:.5f}; "
                  f"held-out max error {100*report['validation_max_error_normalized']:.2f}% of matched span")
            for warning in report["warnings"]:
                print(f"Warning: {warning}")
            print("Provisional alignment only; replay the exact saved pair before trying live observations")
    except (OSError, ValueError, KeyError, TypeError, ImportError, cv2.error) as error:
        parser.exit(1, f"Alignment: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Alignment interrupted\n")


if __name__ == "__main__":
    main()
