"""Compare table/object relief from repeated inference on exactly the same RGB.

    python -m occlusion.quality --help
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .camera import read_rgb
from .geometry import backproject, resize_intrinsics
from .reconstruction import fit_plane


def region_mask(pixels, depth_size, reference_size, boxes):
    """Boxes are [left, top, right, bottom] in original-image coordinates.

    Use the same scale-only pixel convention as DA3's resized intrinsics.
    Regions are manual annotations, not semantic segmentation.
    """
    if not boxes:
        raise ValueError("At least one region box is required")
    width, height = reference_size
    pixel_x = pixels[:, 1] * width / depth_size[0]
    pixel_y = pixels[:, 0] * height / depth_size[1]
    selected = np.zeros(len(pixels), dtype=bool)
    for box in boxes:
        values = np.asarray(box, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("Region boxes must contain four finite coordinates")
        left, top, right, bottom = values
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("Region box lies outside its reference image or is empty")
        selected |= ((pixel_x >= left) & (pixel_x < right)
                     & (pixel_y >= top) & (pixel_y < bottom))
    return selected


def compare_observations(observations, regions, plane_threshold_fraction=0.02):
    """Hold camera rays fixed to the first observation's predicted intrinsics.

    Independent model intrinsics otherwise confound a depth-resolution test.
    Normalize signed distances by camera-to-plane distance, removing global
    multiplicative depth scale. No metric accuracy is inferred.
    """
    if not observations:
        raise ValueError("At least one saved observation is required")
    if not np.isfinite(plane_threshold_fraction) or not 0 < plane_threshold_fraction < 1:
        raise ValueError("Plane threshold fraction must lie between zero and one")
    observations = [Path(path) for path in observations]
    reference = read_rgb(observations[0] / "rgb.png")
    original_size = (reference.shape[1], reference.shape[0])
    if list(original_size) != regions["reference_image_size"]:
        raise ValueError("Region annotations do not match the original image dimensions")
    with np.load(observations[0] / "prediction.npz", allow_pickle=False) as prediction:
        reference_depth = np.load(observations[0] / "depth.npy", allow_pickle=False)
        matrix_original = resize_intrinsics(prediction["intrinsics"],
                                             (reference_depth.shape[1], reference_depth.shape[0]),
                                             original_size)
    results, maps = [], []
    for observation in observations:
        rgb = read_rgb(observation / "rgb.png")
        if not np.array_equal(rgb, reference):
            raise ValueError("Comparison requires exactly the same original RGB pixels; rerun inference on a saved image")
        metadata = json.loads((observation / "metadata.json").read_text())
        if metadata.get("depth_units") != "relative":
            raise ValueError("This diagnostic currently expects relative depth")
        if metadata.get("process_res_method", "upper_bound_resize") != "upper_bound_resize":
            raise ValueError("Only resize-without-crop observations are supported")
        depth = np.load(observation / "depth.npy", allow_pickle=False)
        matrix = resize_intrinsics(matrix_original, original_size, (depth.shape[1], depth.shape[0]))
        points, pixels = backproject(depth, matrix)
        table = region_mask(pixels, (depth.shape[1], depth.shape[0]), original_size, regions["table_boxes"])
        if table.sum() < 30:
            raise ValueError("Table regions contain fewer than 30 valid depth samples")
        typical_depth = float(np.median(points[table, 2]))
        # Bound fitting cost while sampling the table deterministically.
        table_points = points[table]
        if len(table_points) > 5000:
            table_points = table_points[np.random.default_rng(0).choice(len(table_points), 5000, replace=False)]
        plane = fit_plane(table_points, plane_threshold_fraction * typical_depth)
        if plane.inliers.mean() < 0.5:
            raise ValueError("No majority plane in the annotated table regions")
        if plane.offset <= 1e-6 * typical_depth:
            raise ValueError("Plane is too close to the camera origin for distance normalization")
        normalized = plane.signed_distance(points) / plane.offset
        table_residual = np.abs(normalized[table])
        objects = {}
        for name, boxes in regions["object_boxes"].items():
            selected = region_mask(pixels, (depth.shape[1], depth.shape[0]), original_size, boxes)
            if selected.sum() < 3:
                raise ValueError(f"Object region {name!r} contains fewer than three valid depth samples")
            objects[name] = {
                "valid_samples": int(selected.sum()),
                "normalized_signed_distance_p10_p50_p90": np.percentile(normalized[selected], [10, 50, 90]).tolist(),
                "fraction_above_table_p95_residual": float(np.mean(normalized[selected] > np.percentile(table_residual, 95))),
            }
        height_map = np.full(depth.shape, np.nan, dtype=np.float32)
        height_map[pixels[:, 0], pixels[:, 1]] = normalized
        maps.append(height_map)
        results.append({
            "observation": str(observation), "model": metadata.get("model"),
            "processing_resolution": metadata.get("process_resolution"),
            "depth_shape": list(depth.shape), "valid_depth_fraction": len(points) / depth.size,
            "recorded_inference_seconds": metadata.get("inference_seconds"),
            "plane_normal_camera": plane.normal.tolist(), "plane_offset_relative": plane.offset,
            "plane_fit_inlier_fraction": float(plane.inliers.mean()),
            "table_normalized_abs_residual_p50_p95": np.percentile(table_residual, [50, 95]).tolist(),
            "objects": objects,
        })
    return {
        "schema_version": 1, "original_rgb_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "reference_image_size": list(original_size),
        "intrinsics_source": "first_observation_model_prediction_shared_across_resolutions",
        "original_image_intrinsics": matrix_original.tolist(),
        "plane_threshold_fraction_of_median_depth": plane_threshold_fraction,
        "normalized_distance_definition": "signed_plane_distance / camera_to_plane_distance",
        "units": "dimensionless_not_metric_height", "scale_validated": False,
        "limitations": ["Manual regions, not object segmentation", "Uncalibrated intrinsics and lens distortion",
                        "Local shape diagnostic, not ground-truth accuracy or coverage",
                        "Same-image test only; no evidence of multi-view consistency"],
        "regions": regions, "observations": results,
    }, maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for JSON and normalized relief maps")
    parser.add_argument("--plane-threshold-fraction", type=float, default=0.02)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists; choose a new directory")
        regions = json.loads(args.regions.read_text())
        report, maps = compare_observations(args.observations, regions, args.plane_threshold_fraction)
        args.output.mkdir(parents=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(args.output / "relief.npz", **{f"observation_{i}": height for i, height in enumerate(maps)})
        for result in report["observations"]:
            objects = ", ".join(f"{name}={metrics['normalized_signed_distance_p10_p50_p90'][1]:.4f}"
                                for name, metrics in result["objects"].items())
            print(f"Resolution {result['processing_resolution']}: median normalized relief {objects}")
        print(f"Saved {args.output}; dimensionless diagnostics, not metric accuracy")
    except (OSError, ValueError, KeyError, IndexError) as error:
        parser.exit(1, f"Quality diagnostic: {error}\n")


if __name__ == "__main__":
    main()
