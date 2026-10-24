"""Export a saved observation as a colored, relative-unit camera-frame PLY."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .calibration import load_calibration
from .camera import read_rgb
from .geometry import backproject, resize_intrinsics


def export_observation(observation, output, calibration=None, camera_id=None,
                       predicted_intrinsics=False, stride=2):
    observation, output = Path(observation), Path(output)
    if (calibration is None) == (not predicted_intrinsics):
        raise ValueError("Choose calibrated intrinsics or explicitly opt into predicted intrinsics")
    if output.suffix.lower() != ".ply":
        raise ValueError("Output must have a .ply extension")
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Output already exists; choose a new filename")
    metadata = json.loads((observation / "metadata.json").read_text())
    if metadata.get("depth_units") != "relative":
        raise ValueError("This exporter currently supports relative-depth observations only")
    if metadata.get("process_res_method", "upper_bound_resize") != "upper_bound_resize":
        raise ValueError("Unsupported preprocessing; exporter expects DA3 upper_bound_resize without crop")
    depth = np.load(observation / "depth.npy", allow_pickle=False)
    rgb = read_rgb(observation / "rgb_processed.png")
    original = read_rgb(observation / "rgb.png")
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError("Processed RGB must align with the HxW depth map")
    if list(original.shape) != metadata.get("original_shape"):
        raise ValueError("Original image dimensions disagree with observation metadata")
    if calibration is not None:
        record = load_calibration(calibration)
        if not camera_id or record["camera_id"] != camera_id:
            raise ValueError("Specify --camera-id matching the calibration camera identity")
        source_size = (original.shape[1], original.shape[0])
        if list(source_size) != record["image_size"]:
            raise ValueError("Calibration resolution differs from capture; calibrate this camera mode")
        # Current DA3 adapter uses upper_bound_resize with no crop. Do not use
        # this path for observations produced by a different preprocessing mode.
        matrix = resize_intrinsics(record["intrinsics"], source_size,
                                   (depth.shape[1], depth.shape[0]))
        distortion = np.asarray(record["distortion"], dtype=float)
        provenance = {"intrinsics_source": "checkerboard_calibration",
                      "camera_id": camera_id, "calibration_path": str(calibration),
                      "calibration_sha256": hashlib.sha256(Path(calibration).read_bytes()).hexdigest()}
    else:
        with np.load(observation / "prediction.npz", allow_pickle=False) as prediction:
            if "intrinsics" not in prediction:
                raise ValueError("Observation has no predicted intrinsics")
            matrix = prediction["intrinsics"]
        distortion = None
        provenance = {"intrinsics_source": "model_prediction_not_calibration",
                      "warning": "Diagnostic only: intrinsics unvalidated and lens distortion uncorrected"}
    points, pixels = backproject(depth, matrix, distortion, stride=stride)
    if not len(points):
        raise ValueError("No valid depth points to export")
    colors = rgb[pixels[:, 0], pixels[:, 1]]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment units relative_not_meters\n"
                     "comment frame camera_opencv_x_right_y_down_z_forward\n"
                     f"element vertex {len(points)}\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(stream, np.column_stack((points, colors)), fmt="%.8g %.8g %.8g %d %d %d")
    details = dict(provenance, observation=str(observation), units="relative_not_meters",
                   frame="camera_opencv", depth_definition="optical_axis_z",
                   point_count=len(points), stride=stride,
                   processed_intrinsics=np.asarray(matrix).tolist(),
                   distortion_corrected_in_rays=distortion is not None,
                   scale_validated=False, suitable_for_robot_control=False)
    output.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    intrinsics = parser.add_mutually_exclusive_group(required=True)
    intrinsics.add_argument("--calibration", type=Path)
    intrinsics.add_argument("--predicted-intrinsics", action="store_true",
                            help="Diagnostic only; ignores uncalibrated lens distortion")
    parser.add_argument("--camera-id", help="Required with --calibration; user asserts observation identity")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    try:
        details = export_observation(args.observation, args.output, args.calibration,
                                     args.camera_id, args.predicted_intrinsics, args.stride)
        print(f"Saved {details['point_count']} points to {args.output}; RELATIVE units, not meters")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Point cloud: {error}\n")


if __name__ == "__main__":
    main()
