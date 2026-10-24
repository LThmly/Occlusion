"""Lossless numeric depth and aligned RGB, plus a display-only color map."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


def colorize_depth(depth):
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], [2, 98])
        if high > low:
            scaled[valid] = (np.clip((depth[valid] - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored  # BGR for OpenCV display/writing


def write_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Cannot write image: {path}")


def new_capture_dir(root):
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def save_result(directory, original, result, metadata):
    write_rgb(directory / "rgb.png", original)
    write_rgb(directory / "rgb_processed.png", result.rgb)
    np.save(directory / "depth.npy", result.depth, allow_pickle=False)
    arrays = {name: getattr(result, name) for name in ("confidence", "intrinsics", "extrinsics")
              if getattr(result, name) is not None}
    np.savez_compressed(directory / "prediction.npz", **arrays)
    preview = np.hstack((cv2.cvtColor(result.rgb, cv2.COLOR_RGB2BGR), colorize_depth(result.depth)))
    if not cv2.imwrite(str(directory / "preview.png"), preview):
        raise OSError("Cannot write preview")
    details = dict(metadata, depth_units="relative", depth_scale="arbitrary_per_observation",
                   captured_at_utc=metadata.get("captured_at_utc", directory.name.split("-")[0]),
                   original_shape=list(original.shape), depth_shape=list(result.depth.shape),
                   inference_seconds=result.seconds,
                   intrinsics_source="model_prediction_not_calibration",
                   extrinsics_convention="OpenCV world-to-camera, model reference frame",
                   confidence_note="model score; not calibrated geometric uncertainty")
    (directory / "metadata.json").write_text(json.dumps(details, indent=2) + "\n")
    return preview
