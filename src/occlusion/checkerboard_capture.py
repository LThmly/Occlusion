"""Live checkerboard feedback for a stationary two-camera calibration session."""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from .calibration import board_points
from .camera import USBCamera, receive_pair
from .outputs import new_capture_dir, write_rgb

CAMERAS = ("wrist", "overhead")


def detect_board(rgb, columns, rows):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, (columns, rows), cv2.CALIB_CB_NORMALIZE_IMAGE)
    return corners.astype(np.float32) if found else None


def save_board_pair(root, frames, corners, timing, columns, rows, seen):
    """Save raw images, never detection overlays; both boards must be found.

    Layout preserves pair identifiers while letting existing single-camera
    intrinsic fitting read root/wrist and root/overhead separately.
    """
    if any(corners[name] is None for name in CAMERAS):
        raise ValueError("Pattern must be detected in BOTH cameras before saving")
    if any(np.asarray(corners[name]).shape != (columns*rows,1,2)
           or not np.isfinite(corners[name]).all() for name in CAMERAS):
        raise ValueError("Expected a complete finite corner grid in each camera")
    fingerprints = {name: hashlib.sha256(frames[name].tobytes()).hexdigest() for name in CAMERAS}
    if any(fingerprints[name] in seen[name] for name in CAMERAS):
        raise ValueError("An identical image was already saved; change the iPad pose")
    directory = new_capture_dir(Path(root) / "pairs")
    identifier = directory.name
    for name in CAMERAS:
        camera_dir = Path(root) / name / identifier
        camera_dir.mkdir(parents=True, exist_ok=False)
        write_rgb(camera_dir / "rgb.png", frames[name])
        np.save(camera_dir / "corners.npy", corners[name], allow_pickle=False)
    manifest = {
        "schema_version": 1, "pair_id": identifier,
        "inner_columns": columns, "inner_rows": rows,
        "images": {name: f"{name}/{identifier}/rgb.png" for name in CAMERAS},
        "image_sizes": {name: [frames[name].shape[1], frames[name].shape[0]] for name in CAMERAS},
        "raw_rgb_sha256": fingerprints, "timing": timing,
        "host_receive_skew_ms": abs(timing["wrist"]["host_received_ns"] - timing["overhead"]["host_received_ns"])/1e6,
        "timing_note": "Host receipt only; exposure skew and USB buffering unknown. Hold the target still.",
    }
    (directory / "pair.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in CAMERAS:
        seen[name].add(fingerprints[name])
    return identifier


def draw_feedback(frames, corners, columns, rows, saved, message):
    panels = []
    for name in CAMERAS:
        bgr = cv2.cvtColor(frames[name], cv2.COLOR_RGB2BGR)
        if corners[name] is not None:
            cv2.drawChessboardCorners(bgr, (columns, rows), corners[name], True)
        scale = min(640/bgr.shape[1], 480/bgr.shape[0])
        resized = cv2.resize(bgr, (round(bgr.shape[1]*scale), round(bgr.shape[0]*scale)))
        panel = np.full((535,640,3), 28, dtype=np.uint8)
        panel[45:45+resized.shape[0], :resized.shape[1]] = resized
        status = "DETECTED" if corners[name] is not None else "NOT FOUND"
        color = (80,255,80) if corners[name] is not None else (80,80,255)
        cv2.putText(panel, f"{name.upper()}: {status}", (10,28), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
        panels.append(panel)
    canvas = np.full((625,1280,3), 28, dtype=np.uint8)
    canvas[:535] = cv2.hconcat(panels)
    cv2.putText(canvas, f"Saved {saved} pairs | Space/S: save when BOTH detected | Q: finish",
                (10,565), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
    cv2.putText(canvas, message, (10,600), cv2.FONT_HERSHEY_SIMPLEX, .55, (80,230,255), 1)
    return canvas


def capture_checkerboards(args):
    root = Path(args.output)
    board_points(args.columns, args.rows, 1)
    if args.square_mm is not None and (not np.isfinite(args.square_mm) or args.square_mm <= 0):
        raise ValueError("Measured square size must be positive and finite")
    if root.exists():
        raise ValueError("Session directory exists; choose a new output directory")
    if args.wrist_camera == args.overhead_camera:
        raise ValueError("Select two different camera devices")
    if (sys.platform.startswith("linux") and os.environ.get("DISPLAY")
            and (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland")):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    seen = {name: set() for name in CAMERAS}
    count = 0
    message = "Move and tilt the iPad between saves; hold it still for each capture. Keep screen zoom fixed."
    window = "Occlusion | two-camera checkerboard capture"
    try:
        with ExitStack() as stack:
            cameras = {name: stack.enter_context(USBCamera(getattr(args, f"{name}_camera"))) for name in CAMERAS}
            for _ in range(15):
                receive_pair(cameras)
            root.mkdir(parents=True, exist_ok=False)
            (root/"session.json").write_text(json.dumps({
                "schema_version": 1, "inner_columns": args.columns, "inner_rows": args.rows,
                "measured_square_size_mm": args.square_mm,
                "camera_sources": {name: getattr(args,f"{name}_camera") for name in CAMERAS},
                "instructions": "Stationary cameras, fixed focus/mode and screen zoom, varied stationary board poses",
                "depth_inference": False,
            },indent=2)+"\n")
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            while True:
                frames,timing = receive_pair(cameras)
                corners = {name: detect_board(frames[name],args.columns,args.rows) for name in CAMERAS}
                cv2.imshow(window,draw_feedback(frames,corners,args.columns,args.rows,count,message))
                key = cv2.waitKey(1)&0xFF
                if key in (ord("q"),ord("Q"),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                    break
                if key in (ord("s"),ord("S"),ord(" ")):
                    try:
                        identifier=save_board_pair(root,frames,corners,timing,args.columns,args.rows,seen)
                        count+=1
                        message=f"Saved pair {count}. Change board position/distance/tilt before the next save."
                        print(f"Saved {identifier} ({count} pairs)",flush=True)
                    except ValueError as error:
                        message=str(error)
    finally:
        cv2.destroyAllWindows()
    print(f"Saved {count} board pairs in {root}. Aim for 20-30 varied poses; detection is not calibration validation.")
