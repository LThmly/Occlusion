"""Dual-camera RGB/depth inspection and diagnostic point-cloud overlay in Rerun."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from .camera import USBCamera, read_rgb
from .depth import DA3Estimator, DepthResult

CAMERAS = ("wrist", "overhead")


def validate_geometry(intrinsics, world_from_camera):
    """Validate a pinhole matrix and rigid OpenCV-camera-to-world transform."""
    k = np.asarray(intrinsics, dtype=float)
    pose = np.asarray(world_from_camera, dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError("intrinsics must be a finite, invertible 3x3 pinhole matrix")
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
        raise ValueError("world_from_camera must be a rigid 4x4 transform")
    return k, pose


def project_cloud(rgb, depth, intrinsics, world_from_camera, *, scale=1.0,
                  stride=4, max_depth=float("inf")):
    """Back-project valid axial depths into world coordinates with aligned RGB."""
    k, pose = validate_geometry(intrinsics, world_from_camera)
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError("RGB and depth must be aligned on the same pixel grid")
    if stride < 1 or not np.isfinite(scale) or scale <= 0 or max_depth <= 0 or np.isnan(max_depth):
        raise ValueError("stride, scale, and max_depth must be positive")
    v, u = np.mgrid[:depth.shape[0]:stride, :depth.shape[1]:stride]
    z = depth[::stride, ::stride].astype(float) * scale
    valid = np.isfinite(z) & (z > 0) & (z <= max_depth)
    pixels = np.stack((u[valid], v[valid], np.ones(valid.sum())), axis=1)
    points = (pixels @ np.linalg.inv(k).T) * z[valid, None]
    return points @ pose[:3, :3].T + pose[:3, 3], rgb[::stride, ::stride][valid]


def load_capture(directory):
    """Load an existing lossless Occlusion depth capture without loading DA3."""
    rgb = read_rgb(directory / "rgb_processed.png")
    depth = np.load(directory / "depth.npy", allow_pickle=False)
    with np.load(directory / "prediction.npz", allow_pickle=False) as arrays:
        k = arrays["intrinsics"] if "intrinsics" in arrays else None
    return DepthResult(rgb, depth, None, k, None, 0)


def log_frame(rr, name, result, settings, args):
    """Log aligned images and separate RGB/identity-colored overlay entities."""
    rr.log(f"images/{name}/rgb", rr.Image(result.rgb))
    rr.log(f"images/{name}/depth", rr.DepthImage(result.depth))
    k = settings.get("intrinsics", result.intrinsics)
    if k is None:
        raise ValueError(f"{name}: no intrinsics; provide --geometry with depth-grid intrinsics")
    pose = settings.get("world_from_camera", np.eye(4))
    scale = settings.get("depth_scale", 1.0)
    points, colors = project_cloud(result.rgb, result.depth, k, pose, scale=scale,
                                   stride=args.stride, max_depth=args.max_depth)
    rr.log(f"world/rgb/{name}", rr.Points3D(points, colors=colors))
    tint = [255, 130, 40] if name == "wrist" else [40, 170, 255]
    rr.log(f"world/camera_colors/{name}", rr.Points3D(points, colors=tint))


def run(args):
    """Run paired camera observations or replay two saved captures."""
    import rerun as rr
    import rerun.blueprint as rrb

    geometry = json.loads(args.geometry.read_text()) if args.geometry else {}
    if not isinstance(geometry, dict):
        raise ValueError("geometry must be a JSON object keyed by wrist and overhead")
    for name in CAMERAS:
        settings = geometry.get(name, {})
        if not isinstance(settings, dict):
            raise ValueError(f"{name} geometry must be an object")
    rr.init("occlusion-dual-camera", spawn=not args.save)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(args.save))
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Vertical(*[rrb.Horizontal(
            rrb.Spatial2DView(name=f"{name} RGB", origin=f"images/{name}/rgb"),
            rrb.Spatial2DView(name=f"{name} depth", origin=f"images/{name}/depth"),
        ) for name in CAMERAS]),
        rrb.Tabs(rrb.Spatial3DView(name="RGB overlay", origin="world", contents=["world/rgb/**"]),
                 rrb.Spatial3DView(name="Camera identity overlay", origin="world",
                                   contents=["world/camera_colors/**"])),
    )))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    note = ("Diagnostic overlay: independent DA3 depth has arbitrary scale. Identity poses "
            "superimpose camera origins. Supply shared-frame poses and matched depth scales "
            "with --geometry for meaningful alignment. Wrist pose must match each observation. "
            "Live RGB pairs are read before sequential inference; they are not hardware synchronized.")
    print(note, flush=True)
    rr.log("notes", rr.TextDocument(note), static=True)
    try:
        with ExitStack() as stack:
            replay = args.wrist_capture is not None
            cameras = {} if replay else {
                name: stack.enter_context(USBCamera(getattr(args, f"{name}_camera")))
                for name in CAMERAS
            }
            estimator = None if replay else DA3Estimator(args.device, args.resolution)
            index = 0
            while replay or args.frames == 0 or index < args.frames:
                frames = {} if replay else {name: camera.read() for name, camera in cameras.items()}
                rr.set_time_sequence("observation", index)
                for name in CAMERAS:
                    result = (load_capture(getattr(args, f"{name}_capture")) if replay
                              else estimator.predict(frames[name]))
                    log_frame(rr, name, result, geometry.get(name, {}), args)
                index += 1
                if replay:
                    break
    finally:
        rr.disconnect()


def main():
    """Parse and validate the dual-camera visualizer command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrist-camera", default="/dev/video2")
    parser.add_argument("--overhead-camera", default="/dev/video0")
    parser.add_argument("--wrist-capture", type=Path)
    parser.add_argument("--overhead-capture", type=Path)
    parser.add_argument("--geometry", type=Path, help="JSON: depth-grid intrinsics, world_from_camera, depth_scale per camera")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resolution", type=int, default=252)
    parser.add_argument("--frames", type=int, default=0, help="Paired observations; 0 runs until Ctrl-C")
    parser.add_argument("--stride", type=int, default=4, help="Point-cloud pixel subsampling")
    parser.add_argument("--max-depth", type=float, default=float("inf"), help="Clip after depth scaling")
    parser.add_argument("--save", type=Path, help="Write a headless .rrd recording instead of launching viewer")
    args = parser.parse_args()
    if bool(args.wrist_capture) != bool(args.overhead_capture):
        parser.error("provide both --wrist-capture and --overhead-capture")
    if (args.frames < 0 or args.stride < 1 or args.max_depth <= 0 or np.isnan(args.max_depth)
            or args.resolution < 28 or args.resolution % 14):
        parser.error("frames must be nonnegative, stride/max-depth positive, resolution a multiple of 14 >= 28")
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError, ImportError, KeyError, TypeError) as error:
        parser.exit(1, f"Visualizer: {error}\nInstall viewer dependencies with pip install -e '.[viewer]'.\n")


if __name__ == "__main__":
    main()
