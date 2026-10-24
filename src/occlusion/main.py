"""Snapshot or sequential live depth inference; no robot connection or motion."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import resource
import sys


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Infer from a saved RGB image")
    source.add_argument("--camera", default="0", help="USB index or /dev/video path (default: 0)")
    result.add_argument("--capture-only", action="store_true", help="Save RGB without loading DA3")
    result.add_argument("--preview", action="store_true",
                        help="Continuously preview; Space/s saves a frame, q quits. Add --capture-only for RGB only")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--resolution", type=int, default=252, help="DA3 processing size (default: 252)")
    result.add_argument("--frames", type=int, default=1, help="Camera observations; 0 repeats until Ctrl-C/q")
    result.add_argument("--output", type=Path, default=Path("data/captures"))
    result.add_argument("--no-display", action="store_true")
    return result


def run(args):
    # The Linux opencv-python wheel bundles Qt's X11 plugin, not Wayland.
    # Use XWayland when available, while respecting an explicit Qt selection.
    if (not args.no_display and sys.platform.startswith("linux")
            and (os.environ.get("WAYLAND_DISPLAY")
                 or os.environ.get("XDG_SESSION_TYPE") == "wayland")
            and os.environ.get("DISPLAY")):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    import cv2
    from .camera import USBCamera, read_rgb
    from .depth import DA3Estimator, DEFAULT_MODEL
    from .outputs import colorize_depth, new_capture_dir, save_result, write_rgb

    estimator = None if args.capture_only else DA3Estimator(args.device, args.resolution)
    with ExitStack() as stack:
        camera = None if args.image else stack.enter_context(USBCamera(args.camera))
        index = 0
        while args.preview or args.frames == 0 or index < args.frames:
            rgb = read_rgb(args.image) if args.image else camera.read()
            if estimator is None:
                preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                prediction = estimator.predict(rgb)
                metadata = {
                    "model": DEFAULT_MODEL, "device": args.device,
                    "process_resolution": args.resolution,
                    "process_res_method": "upper_bound_resize",
                    "source": str(args.image) if args.image else f"camera:{args.camera}",
                    "peak_cuda_allocated_bytes": estimator.peak_cuda_bytes(),
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                }
                preview = cv2.hconcat((cv2.cvtColor(prediction.rgb, cv2.COLOR_RGB2BGR),
                                      colorize_depth(prediction.depth)))
                if not args.preview:
                    print(f"Inference: {prediction.seconds:.2f}s", flush=True)

            def save_current():
                directory = new_capture_dir(args.output)
                if estimator is None:
                    write_rgb(directory / "rgb.png", rgb)
                else:
                    save_result(directory, rgb, prediction, metadata)
                print(f"Saved {directory}", flush=True)

            if not args.preview:
                save_current()
            index += 1
            if not args.no_display:
                cv2.imshow("Occlusion | Space/s: save | q: quit", preview)
                delay = 0 if not args.preview and (args.image or (args.frames > 0 and index == args.frames)) else 1
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord("q"), 27):
                    break
                if args.preview and key in (ord("s"), ord(" ")):
                    save_current()
            if args.image:
                break


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()
    if args.preview and (args.image or args.no_display or args.frames != 1):
        arg_parser.error("--preview uses a camera and display; omit --image, --no-display, and --frames")
    if args.frames < 0 or args.resolution < 28 or args.resolution % 14:
        arg_parser.error("--frames must be nonnegative; --resolution must be a multiple of 14 >= 28")
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        arg_parser.exit(1, f"Occlusion: {error}\nSee README.md for CPU/DA3 setup.\n")
    finally:
        if "cv2" in sys.modules:
            sys.modules["cv2"].destroyAllWindows()


if __name__ == "__main__":
    main()
