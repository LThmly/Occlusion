"""Snapshot or sequential live depth inference; no robot connection or motion."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import resource
import sys


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Infer from a saved RGB image")
    source.add_argument("--camera", default="0", help="USB index or /dev/video path (default: 0)")
    result.add_argument("--capture-only", action="store_true", help="Save RGB without loading DA3")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--resolution", type=int, default=252, help="DA3 processing size (default: 252)")
    result.add_argument("--frames", type=int, default=1, help="Camera observations; 0 repeats until Ctrl-C/q")
    result.add_argument("--output", type=Path, default=Path("data/captures"))
    result.add_argument("--no-display", action="store_true")
    return result


def run(args):
    import cv2
    from .camera import USBCamera, read_rgb
    from .depth import DA3Estimator, DEFAULT_MODEL
    from .outputs import new_capture_dir, save_result, write_rgb

    estimator = None if args.capture_only else DA3Estimator(args.device, args.resolution)
    with ExitStack() as stack:
        camera = None if args.image else stack.enter_context(USBCamera(args.camera))
        index = 0
        while args.frames == 0 or index < args.frames:
            rgb = read_rgb(args.image) if args.image else camera.read()
            directory = new_capture_dir(args.output)
            if estimator is None:
                write_rgb(directory / "rgb.png", rgb)
                preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                prediction = estimator.predict(rgb)
                preview = save_result(directory, rgb, prediction, {
                    "model": DEFAULT_MODEL, "device": args.device,
                    "process_resolution": args.resolution,
                    "source": str(args.image) if args.image else f"camera:{args.camera}",
                    "peak_cuda_allocated_bytes": estimator.peak_cuda_bytes(),
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                })
                print(f"Inference: {prediction.seconds:.2f}s", flush=True)
            print(f"Saved {directory}", flush=True)
            index += 1
            if not args.no_display:
                cv2.imshow("Occlusion | RGB, relative depth | q: quit", preview)
                delay = 0 if args.image or (args.frames > 0 and index == args.frames) else 1
                if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                    break
            if args.image:
                break


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()
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
