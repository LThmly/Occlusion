"""Isolated camera acquisition and OpenCV point-picking for manual alignment."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from .alignment import CAMERAS, load_pair, pair_identity, sample_point
from .camera import USBCamera, receive_pair
from .depth import DA3Estimator, DEFAULT_MODEL
from .outputs import save_result, write_rgb


def capture_pair(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("Pair output already exists; choose a new directory")
    if args.wrist_camera == args.overhead_camera:
        raise ValueError("Wrist and overhead must be different camera devices")
    print("Keep the arm, cameras, and scene stationary. Warming up both cameras...", flush=True)
    with ExitStack() as stack:
        cameras = {name: stack.enter_context(USBCamera(getattr(args, f"{name}_camera"))) for name in CAMERAS}
        for _ in range(args.warmup_frames):
            receive_pair(cameras)
        frames, timing = receive_pair(cameras)
    output.mkdir(parents=True)
    for name in CAMERAS:
        (output / name).mkdir()
        write_rgb(output / name / "rgb.png", frames[name])
    manifest = {
        "schema_version": 1, "mode": "stationary_pair_not_hardware_synchronized",
        "camera_sources": {name: getattr(args, f"{name}_camera") for name in CAMERAS},
        "timing": timing,
        "host_receive_skew_ms": abs(timing["wrist"]["host_received_ns"] - timing["overhead"]["host_received_ns"]) / 1e6,
        "timing_note": "Host receipt only; buffered frames and exposure skew are unknown",
        "resolution": args.resolution,
    }
    (output / "pair.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved both RGB frames. Loading DA3 and processing the frozen pair...", flush=True)
    estimator = DA3Estimator(args.device, args.resolution)
    for name in CAMERAS:
        result = estimator.predict(frames[name])
        save_result(output / name, frames[name], result, {
            "camera_id": name, "source": getattr(args, f"{name}_camera"),
            "model": DEFAULT_MODEL, "device": args.device,
            "process_resolution": args.resolution, "process_res_method": "upper_bound_resize",
            "captured_at_utc": timing[name]["captured_at_utc"],
            "host_received_ns": timing[name]["host_received_ns"],
            "capture_mode": manifest["mode"],
        })
    print(f"Saved paired observations to {output}. Next: python -m occlusion.alignment pick --pair {output}", flush=True)


class MatchPicker:
    """Testable selection state; UI coordinates map to exact processed pixels."""

    def __init__(self, results):
        self.results = results
        self.pairs = []
        self.pending = None
        self.message = "Click a feature in WRIST, then the SAME feature in OVERHEAD."
        self.layouts = {}
        self.viewports = {}
        left = 0
        for name in CAMERAS:
            result = results[name]
            scale = min(640 / result.rgb.shape[1], 480 / result.rgb.shape[0])
            width, height = round(result.rgb.shape[1]*scale), round(result.rgb.shape[0]*scale)
            self.layouts[name] = (left, 65, width, height)
            self.viewports[name] = (0, 0, result.rgb.shape[1], result.rgb.shape[0])
            left += width + 20

    def zoom(self, x, y, factor):
        """Zoom one pane around the pointer; selected pixels stay on the original grid."""
        for name, (left, top, width, height) in self.layouts.items():
            if left <= x < left+width and top <= y < top+height:
                x0, y0, roi_w, roi_h = self.viewports[name]
                image_h, image_w = self.results[name].depth.shape
                fx, fy = (x-left)/width, (y-top)/height
                new_w = int(np.clip(round(roi_w/factor), max(4, np.ceil(image_w/8)), image_w))
                new_h = int(np.clip(round(roi_h/factor), max(4, np.ceil(image_h/8)), image_h))
                new_x = int(np.clip(round(x0+fx*(roi_w-new_w)), 0, image_w-new_w))
                new_y = int(np.clip(round(y0+fy*(roi_h-new_h)), 0, image_h-new_h))
                self.viewports[name] = (new_x, new_y, new_w, new_h)
                return

    def reset_zoom(self):
        for name, result in self.results.items():
            self.viewports[name] = (0, 0, result.depth.shape[1], result.depth.shape[0])

    def depth_edge(self, name, pixel):
        u, v = pixel
        depth = self.results[name].depth
        patch = depth[max(0,v-2):v+3, max(0,u-2):u+3]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        return len(valid) < patch.size or np.ptp(valid) > 0.05*np.median(valid)

    def click(self, x, y):
        for name, (left, top, width, height) in self.layouts.items():
            if left <= x < left + width and top <= y < top + height:
                result = self.results[name]
                x0, y0, roi_w, roi_h = self.viewports[name]
                pixel = [x0+int((x-left)*roi_w/width), y0+int((y-top)*roi_h/height)]
                try:
                    sample_point(result, pixel)
                except ValueError as error:
                    self.message = str(error)
                    return
                if name == "wrist":
                    if self.pending is not None:
                        self.message = "Finish this pair in OVERHEAD, or press U to undo."
                    elif any(item[name] == pixel for item in self.pairs):
                        self.message = "That pixel was already selected. Choose a different feature."
                    else:
                        self.pending = pixel
                        self.message = "Now click the SAME physical feature in OVERHEAD."
                        if self.depth_edge(name, pixel):
                            self.message = "Steep WRIST depth change here. U to undo; prefer an interior printed feature."
                elif self.pending is None:
                    self.message = "Start each pair in WRIST."
                elif any(item[name] == pixel for item in self.pairs):
                    self.message = "That overhead pixel was already selected. Choose the matching feature."
                else:
                    self.pairs.append({"wrist": self.pending, "overhead": pixel})
                    self.pending = None
                    self.message = f"{len(self.pairs)} complete pairs. Click the next feature in WRIST."
                    if any(self.depth_edge(camera, self.pairs[-1][camera]) for camera in CAMERAS):
                        self.message = f"Pair {len(self.pairs)} is near a depth edge. Consider U and an interior feature."
                return

    def undo(self):
        if self.pending is not None:
            self.pending = None
        elif self.pairs:
            self.pairs.pop()
        self.message = "Undone. Click a feature in WRIST."

    def draw(self):
        width = max(left+w for left, _, w, _ in self.layouts.values())
        bottom = max(top+h for _, top, _, h in self.layouts.values())
        canvas = np.full((bottom+85, width, 3), 28, dtype=np.uint8)
        cv2.putText(canvas, "U/right-click: undo | Enter: save (9+, aim for 12) | Q: cancel",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, "Hover a feature: Z zoom in / X zoom out | R reset zoom | scroll also zooms",
                    (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for name, (left, top, w, h) in self.layouts.items():
            result = self.results[name]
            x0, y0, roi_w, roi_h = self.viewports[name]
            crop = result.rgb[y0:y0+roi_h, x0:x0+roi_w]
            canvas[top:top+h, left:left+w] = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), (w, h))
            cv2.putText(canvas, name.upper(), (left+10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            points = [(i+1, item[name]) for i, item in enumerate(self.pairs)]
            if name == "wrist" and self.pending is not None:
                points.append((len(self.pairs)+1, self.pending))
            for index, pixel in points:
                if not (x0 <= pixel[0] < x0+roi_w and y0 <= pixel[1] < y0+roi_h):
                    continue
                pos = (left + int((pixel[0]-x0+0.5)*w/roi_w),
                       top + int((pixel[1]-y0+0.5)*h/roi_h))
                cv2.circle(canvas, pos, 5, (0, 255, 255), 1)
                cv2.putText(canvas, str(index), (pos[0]+7, pos[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(canvas, self.message, (10, bottom+30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,230,255), 1)
        cv2.putText(canvas, "Spread matches across the scene and heights. Avoid silhouettes, glare, and hidden features.",
                    (10, bottom+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        return canvas


def pick_matches(pair, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Matches file already exists; pass --output with a new filename to pick again")
    if (sys.platform.startswith("linux") and os.environ.get("DISPLAY")
            and (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland")):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    picker = MatchPicker(load_pair(pair))
    window = "Occlusion | stationary pair alignment"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    pointer = [0, 0]

    def mouse(event, x, y, flags, parameter):
        pointer[:] = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN:
            picker.click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            picker.undo()
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = flags >> 16
            if delta:
                picker.zoom(x, y, 2 if delta > 0 else 0.5)

    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            cv2.imshow(window, picker.draw())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                print("Selection cancelled; no matches written")
                return
            if key in (ord("u"), 8, 127):
                picker.undo()
            elif key in (ord("z"), ord("+"), ord("=")):
                picker.zoom(*pointer, 2)
            elif key in (ord("x"), ord("-")):
                picker.zoom(*pointer, 0.5)
            elif key == ord("r"):
                picker.reset_zoom()
            elif key in (10, 13):
                if len(picker.pairs) < 9 or picker.pending is not None:
                    picker.message = "Finish at least 9 complete pairs; 12 spread-out pairs are recommended."
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"schema_version": 1, "capture_identity": pair_identity(pair),
                                              "pixel_grid": "processed_rgb_and_depth",
                                              "pairs": picker.pairs}, indent=2) + "\n")
                print(f"Saved {len(picker.pairs)} matches to {output}")
                return
    finally:
        cv2.destroyAllWindows()
