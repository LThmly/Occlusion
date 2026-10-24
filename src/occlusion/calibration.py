"""Checkerboard generation and offline intrinsic calibration; no robot access."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def board_points(columns, rows, square_size_m):
    if columns < 3 or rows < 3 or not np.isfinite(square_size_m) or square_size_m <= 0:
        raise ValueError("Board needs >=3 inner corners per axis and positive square size")
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_size_m
    return points


def write_board_pdf(path, columns=9, rows=6, square_mm=20):
    """Generate a vector A4 landscape PDF with white border and scale ruler."""
    board_points(columns, rows, square_mm / 1000)
    page_w, page_h = 297, 210
    width, height = (columns + 1) * square_mm, (rows + 1) * square_mm
    if width > page_w - 40 or height > page_h - 50:
        raise ValueError("Board does not fit A4 with the required border; reduce square size")
    factor = 72 / 25.4
    left, bottom = (page_w - width) / 2, (page_h - height) / 2
    content = ["0 g"]
    for row in range(rows + 1):
        for col in range(columns + 1):
            if (row + col) % 2 == 0:
                x, y = (left + col * square_mm) * factor, (bottom + row * square_mm) * factor
                content.append(f"{x:.5f} {y:.5f} {square_mm*factor:.5f} {square_mm*factor:.5f} re f")
    label = (f"Occlusion | {columns} x {rows} INNER corners | {square_mm:g} mm squares | "
             "Print ACTUAL SIZE / 100%, no fit to page")
    content.append(f"BT /F1 10 Tf 40 560 Td ({label}) Tj ET")
    x, y, length = 40 * factor, 15 * factor, 100 * factor
    content.append(f"0 G 0.7 w {x:.5f} {y:.5f} m {x+length:.5f} {y:.5f} l S")
    for mark in range(11):
        mx = x + mark * 10 * factor
        content.append(f"{mx:.5f} {y-4:.5f} m {mx:.5f} {y+4:.5f} l S")
    content.append(f"BT /F1 10 Tf {x:.5f} {y-17:.5f} Td (Verify this ruler measures 100 mm before calibration) Tj ET")
    stream = "\n".join(content).encode("ascii")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w*factor:.5f} {page_h*factor:.5f}] "
                "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>").encode(),
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"]
    pdf, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf)


def write_screen_board(path, columns=9, rows=6, square_pixels=200):
    """Lossless, white-bordered checkerboard for a screen; physical size unknown.

    Default canvas is 2400x1800 (4:3). No nominal millimeter size or display DPI
    is asserted. Measure the physical squares after displaying on the iPad.
    """
    board_points(columns, rows, 1)
    if not isinstance(square_pixels, int) or square_pixels < 8:
        raise ValueError("Screen target square size must be an integer >=8 pixels")
    margin = square_pixels
    image = np.full(((rows+1)*square_pixels+2*margin,
                     (columns+1)*square_pixels+2*margin), 255, dtype=np.uint8)
    for row in range(rows+1):
        for col in range(columns+1):
            if (row+col) % 2 == 0:
                image[margin+row*square_pixels:margin+(row+1)*square_pixels,
                      margin+col*square_pixels:margin+(col+1)*square_pixels] = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".png" or not cv2.imwrite(str(path), image):
        raise ValueError("Screen target output must be a writable .png file")


def fit_intrinsics(corners, image_size, columns, rows, square_size_m,
                   min_training_views=10, max_rms_px=1.0):
    """Reserve every fifth detected view for validation, solving its pose only.

    Held-out RMS checks transfer of intrinsics; it is not independent metric
    depth accuracy. Varied target poses and field coverage remain necessary.
    """
    obj = board_points(columns, rows, square_size_m)
    if len(corners) < 3:
        raise ValueError("Not enough detected views")
    heldout = list(range(0, len(corners), 5))
    training = [i for i in range(len(corners)) if i not in heldout]
    if len(training) < min_training_views:
        raise ValueError(f"Need >= {min_training_views} training views plus held-out views; "
                         f"only {len(training)} training / {len(heldout)} held-out detected")
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        [obj] * len(training), [corners[i] for i in training], image_size, None, None)
    residuals = []
    for index, observed in enumerate(corners):
        if index in training:
            local = training.index(index)
            rvec, tvec = rvecs[local], tvecs[local]
        else:
            ok, rvec, tvec = cv2.solvePnP(obj, observed, matrix, distortion)
            if not ok:
                raise ValueError(f"Cannot estimate held-out board pose for view {index}")
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        error = np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - observed.reshape(-1, 2))**2, axis=1)))
        residuals.append(float(error))
    normals = np.array([cv2.Rodrigues(rvec)[0][:, 2] for rvec in rvecs])
    mean_normal = normals.mean(axis=0)
    mean_normal /= np.linalg.norm(mean_normal)
    spread = float(np.max(np.degrees(np.arccos(np.clip(normals @ mean_normal, -1, 1)))))
    warnings = []
    if spread < 10:
        warnings.append("Little board tilt diversity; collect more varied orientations")
    finite = np.isfinite(matrix).all() and np.isfinite(distortion).all() and np.isfinite(residuals).all()
    passed = bool(finite and matrix[0, 0] > 0 and matrix[1, 1] > 0
                  and max(residuals) <= max_rms_px and not warnings)
    return {
        "schema_version": 1, "image_size": list(image_size),
        "intrinsics": matrix.tolist(), "distortion": distortion.ravel().tolist(),
        "distortion_model": "opencv_pinhole_k1_k2_p1_p2_k3",
        "board": {"inner_columns": columns, "inner_rows": rows, "square_size_m": square_size_m},
        "training_indices": training, "heldout_indices": heldout,
        "training_rms_px": float(rms), "per_view_rms_px": residuals,
        "heldout_rms_px": [residuals[i] for i in heldout],
        "max_allowed_view_rms_px": max_rms_px, "board_normal_spread_deg": spread,
        "warnings": warnings, "validation_passed": passed,
        "validation_note": "Reprojection and tilt-diversity gates only; not metric depth validation",
    }


def calibrate_directory(directory, columns, rows, square_mm, camera_id, max_rms_px=1.0):
    directory = Path(directory)
    paths = sorted(directory.rglob("rgb.png"))
    if not paths:
        paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        raise ValueError(f"No RGB calibration images in {directory}")
    corners, detected, rejected, seen = [], [], [], set()
    image_size = None
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Cannot read {path}")
        size = (gray.shape[1], gray.shape[0])
        if image_size is not None and size != image_size:
            raise ValueError("Mixed image sizes: calibrate each camera mode separately")
        image_size = size
        digest = hashlib.sha256(gray.tobytes()).hexdigest()
        if digest in seen:
            rejected.append({"path": str(path), "reason": "identical image"})
            continue
        seen.add(digest)
        found, observed = cv2.findChessboardCornersSB(gray, (columns, rows), cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            corners.append(observed.astype(np.float32))
            detected.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        else:
            rejected.append({"path": str(path), "reason": "board not detected"})
    result = fit_intrinsics(corners, image_size, columns, rows, square_mm / 1000,
                            max_rms_px=max_rms_px)
    result.update(camera_id=camera_id, views=detected, rejected_images=rejected)
    return result


def load_calibration(path):
    from .geometry import validate_intrinsics

    result = json.loads(Path(path).read_text())
    validate_intrinsics(result["intrinsics"])
    if result.get("schema_version") != 1 or not result.get("validation_passed"):
        raise ValueError("Calibration has not passed its validation gates")
    if result.get("distortion_model") != "opencv_pinhole_k1_k2_p1_p2_k3":
        raise ValueError("Unsupported distortion model")
    distortion = np.asarray(result["distortion"], dtype=float)
    if distortion.shape != (5,) or not np.isfinite(distortion).all():
        raise ValueError("Expected five finite distortion coefficients")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    board = commands.add_parser("board", help="Write printable A4 landscape checkerboard PDF")
    board.add_argument("--output", type=Path, required=True)
    screen = commands.add_parser("screen-board", help="Write white-bordered PNG for an iPad or other flat screen")
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--columns", type=int, default=9)
    screen.add_argument("--rows", type=int, default=6)
    screen.add_argument("--square-pixels", type=int, default=200)
    capture = commands.add_parser("capture-pair", help="Live checkerboard detection and paired RGB capture, no depth inference")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--wrist-camera", default="/dev/video2")
    capture.add_argument("--overhead-camera", default="/dev/video0")
    capture.add_argument("--columns", type=int, default=9)
    capture.add_argument("--rows", type=int, default=6)
    capture.add_argument("--square-mm", type=float, help="Measured displayed square size; may be recorded later")
    fit = commands.add_parser("fit", help="Fit intrinsics from saved RGB captures")
    fit.add_argument("--images", type=Path, required=True)
    fit.add_argument("--camera-id", required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--max-rms-px", type=float, default=1.0)
    for command in (board, fit):
        command.add_argument("--columns", type=int, default=9, help="Inner corners, not squares")
        command.add_argument("--rows", type=int, default=6, help="Inner corners, not squares")
        command.add_argument("--square-mm", type=float, default=20)
    args = parser.parse_args()
    try:
        if args.command == "screen-board":
            write_screen_board(args.output, args.columns, args.rows, args.square_pixels)
            print(f"Saved {args.output}; display the complete pattern with its white border, "
                  "then measure physical square size. Screen pixels are not millimeters.")
            return
        if args.command == "capture-pair":
            from .checkerboard_capture import capture_checkerboards
            capture_checkerboards(args)
            return
        board_points(args.columns, args.rows, args.square_mm / 1000)
        if args.command == "board":
            write_board_pdf(args.output, args.columns, args.rows, args.square_mm)
            print(f"Saved {args.output}; print at 100% and verify the 100 mm ruler")
        else:
            if not np.isfinite(args.max_rms_px) or args.max_rms_px <= 0:
                raise ValueError("RMS threshold must be finite and positive")
            result = calibrate_directory(args.images, args.columns, args.rows, args.square_mm,
                                         args.camera_id, args.max_rms_px)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(f"Saved {args.output}; training RMS {result['training_rms_px']:.3f}px; "
                  f"max held-out RMS {max(result['heldout_rms_px']):.3f}px")
            if not result["validation_passed"]:
                parser.exit(1, "Calibration failed validation; inspect per-view errors/warnings and retake images\n")
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Calibration: {error}\n")


if __name__ == "__main__":
    main()
