# Milestone 2: camera calibration and single-view geometry

The new tools run without a robot or GPU. Physical calibration is not complete
until you capture and validate a real board dataset for each camera. No robot
movement is needed for this step.

For an iPad instead of paper, use the [screen-target and paired capture guide](ipad.md).
Measure the displayed square size; the printed target's nominal millimeters do
not apply to a screen.

## 1. Print and verify the target

Open [checkerboard-9x6-20mm.pdf](checkerboard-9x6-20mm.pdf). It is a vector A4
landscape page containing **10 x 7 squares**, giving **9 x 6 inner corners**.
Each square is 20 mm wide; the pattern is 200 x 140 mm. A white margin surrounds
the pattern for detection.

Print at **Actual size / 100%**, in landscape, without fit-to-page scaling.
Measure the included ruler: it must be 100 mm. Check individual squares too.
Attach the sheet flat to a rigid backing; avoid glare, curling, folds, and
covering the white border. The board must remain flat when tilted.

The PDF can be regenerated without extra dependencies:

```bash
python -m occlusion.calibration board \
  --output docs/calibration/checkerboard-9x6-20mm.pdf
```

`--columns` and `--rows` always mean **inner corners**, not square counts.
For another board, supply its actual dimensions and measured square size to
both generation and calibration commands.

## 2. Collect each camera's dataset separately

Keep the camera stationary and vary the **board** position and orientation.
Keep focus, resolution, and camera image mode fixed. Recalibrate after changing
focus or camera mode. Remove the tabletop test objects temporarily if they
obstruct the board. Do not move the arm under software control.

Overhead Logitech:

```bash
QT_QPA_PLATFORM=xcb occlusion --camera /dev/video0 --preview --capture-only \
  --output data/calibration/overhead
```

Wrist Innomaker:

```bash
QT_QPA_PLATFORM=xcb occlusion --camera /dev/video2 --preview --capture-only \
  --output data/calibration/wrist
```

Press Space or `s` to save, then change the board pose before the next capture.
Collect about **20-30 sharp, distinct images per camera**, including the center,
corners, and edges of the field of view. Use different distances and tilts around
both board axes. Keep all squares and the surrounding white margin visible.
Pause board motion before saving. A burst of nearly identical images does not
provide equivalent calibration constraints.

The initial CLI saves the image you see, without a board-detection overlay.
Detection and rejection reasons are reported during offline fitting. These
capture folders are ignored by Git by default; share a reviewed dataset
explicitly if your teammate needs it.

## 3. Fit intrinsics and inspect validation

```bash
python -m occlusion.calibration fit \
  --images data/calibration/overhead --camera-id overhead \
  --columns 9 --rows 6 --square-mm 20 --output configs/overhead.json

python -m occlusion.calibration fit \
  --images data/calibration/wrist --camera-id wrist \
  --columns 9 --rows 6 --square-mm 20 --output configs/wrist.json
```

The tool uses OpenCV's sector-based checkerboard detector and the standard
pinhole model with five radial/tangential distortion coefficients. Identical
images are excluded. Mixed image resolutions are rejected. Every fifth detected
view is held out of the intrinsic fit; its pose is estimated using the fitted
intrinsics, then its reprojection error is measured. At least 10 training views
plus held-out images are required (13 detected views at this split).

The JSON records image hashes, accepted and rejected image paths, intrinsics,
distortion, board dimensions, per-view pixel RMS, held-out RMS, and a board-tilt
diversity diagnostic. Pose translations are based on square dimensions in meters.
They are board poses during calibration, not wrist poses or robot-base poses.

Initial acceptance gates require every view's reprojection RMS to be at most
1 pixel and training board normals to span at least 10 degrees from their mean
direction. These are preliminary diagnostics, not proof of a well-conditioned
calibration. Inspect the results and collect more poses if necessary. Changing
`--max-rms-px` must be justified by measured image quality rather than used to
hide bad views. A failed fit saves its report and returns a nonzero exit status.

Low training error alone is insufficient. Check held-out errors, image coverage,
physically plausible focal lengths, and repeatability with a separate dataset.
The tools do not currently estimate parameter covariance, detect all forms of
degeneracy, or fit a fisheye model. Calibration does not make relative depth
metric.

See [OpenCV's calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
and [camera geometry reference](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

## 4. Export a camera-frame point cloud

Capture a depth observation at the **same camera mode and focus** used for
calibration, with resolution 504 for more object detail:

```bash
occlusion --camera /dev/video0 --resolution 504 --output data/geometry/overhead
```

Use its saved observation directory:

```bash
python -m occlusion.pointcloud \
  --observation data/geometry/overhead/<capture> \
  --calibration configs/overhead.json --camera-id overhead \
  --output data/pointclouds/overhead-relative.ply
```

The camera ID is your explicit assertion of which camera made the observation;
older capture metadata does not store a calibrated camera identity. The exporter
checks calibration status and image dimensions, but cannot verify physical lens
identity or whether focus changed. Never apply one camera's calibration to the
other camera.

For current observations, DA3 performs resize without crop. The exporter scales
intrinsics to the actual processed depth dimensions using DA3's scale-only pixel
convention. It then corrects lens distortion in the pixel rays with
`cv2.undistortPoints`, rather than interpolating depth. This assumes optical-axis
depth and retains depth values at their original processed pixel locations.
Do not use it unchanged with cropped, rectified, or differently processed
observations. Distortion correction improves rays; it does not correct learned
depth errors caused by distorted input images.

The colored PLY uses OpenCV camera axes: x right, y down, z forward. Its sidecar
JSON records provenance, calibration hash, point count, and units. Invalid depth
is omitted. `--stride 2` is the default; `--stride 1` exports every valid pixel.
Existing output files are not overwritten.

**All current exports remain in arbitrary relative units, not meters.** They
are for single-view inspection and cannot drive robot motion or establish
multi-view alignment.

Before calibration is available, an explicitly uncalibrated diagnostic cloud
can be exported from the shared object-scene observation:

```bash
python -m occlusion.pointcloud \
  --observation data/overhead-depth/20261024T033417.716243Z-83ee4947 \
  --predicted-intrinsics --output data/pointclouds/diagnostic-relative.ply
```

This fallback uses DA3-predicted intrinsics and **does not correct distortion**.
It is labeled accordingly. It is not a substitute for calibration.

## 5. What remains after this step

Use measured geometry to validate a per-observation metric scale anchor. Assess
residuals away from the board, not only on the board itself. Then validate
camera-to-wrist pose and common-frame transforms before multi-view fusion.
Scale fitting, board-based metric depth, hand–eye calibration, and a point-cloud
viewer are not implemented by these tools yet.
