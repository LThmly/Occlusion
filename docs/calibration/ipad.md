# Calibrate with a checkerboard on an iPad

This replaces the printed board with a flat screen showing a fixed pattern.
First collect camera RGB calibration data. Depth inference and robot movement
are not used during collection. Lens and relative-camera calibration provide
independent camera geometry; they do not automatically repair predicted depth.

## Teammate setup

The iPad displays the target. Run capture on the Linux computer connected to
both USB cameras; the Python tools do not run on the iPad.

For a new checkout with [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/LThmly/Occlusion.git
cd Occlusion
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

For an existing checkout, pull the changes, activate its virtual environment,
and run `uv pip install -e .`. Checkerboard generation, detection, capture, and
intrinsic fitting need only the base package. Torch, DA3, Rerun, and a GPU are
not required for this workflow.

## Display and measure

Open [checkerboard-ipad.png](checkerboard-ipad.png) on the iPad, preferably as a
full-screen image in landscape. It has **10 columns x 7 rows of squares**, or
**9 x 6 inner corners**, with a white border. Keep the entire pattern and white
border visible. Do not stretch the image or crop the pattern.

The file is 2400 x 1800 pixels. It has **no specified physical square size**.
Do not use the printable PDF's nominal 20 mm for this screen image. Measure the
physical width of five adjacent squares with a ruler, then divide by five.
For example, 75 mm across five squares means `--square-mm 15`.

After measuring, keep display zoom, orientation, and image size fixed throughout
the session. Keep the screen awake and avoid overlaying notifications or image
controls on the pattern. Use brightness that produces clear black/white regions
without glare or saturated glare spots. Adjust the iPad angle if reflections
obscure corners. Keep fingers off the pattern and its white border.

If direct transfer to the iPad is inconvenient, serve this directory from the
laptop on your local network:

```bash
python -m http.server 8000 --directory docs/calibration
```

Open `http://YOUR_LAPTOP_LAN_IP:8000/checkerboard-ipad.png` on the iPad. Use the
laptop's LAN address, not `localhost`. The iPad and laptop must be on a network
that allows the connection. Stop the server with Ctrl-C when finished. This
serves only the calibration documentation directory.

## Collect a paired session

Stop any existing camera preview or Rerun acquisition first. Keep the arm and
both camera mounts stationary. Fix each camera's focus and image mode; changing
focus or mode can change intrinsics. Move only the iPad.

```bash
QT_QPA_PLATFORM=xcb python -m occlusion.calibration capture-pair \
  --output data/checkerboard/session1
```

Defaults: wrist `/dev/video2`, overhead `/dev/video0`, pattern 9 x 6 inner
corners. Override device paths if necessary. If measured square size is already
known, add `--square-mm ACTUAL_SIZE_MM` to record it in the session manifest.
Otherwise record the measurement separately for fitting. The output directory
must be new.

The window draws detected corner grids and shows **DETECTED** or **NOT FOUND**
for each camera. Press Space or `S`/`s` to save when both are detected; `Q`/Escape
finishes. The raw images are saved without the colored detection overlay.

Aim for **20-30 distinct paired poses**. Move the iPad across the shared field
of view, vary distance within the cameras' focus range, and tilt it about both
axes. Keep the full board visible and sharp in both cameras. Hold it stationary
for each save. Keep the same edge uppermost rather than turning the pattern
upside down; plain checkerboard ordering must be checked before cross-camera
pose fitting. Detection alone does not establish correct ordering or adequate
pose diversity.

If the pattern is not detected, first increase its visible size, reduce tilt,
remove glare, and check focus. Do not collect a long sequence of nearly identical
frontal images just because detection succeeds. If overlap is too small for a
useful paired dataset, record that limitation instead of moving the arm under
software control.

The tool grabs both camera frames before decoding, but host times are not sensor
exposure times. Keeping the iPad stationary during a saved pair makes precise
time synchronization unnecessary for this static calibration experiment.

## Saved data

```text
data/checkerboard/session1/
├── session.json
├── wrist/<pair-id>/rgb.png
├── wrist/<pair-id>/corners.npy
├── overhead/<pair-id>/rgb.png
├── overhead/<pair-id>/corners.npy
└── pairs/<pair-id>/pair.json
```

Pair manifests preserve common identifiers, image dimensions, image hashes,
camera receipt times, and board dimensions. Literal duplicate images are
rejected; noisy repeated frames may still be geometrically redundant. These
session directories are ignored by Git unless explicitly shared.

## Fit each lens, then assess cross-camera pose

Replace `ACTUAL_SIZE_MM` below with your measured square width, without units:

```bash
python -m occlusion.calibration fit \
  --images data/checkerboard/session1/wrist --camera-id wrist \
  --square-mm ACTUAL_SIZE_MM --output configs/wrist.json

python -m occlusion.calibration fit \
  --images data/checkerboard/session1/overhead --camera-id overhead \
  --square-mm ACTUAL_SIZE_MM --output configs/overhead.json
```

These commands fit intrinsics/distortion and report training and held-out pixel
errors. Review their diagnostics before cross-camera pose fitting. They do not
yet generate a two-camera Rerun geometry file. The paired dataset is also the
input for estimating relative camera pose once board ordering and fit quality
have been checked. Metric depth-scale validation remains a separate gate.

Source conventions:
[OpenCV camera calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
and [stereo calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
