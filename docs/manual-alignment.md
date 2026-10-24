# Align the two camera clouds without a printed target

This workflow fits a **provisional similarity transform**, not metric lens or
stereo calibration. It uses matching physical features in two frozen RGB/depth
observations. The overhead camera supplies the reference frame and relative
units; the wrist cloud receives a fitted scale, rotation, and translation.
Nothing here connects to the arm or commands motion.

## 1. Arrange a stationary shared scene

Stop live acquisition with Ctrl-C in its terminal so the cameras are available.
Keep the arm and both camera mounts stationary throughout capture and checking.

Place a box or another textured raised object where **both cameras see the same
physical features**. Include visible features at several heights and spread them
across the shared field of view. A few hand-drawn crosses on small pieces of
paper attached flat to the table and box can help. No known marker dimensions
are required for this relative alignment. Make all markers unique enough to
identify from both angles. Aim for twelve distinct shared features.

Use the center of a marking or a distinctive interior texture point rather than
a silhouette, shiny reflection, or thin edge. Depth predictions near object
boundaries can blend foreground and background. Do not match two different
corners merely because they look similar. A feature hidden in one camera cannot
be used. The thin remote alone is not an adequate scene for validating shape.

## 2. Save one camera pair

From the active project virtual environment:

```bash
python -m occlusion.alignment capture --output data/alignment/pair1
```

Defaults are wrist `/dev/video2`, overhead `/dev/video0`, CPU, resolution 504.
Override devices with `--wrist-camera` and `--overhead-camera` if needed. Stable
`/dev/v4l/by-id/` paths are accepted. The output directory must be new; use
`pair2`, etc. for another attempt.

The cameras warm up, both frames are grabbed before decoding, and both RGB
images are saved **before** depth inference. Cameras are then released and DA3
processes the frozen observations sequentially. This avoids capturing the second
image after waiting for the first inference.

Inspect `pair1/wrist/rgb.png` and `pair1/overhead/rgb.png` if the shared scene is
unclear. Each subdirectory contains the standard depth-observation files. The
pair manifest records camera sources and host grab/receipt times. These are not
sensor exposure timestamps: USB buffering and exposure timing remain unknown.
Keep the scene stationary; this command does not provide hardware synchronization.

## 3. Click matching features

```bash
QT_QPA_PLATFORM=xcb python -m occlusion.alignment pick --pair data/alignment/pair1
```

The picker opens the processed images side by side. They are the exact grids
used for depth and intrinsics, enlarged only for display.

1. Click one identifiable feature in the **left/wrist** image.
2. Click that **same physical feature** in the **right/overhead** image.
3. Repeat for twelve spread-out pairs. Select at least nine complete pairs.
4. Press Enter to save `data/alignment/pair1/matches.json`.

`U` or right-click undoes the pending selection or the most recent complete pair.
Hover over a feature and press `Z` to zoom that pane in; `X` zooms out and `R`
resets both views. Mouse-wheel zoom is also supported. Zoom changes only the
display, not the saved pixel coordinates. A warning appears when a selected
pixel has a steep depth change in its 5x5 neighborhood. Prefer printed interior
details over silhouettes; a finite depth value at an edge can still be wrong.
`Q`/Escape cancels without writing. Numbered markers show selection order. The
tool rejects invalid-depth pixels and exact duplicate pixel selections. It
cannot know whether two selected features are the same physical point.

Every third completed pair is reserved for validation rather than fitting. Spread
features and heights throughout the selection order. A matches file records
hashes of the exact saved RGB, depth, and predicted-intrinsics files. Replacing
those observations invalidates the selections. To repick without overwriting,
use `--output data/alignment/pair1/matches-v2.json` and pass that file to `fit`.

## 4. Fit and check the transform

```bash
python -m occlusion.alignment fit --pair data/alignment/pair1 \
  --output configs/geometry.json
```

The tool fits the training matches with seeded RANSAC and a positive-scale,
nonreflecting least-squares similarity fit:

$$
p_{\mathrm{overhead}}=sRp_{\mathrm{wrist}}+t.
$$

The held-out points do not participate in fitting or setting the normalization
span. Six or more training inliers and at least 75% training support are required.
Training inliers must lie within 5% of the maximum pairwise span of the overhead
training points. All held-out errors must be within 10% of that same span.
These are initial diagnostic tolerances, not centimeter accuracy guarantees.
Collinear configurations are rejected; nearly planar matches produce a warning.

On success, `configs/geometry.json` is directly usable by the Rerun viewer.
`configs/geometry.report.json` records each point's residual, training/held-out
indices, fitted parameters, thresholds, capture hashes, and limitations.
Inspect the report and the overlay. A numerical pass alone is not proof of
correct physical geometry.

If training support or held-out validation fails, a diagnostic report is saved
and **no geometry file is written**. Degenerate or invalid inputs can fail earlier.
The report also includes an image-space planar correspondence diagnostic where
possible. Good image agreement with poor 3D agreement points to depth or inferred
camera-geometry inconsistency; it is not a reason to keep reselecting the same
features. This image check does not participate in acceptance. First check matching
point identities, visibility, depth boundaries, and scene stability. Collect
more varied correspondences or a better scene if the model's shapes disagree.
Do not increase tolerances merely to make a bad fit pass. Existing configuration
and report files are not overwritten; use a new output name for a revised fit.

To inspect the fitted point discrepancies and test small click perturbations:

```bash
uv pip install -e '.[analysis]'
python -m occlusion.alignment_diagnostics \
  --pair data/alignment/pair1 \
  --matches data/alignment/pair1/matches-v2.json \
  --report configs/geometry-v2.report.json \
  --output data/alignment-diagnostics/pair1-v2 --trials 100
```

This saves point-error plots and a seeded sensitivity report. It independently
perturbs each camera's clicks by uniform integer offsets up to 1, 2, and 3
processed pixels, resamples depth, and refits using training matches only.
It reports checks normalized both to each trial's span and to the original
span, so a larger span caused by a depth discontinuity cannot quietly hide
absolute error. These simulations are sensitivity tests, not measured click
uncertainty, uncertainty calibration, or new evidence of accurate geometry.
No perturbed fit is exported or selected as an alignment configuration.

## 5. Replay this exact pair in Rerun

```bash
occlusion-viewer \
  --wrist-capture data/alignment/pair1/wrist \
  --overhead-capture data/alignment/pair1/overhead \
  --geometry configs/geometry.json
```

Switch to Camera identity overlay and orbit the view. Check the table, shared
object surfaces, and held-out features from several angles. Fitted scale can
align overall size; it cannot repair shape errors that differ between cameras.
This replay is finite and avoids recording thousands of live frames while
tuning alignment. The viewer rejects a different saved pair when using this
generated configuration, because its scale and intrinsics were fitted to these
specific observations. It also rejects a different processed image size.

## 6. Try live overlay only after replay looks credible

```bash
occlusion-viewer --geometry configs/geometry.json --resolution 504
```

Keep the wrist fixed. New single-view predictions may change scale or inferred
camera geometry even without motion, so live overlay can drift. Only the saved
pair was checked. The configuration supplies fixed intrinsics, not per-frame
calibration or scale tracking.

Lens calibration, metric scale validation, hardware timing synchronization, and
moving-wrist pose tracking remain separate work. A wrist motion invalidates this
fixed transform. None of these provisional coordinates may command robot motion.

The similarity-fit formulation follows
[Umeyama, 1991](https://web.stanford.edu/class/cs273/refs/umeyama.pdf).
