# Initial camera and depth quality assessment

Examined six Logitech RGB captures, one Innomaker RGB capture, and three
Innomaker depth observations from 2026-10-24 UTC. This is an observational
assessment without calibrated depth ground truth. No robot movement was used.

## Findings

- Both cameras produced 640 x 480 RGB images. The model operated on 252 x 196
  images, approximately 16% of the input pixel count. Small geometric details
  therefore have limited representation in the current inference input.
- The overhead Logitech image is usable for framing the workspace. White robot
  parts lose detail in bright regions; in the first capture, 7.66% of pixels
  have grayscale values >=250, versus 2.37% in the latest. Median grayscale
  across the six images ranges from 117 to 167. This indicates capture variation,
  but camera/scene changes prevent attributing it solely to auto-exposure.
- Wrist imagery shows table texture and the gripper. The gripper occupies the
  lower central field of view and casts a strong nearby shadow. These should be
  masked from target reconstruction. The available images do not establish
  focus quality on a target object at the intended working distance.
- The depth preview separates the nearby gripper from the tabletop qualitatively,
  with coarse transitions around thin gripper features. The scene contains no
  separate target/occluder pair, so occlusion recovery is not assessable.

## Numeric checks

All three depth maps contain 100% finite positive values. This is numeric
validity, not evidence that every pixel has correct depth.

| Capture UTC | Inference seconds | Process peak RSS, MiB |
| --- | ---: | ---: |
| 03:27:53 | 0.174 | 760.1 |
| 03:28:54 | 0.139 | 763.2 |
| 03:29:07 | 0.150 | 762.8 |

The inference timings exclude model loading and are not end-to-end preview FPS.

Compared with the first depth observation, the later processed RGB images differ
by a mean absolute 1.77 and 1.42 intensity levels on a 0-255 channel scale.
Assuming the camera and scene stayed fixed, this supports approximate pixelwise
comparison; no image registration was performed.

For each later depth map D, fit a diagnostic multiplicative scale using
s = median(D_first / D), then measure abs(s*D - D_first) / D_first.

| Comparison with first map | Scale factor | Median relative difference | 95th percentile |
| --- | ---: | ---: | ---: |
| 03:28:54 | 1.00760 | 0.423% | 1.121% |
| 03:29:07 | 1.00291 | 0.261% | 0.659% |

These are same-scene consistency measurements, not metric accuracy or
multi-view consistency. A consistently wrong prediction can pass this check.

As a local shape diagnostic, fit inverse depth as an affine function of image
coordinates on a table-only interior ROI: x in [15%,85%), y in [10%,45%). A
plane has this form under an ideal undistorted pinhole camera with axial depth.
Median relative residuals are 0.20-0.26%; 95th-percentile residuals are
0.53-0.67%. This supports local smoothness/approximate planarity only. It does
not validate the whole table, camera distortion, scale, or object geometry.

## Next controlled capture

Use a matte box with measured dimensions and a second box as an occluder. Keep
lighting and the scene fixed. Capture repeat observations from each camera,
then compare resolution 252 versus 504 on the exact same saved RGB image.
Check separation from the table, straight box faces, edge bleeding, shadow
artifacts, latency, and memory. Use a calibration board before any metric
reconstruction claim. Overhead depth has not yet been captured in this dataset.
