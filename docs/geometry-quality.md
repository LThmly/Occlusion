# Geometry work while calibration and the visualizer are pending

The current question is whether depth preserves useful object shape before we
attempt multi-view reconstruction. A visually recognizable point cloud can still
flatten an object onto the table. The quality diagnostic measures this explicitly
using manually annotated table and object regions. It has no UI or robot access.

## Reproduce the same-image resolution comparison

```bash
python -m occlusion.quality \
  --observations \
    data/overhead-depth/20261024T033417.716243Z-83ee4947 \
    data/analysis/overhead-504/20261024T033518.265299Z-97360a04 \
  --regions configs/quality/overhead-object-scene.json \
  --output data/quality-runs/resolution-comparison
```

The output directory must be new. `report.json` records the inputs, original RGB
hash, regions, camera rays, plane fit, object statistics, and limitations.
`relief.npz` contains one float32 map per observation, indexed in report order
as `observation_0`, `observation_1`, etc. Each map has the corresponding processed
depth shape; invalid depth pixels are NaN. These files can feed the Rerun viewer
without adding visualization dependencies to the geometry code.

The tool requires exactly the same original RGB pixels in all observations.
This isolates inference settings from camera or scene changes. It cannot compare
the overhead and wrist views this way. To evaluate a new scene, save one RGB
image, run inference at each desired resolution, and annotate new regions in a
separate config. Boxes use original RGB coordinates, not processed depth pixels.
The sample config applies only to the checked-in object arrangement.

## What is measured

1. Share the first observation's model-predicted intrinsics across all runs,
   scaling them to each processed depth size. Changing predicted focal length
   would otherwise confound the comparison.
2. Back-project valid depth and restrict table fitting to annotated table areas.
3. Fit a plane with deterministic three-point RANSAC and inlier SVD refinement.
   The fitting threshold defaults to 2% of median table axial depth, so it scales
   with arbitrary multiplicative depth units. It is a diagnostic setting to
   record, not a physical tolerance.
4. Orient its unit normal toward the camera. If the plane is
   $n^Tp+d=0$, camera-to-plane distance is $d>0$.
5. Calculate normalized signed relief:

$$
h(p)=\frac{n^Tp+d}{d}.
$$

Positive values lie on the camera side of the fitted table. Global scaling of
all points scales the numerator and denominator equally, so this quantity is
dimensionless. It does not remove local depth errors or incorrect camera rays.

The report includes table residual percentiles, object-region relief percentiles,
and the fraction of an object's samples above the table's 95th-percentile absolute
residual. This last value is an empirical separation indicator, not a detection
probability, calibrated confidence, or ground-truth accuracy measure. Manual
regions are not complete object masks. A failed fit is rejected if it lacks a
majority plane or has insufficient table support.

No smoothing or synthetic object extrusion is applied. Changing geometry merely
to make objects appear taller would hide the failure we need to measure.

## Findings on the shared scene

The two resolutions were generated from identical RGB pixels. Plane fitting and
normalization use the same rules at both resolutions.

| Diagnostic | Resolution 252 | Resolution 504 |
| --- | ---: | ---: |
| Median normalized bottle relief | 0.2630 | 0.2974 |
| Median normalized robot relief | 0.0034 | 0.1438 |
| Median normalized phone relief | 0.0253 | 0.0225 |
| Median normalized remote relief | 0.0139 | 0.0080 |
| Table absolute residual, 95th percentile | 0.0431 | 0.0179 |

The robot has much stronger geometric separation at 504. The phone's relief did
not increase, but the lower table residual makes it more distinguishable. The
remote remains within the table residual envelope. These findings support 504
for current shape experiments, not a general accuracy claim or a promise of
metric performance. Recorded inference times were 0.066 and 0.340 seconds;
they are single measurements from different runs, excluding model loading.

Generated comparison artifacts are under `data/analysis/resolution-comparison/`.
The directory also contains `overhead-504-relative.ply` with 47,628 colored points
and its provenance sidecar. That viewer-ready PLY uses the 504 observation's own
predicted intrinsics, as the standard exporter does. The resolution diagnostic
instead shares the 252 observation's rays across both resolutions to isolate
depth changes; its plane measurements therefore must not be applied directly to
the separately exported PLY without recomputing the fit in that cloud's frame.

## Next experiment

Capture a matte box with measured dimensions and a partial occluder from the
wrist. Keep both objects stationary. Run the same saved image at 252 and 504,
annotate table/object interiors, and assess whether the box remains geometrically
distinct. A real camera-angle change is a separate experiment requiring camera
pose and scale reasoning. Calibration and metric validation remain necessary
before reliable multi-view fusion or physical planning.

RANSAC plane terminology follows the standard
[point-cloud plane segmentation formulation](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#plane-segmentation).
The small NumPy implementation avoids an additional Open3D dependency in the
CPU-only core; it does not replace a full reconstruction library.
