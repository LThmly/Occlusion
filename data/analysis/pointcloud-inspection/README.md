# Export inspection: overhead-prototype.ply

The user-exported PLY contains 12,348 colored vertices, matching its sidecar
metadata and the resolution-252 source observation sampled at stride 2.
All XYZ coordinates are finite and all camera-axis depths are positive.
RGB channels lie in [6,255]. The export is structurally valid.

`views.png` renders the exact exported cloud from two angles. `inspection.png`
adds a diagnostic fitted-table residual map. `metrics.json` records the numeric
checks. These images are renderings, not additional observations.

The bottle has clear relief, but a smoothed transition into the tabletop.
The robot, phone, and remote retain RGB appearance while much of their expected
height is absent. A robot interior ROI has median signed distance about -0.00012
relative units from a fitted table plane; a bottle interior ROI is about +0.201.
These are manually selected image regions, not object segmentations or physical
height measurements. Table-plane residuals also reflect learned shape error,
uncalibrated camera geometry, and potentially imperfect masks.

The sloping table in camera coordinates is not inherently an error: a tilted
camera sees a sloped table. Missing geometric relief around visible objects is
the more relevant problem. A single view cannot reconstruct hidden surfaces.
The cloud is an open, partial collection of points, not a closed solid model.

Use the existing same-image resolution-504 observation for the next comparison:
`data/analysis/overhead-504/20261024T033518.265299Z-97360a04`.
The depth preview at that resolution resolved the robot more clearly. Exporting
it will not establish metric scale or calibrate camera rays.
