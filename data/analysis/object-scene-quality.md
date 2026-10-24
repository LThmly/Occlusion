# Object-scene quality assessment

Source: `data/overhead-depth/20261024T033417.716243Z-83ee4947`.
The image contains a bottle, phone, remote, robot, and tabletop. Only overhead
captures of this arrangement were available for this assessment.

The original 640 x 480 RGB image resolves the objects sufficiently for this
initial test. The bottle is clipped by the top edge. Phone and remote have low
height above the table and dark surfaces, making them useful challenge cases.

## Same-image processing comparison

The exact saved RGB image was reprocessed offline with DA3-Small at resolution
504. No camera or robot movement was performed.

| Setting | Depth array | Inference wall time |
| --- | --- | --- |
| 252, saved camera observation | 196 x 252 | 0.066 s |
| 504, new offline observation | 378 x 504 | 0.340 s |

These are individual measurements, not a controlled repeated latency benchmark.
They exclude model loading. Both maps contain finite positive depth everywhere;
this does not imply accurate geometry.

At 252, the bottle is evident, the phone and remote have weak contrast, and
most of the robot merges with the table gradient in the visualization.

At 504, the robot's silhouette and cable are much more clearly represented.
Bottle boundaries are more detailed. The remote remains a broad, smoothed shape
and the phone remains weakly distinguished from the table. Thin object height
and boundaries are still unresolved challenges. The RGB and depth previews use
independent percentile normalization, so this is qualitative feature assessment,
not a direct quantitative comparison of preview colors or depth magnitudes.

The strong table-depth gradient can reflect camera tilt. Without camera
calibration and metric ground truth it should not be labeled either correct
geometry or an artifact. Relative depth does not provide object dimensions.

## Decision and next experiment

Use resolution 504 for the next geometry-quality captures while retaining 252
for faster preview. Higher resolution helps this scene substantially but does
not establish metric accuracy or reliable reconstruction of thin objects.

Place a matte box with measured dimensions fully inside the image. Include a
second object partially hidden from the wrist camera and acquire both wrist
and overhead observations without changing the scene. This will test object
separation and actual occlusion. A known calibration target is needed before
testing metric shape or fusing these cameras.

504 output:
`data/analysis/overhead-504/20261024T033518.265299Z-97360a04/preview.png`.
