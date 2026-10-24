# Shared camera experiments

These small capture sets are checked in so development can proceed without
connected cameras. Model caches and the upstream sample-image smoke tests remain
local and ignored.

| Directory | Contents |
| --- | --- |
| `logitech/` | Six overhead-camera RGB snapshots |
| `innomaker/` | Wrist-camera RGB snapshot |
| `depth-test/` | Three wrist observations with DA3-Small depth at resolution 252 |
| `overhead-depth/` | Two object-scene overhead observations at resolution 252 |
| `analysis/` | Quality reports and the same-image resolution-504 comparison |

The camera setup for these sessions was Logitech C922 overhead (`/dev/video0`)
and Innomaker U20CAM wrist (`/dev/video2`). Device numbers can change on reconnect.

Replay the object scene after completing the root README's inference setup:

```bash
occlusion --image data/overhead-depth/20261024T033417.716243Z-83ee4947/rgb.png \
  --resolution 504 --no-display --output data/local-replay
```

`data/local-replay` is ignored. Saved raw depth is relative, not meters. The
dataset has no calibrated camera poses, metric ground truth, or synchronized
cross-camera observations. Acquisition timestamps are UTC; original capture
metadata is preserved. Timing measurements describe the recorded execution
environment and should be remeasured on each teammate's machine.

Read `analysis/initial-camera-quality.md` and `analysis/object-scene-quality.md`
for findings and limitations. Useful next work is camera-intrinsic calibration,
calibrated single-view back-projection, or offline geometry tests. Coordinate
frame and scale contracts are defined in the root README. Robot movement is
not implemented and requires explicit approval before physical testing.
