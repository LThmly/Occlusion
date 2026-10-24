import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from occlusion.alignment import (align_correspondences, fit_pair, fit_similarity,
                                 load_pair, pair_identity, sample_point)
from occlusion.alignment_ui import MatchPicker, capture_pair, receive_pair
from occlusion.depth import DepthResult
from occlusion.outputs import save_result
from occlusion.visualizer import log_frame, project_cloud


def known_points(count=18):
    rng = np.random.default_rng(17)
    source = rng.uniform([-1, -1, 1], [1, 1, 3], (count, 3))
    rotation = cv2.Rodrigues(np.array([0.3, -0.2, 0.6]))[0]
    scale, translation = 1.7, np.array([0.2, -0.4, 0.8])
    target = scale * source @ rotation.T + translation
    return source, target, scale, rotation, translation


def test_similarity_recovers_scale_rotation_translation():
    source, target, scale, rotation, translation = known_points()
    estimated = fit_similarity(source, target)
    np.testing.assert_allclose(estimated[0], scale, atol=1e-12)
    np.testing.assert_allclose(estimated[1], rotation, atol=1e-12)
    np.testing.assert_allclose(estimated[2], translation, atol=1e-12)
    assert np.linalg.det(estimated[1]) == pytest.approx(1)


def test_similarity_does_not_fit_reflections():
    source, _, _, _, _ = known_points()
    target = source * [-1, 1, 1]
    scale, rotation, translation = fit_similarity(source, target)
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert np.mean(np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1)) > 0.1


def test_ransac_rejects_training_outlier_without_using_validation_points():
    source, target, _, _, _ = known_points()
    target[1] += [4, 0, 0]  # Training outlier.
    _, _, _, report = align_correspondences(source, target)
    assert report["accepted"]
    assert 1 not in report["training_inlier_indices_zero_based"]
    assert report["validation_max_error_normalized"] < 1e-10
    baseline = align_correspondences(source, target)[:3]
    target[0] += [5, 0, 0]  # Held-out outlier must fail, not change the fitted transform.
    changed = align_correspondences(source, target)
    assert not changed[3]["accepted"]
    for first, second in zip(baseline, changed[:3]):
        np.testing.assert_allclose(first, second)


def test_alignment_rejects_collinear_and_insufficient_matches():
    line = np.column_stack((np.arange(12), np.zeros((12, 2))))
    with pytest.raises(ValueError, match="collinear"):
        align_correspondences(line, line)
    with pytest.raises(ValueError, match="nine"):
        align_correspondences(line[:8], line[:8])


def test_planar_correspondences_fit_but_warn():
    source, _, _, rotation, translation = known_points()
    source[:, 2] = 2
    target = 2 * source @ rotation.T + translation
    _, _, _, report = align_correspondences(source, target)
    assert report["accepted"]
    assert report["warnings"]


def test_sampling_rejects_invalid_depth_and_outside_pixels():
    result = DepthResult(np.zeros((3, 4, 3), dtype=np.uint8), np.ones((3, 4)),
                         None, np.eye(3), None, 0)
    np.testing.assert_allclose(sample_point(result, [2, 1]), [2, 1, 1])
    result.depth[1, 2] = np.nan
    with pytest.raises(ValueError, match="invalid depth"):
        sample_point(result, [2, 1])
    with pytest.raises(ValueError, match="outside"):
        sample_point(result, [-0.1, 1])


def make_saved_pair(root):
    rng = np.random.default_rng(7)
    rgb = np.full((24, 32, 3), [120, 50, 80], dtype=np.uint8)
    depth = rng.uniform(1, 2, (24, 32)).astype(np.float32)
    intrinsics = np.array([[30, 0, 16], [0, 30, 12], [0, 0, 1]], dtype=float)
    for name, scale in (("wrist", 1), ("overhead", 2)):
        directory = root / name
        directory.mkdir(parents=True)
        result = DepthResult(rgb, depth * scale, None, intrinsics, None, 0.1)
        save_result(directory, rgb, result, {})
    pixels = [[3, 3], [10, 3], [20, 3], [28, 4], [4, 10], [12, 10],
              [22, 10], [28, 12], [3, 18], [10, 20], [20, 20], [28, 20]]
    matches = {"schema_version": 1, "capture_identity": pair_identity(root),
               "pairs": [{"wrist": uv, "overhead": uv} for uv in pixels]}
    path = root / "matches.json"
    path.write_text(json.dumps(matches))
    return path


def test_saved_pair_fit_is_compatible_with_viewer_and_hash_bound(tmp_path):
    pair = tmp_path / "pair"
    matches = make_saved_pair(pair)
    output = tmp_path / "geometry.json"
    report = fit_pair(pair, matches, output)
    assert report["accepted"]
    geometry = json.loads(output.read_text())
    assert geometry["wrist"]["depth_scale"] == pytest.approx(2)
    results = load_pair(pair)
    clouds = []
    for name in ("wrist", "overhead"):
        result, settings = results[name], geometry[name]
        points, _ = project_cloud(result.rgb, result.depth, settings["intrinsics"],
                                  settings["world_from_camera"], scale=settings["depth_scale"], stride=2)
        clouds.append(points)
    np.testing.assert_allclose(*clouds, atol=1e-7)
    assert not geometry["_alignment"]["scale_validated"]
    np.save(pair / "wrist" / "depth.npy", np.ones((24, 32)))
    with pytest.raises(ValueError, match="exact saved pair"):
        fit_pair(pair, matches, tmp_path / "changed.json")


def test_failed_validation_writes_report_without_geometry(tmp_path):
    pair = tmp_path / "pair"
    matches_path = make_saved_pair(pair)
    matches = json.loads(matches_path.read_text())
    matches["pairs"][0]["overhead"] = [30, 22]
    matches_path.write_text(json.dumps(matches))
    output = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="Held-out"):
        fit_pair(pair, matches_path, output)
    assert not output.exists()
    assert not json.loads(output.with_suffix(".report.json").read_text())["accepted"]


def test_inconsistent_training_returns_rejected_diagnostic_report():
    source, target, _, _, _ = known_points()
    rng = np.random.default_rng(15)
    target[[1, 2, 4, 5, 7, 8]] += rng.uniform(-1, 1, (6, 3))
    _, _, _, report = align_correspondences(source, target)
    assert not report["accepted"]
    assert not report["training_support_passed"]
    assert any("Training" in reason for reason in report["failure_reasons"])
    assert len(report["per_pair_errors_normalized"]) == len(source)


def test_picker_orders_points_and_undoes_pending_or_complete_pair():
    result = DepthResult(np.zeros((24, 32, 3), dtype=np.uint8), np.ones((24, 32)),
                         None, np.eye(3), None, 0)
    picker = MatchPicker({"wrist": result, "overhead": result})
    wx, wy, _, _ = picker.layouts["wrist"]
    ox, oy, _, _ = picker.layouts["overhead"]
    picker.click(ox+50, oy+50)  # Cannot start on overhead.
    assert not picker.pairs and picker.pending is None
    picker.click(wx+50, wy+50)
    assert picker.pending is not None
    picker.click(ox+50, oy+50)
    assert len(picker.pairs) == 1 and picker.pending is None
    picker.click(wx+100, wy+100)
    picker.undo()
    assert len(picker.pairs) == 1 and picker.pending is None
    picker.undo()
    assert not picker.pairs
    assert picker.draw().shape[2] == 3


def test_picker_zoom_preserves_original_pixel_grid_and_can_reset():
    result = DepthResult(np.zeros((24, 32, 3), dtype=np.uint8), np.ones((24, 32)),
                         None, np.eye(3), None, 0)
    picker = MatchPicker({"wrist": result, "overhead": result})
    left, top, width, height = picker.layouts["wrist"]
    picker.zoom(left+width//2, top+height//2, 2)
    x0, y0, rw, rh = picker.viewports["wrist"]
    assert rw == 16 and rh == 12
    pixel = [14, 11]
    picker.click(left+int((pixel[0]-x0+0.5)*width/rw),
                 top+int((pixel[1]-y0+0.5)*height/rh))
    assert picker.pending == pixel
    assert picker.viewports["overhead"] == (0, 0, 32, 24)
    assert picker.draw().shape[2] == 3
    picker.reset_zoom()
    assert picker.viewports["wrist"] == (0, 0, 32, 24)


def test_picker_warns_when_clicked_depth_is_near_a_discontinuity():
    depth = np.ones((24, 32))
    depth[:, 16:] = 2
    result = DepthResult(np.zeros((24, 32, 3), dtype=np.uint8), depth, None, np.eye(3), None, 0)
    picker = MatchPicker({"wrist": result, "overhead": result})
    assert picker.depth_edge("wrist", [15, 10])
    assert not picker.depth_edge("wrist", [5, 10])


def test_receive_pair_grabs_both_before_decoding():
    events = []

    class Camera:
        def __init__(self, name):
            self.name = name

        def grab(self):
            events.append(("grab", self.name))

        def retrieve_rgb(self):
            events.append(("retrieve", self.name))
            return np.zeros((4, 5, 3), dtype=np.uint8)

    frames, timing = receive_pair({name: Camera(name) for name in ("wrist", "overhead")})
    assert events == [("grab", "wrist"), ("grab", "overhead"), ("retrieve", "wrist"), ("retrieve", "overhead")]
    assert frames["wrist"].shape == (4, 5, 3)
    assert timing["wrist"]["exposure_timestamp"] is None


def test_capture_pair_records_both_frames_before_inference_and_releases_cameras(tmp_path, monkeypatch):
    import occlusion.alignment_ui as ui
    events = []

    class Camera:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *_):
            events.append("release")

        def grab(self):
            events.append("grab")

        def retrieve_rgb(self):
            events.append("retrieve")
            return np.full((24, 32, 3), 128, dtype=np.uint8)

    class Estimator:
        def __init__(self, *args):
            assert events[-2:] == ["release", "release"]

        def predict(self, rgb):
            events.append("inference")
            return DepthResult(rgb, np.ones((24, 32)), None, np.eye(3), None, 0.1)

    monkeypatch.setattr(ui, "USBCamera", Camera)
    monkeypatch.setattr(ui, "DA3Estimator", Estimator)
    args = SimpleNamespace(output=tmp_path / "pair", wrist_camera="2", overhead_camera="0",
                           warmup_frames=1, device="cpu", resolution=504)
    capture_pair(args)
    assert events.count("inference") == 2
    assert len(pair_identity(args.output)) == 2
    metadata = json.loads((args.output / "wrist" / "metadata.json").read_text())
    assert metadata["camera_id"] == "wrist"
    assert "T" in metadata["captured_at_utc"]


def test_viewer_rejects_wrong_processed_resolution_before_logging():
    result = DepthResult(np.zeros((4, 5, 3)), np.ones((4, 5)), None, np.eye(3), None, 0)
    with pytest.raises(ValueError, match="alignment image size"):
        log_frame(None, "wrist", result, {"image_size": [10, 8]}, SimpleNamespace())
