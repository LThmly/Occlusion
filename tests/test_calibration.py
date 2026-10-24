import json

import cv2
import numpy as np
import pytest

from occlusion.calibration import board_points, fit_intrinsics, load_calibration, write_board_pdf
from occlusion.pointcloud import export_observation


def synthetic_views(count=20):
    matrix = np.array([[600, 0, 320], [0, 610, 240], [0, 0, 1]], dtype=float)
    obj = board_points(9, 6, 0.02)
    rng = np.random.default_rng(42)
    corners = []
    for _ in range(count):
        rvec = rng.uniform(-0.45, 0.45, 3)
        tvec = np.array([rng.uniform(-0.13, -0.03), rng.uniform(-0.1, 0.01), rng.uniform(0.5, 0.85)])
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, np.zeros(5))
        corners.append(projected.astype(np.float32))
    return matrix, corners


def test_calibration_recovers_known_intrinsics_with_heldout_views(tmp_path):
    matrix, corners = synthetic_views()
    result = fit_intrinsics(corners, (640, 480), 9, 6, 0.02)
    assert result["validation_passed"]
    assert len(result["training_indices"]) == 16
    assert len(result["heldout_indices"]) == 4
    np.testing.assert_allclose(result["intrinsics"], matrix, atol=0.03)
    assert max(result["heldout_rms_px"]) < 0.001
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(result))
    assert load_calibration(path)["image_size"] == [640, 480]


def test_insufficient_views_and_bad_calibration_are_rejected(tmp_path):
    _, corners = synthetic_views(10)
    with pytest.raises(ValueError, match="training views"):
        fit_intrinsics(corners, (640, 480), 9, 6, 0.02)
    matrix, corners = synthetic_views()
    rng = np.random.default_rng(4)
    corners[0] += rng.normal(0, 5, corners[0].shape).astype(np.float32)
    result = fit_intrinsics(corners, (640, 480), 9, 6, 0.02)
    assert not result["validation_passed"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="validation"):
        load_calibration(path)


def test_board_dimensions_and_pdf(tmp_path):
    obj = board_points(9, 6, 0.02)
    assert obj.shape == (54, 3)
    np.testing.assert_allclose(obj[-1], [0.16, 0.10, 0])
    path = tmp_path / "board.pdf"
    write_board_pdf(path)
    assert path.read_bytes().startswith(b"%PDF-1.4")
    assert b"100 mm" in path.read_bytes()
    with pytest.raises(ValueError, match="fit A4"):
        write_board_pdf(path, square_mm=50)


def test_pointcloud_export_labels_relative_units_and_preserves_colors(tmp_path):
    observation = tmp_path / "capture"
    observation.mkdir()
    np.save(observation / "depth.npy", np.array([[2, np.nan], [3, 4]], dtype=np.float32))
    np.savez(observation / "prediction.npz", intrinsics=np.eye(3))
    rgb = np.full((2, 2, 3), [50, 100, 200], dtype=np.uint8)
    cv2.imwrite(str(observation / "rgb.png"), rgb[:, :, ::-1])
    cv2.imwrite(str(observation / "rgb_processed.png"), rgb[:, :, ::-1])
    (observation / "metadata.json").write_text(json.dumps({"depth_units": "relative", "original_shape": [2, 2, 3]}))
    output = tmp_path / "cloud.ply"
    details = export_observation(observation, output, predicted_intrinsics=True, stride=1)
    assert details["point_count"] == 3
    assert details["units"] == "relative_not_meters"
    assert not details["suitable_for_robot_control"]
    assert "50 100 200" in output.read_text()
    with pytest.raises(ValueError, match="already exists"):
        export_observation(observation, output, predicted_intrinsics=True)


def test_calibrated_export_refuses_wrong_camera_or_resolution(tmp_path):
    matrix, corners = synthetic_views()
    calibration = fit_intrinsics(corners, (640, 480), 9, 6, 0.02)
    calibration["camera_id"] = "wrist"
    path = tmp_path / "wrist.json"
    path.write_text(json.dumps(calibration))
    observation = tmp_path / "capture"
    observation.mkdir()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    for name in ("rgb.png", "rgb_processed.png"):
        cv2.imwrite(str(observation / name), image)
    np.save(observation / "depth.npy", np.ones((2, 2)))
    (observation / "metadata.json").write_text(json.dumps({"depth_units": "relative", "original_shape": [2, 2, 3]}))
    with pytest.raises(ValueError, match="identity"):
        export_observation(observation, tmp_path / "cloud.ply", path, "overhead")
    with pytest.raises(ValueError, match="resolution"):
        export_observation(observation, tmp_path / "cloud.ply", path, "wrist")


def test_calibrated_export_scales_intrinsics_to_depth_grid(tmp_path):
    _, corners = synthetic_views()
    calibration = fit_intrinsics(corners, (640, 480), 9, 6, 0.02)
    calibration["camera_id"] = "overhead"
    path = tmp_path / "overhead.json"
    path.write_text(json.dumps(calibration))
    observation = tmp_path / "capture"
    observation.mkdir()
    cv2.imwrite(str(observation / "rgb.png"), np.zeros((480, 640, 3), dtype=np.uint8))
    cv2.imwrite(str(observation / "rgb_processed.png"), np.zeros((240, 320, 3), dtype=np.uint8))
    np.save(observation / "depth.npy", np.full((240, 320), 2, dtype=np.float32))
    (observation / "metadata.json").write_text(json.dumps({"depth_units": "relative", "original_shape": [480, 640, 3]}))
    details = export_observation(observation, tmp_path / "cloud.ply", path, "overhead", stride=160)
    assert details["point_count"] == 4
    assert details["distortion_corrected_in_rays"]
    assert len(details["calibration_sha256"]) == 64
    np.testing.assert_allclose(details["processed_intrinsics"],
                               [[300, 0, 160], [0, 305, 120], [0, 0, 1]], atol=0.03)
