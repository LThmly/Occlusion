import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from occlusion.camera import USBCamera, read_rgb
from occlusion.depth import DepthResult
from occlusion.main import main
from occlusion.outputs import colorize_depth, new_capture_dir, save_result


def test_saved_depth_is_lossless_and_rgb_is_aligned(tmp_path):
    original = np.full((4, 6, 3), [255, 0, 40], dtype=np.uint8)
    rgb = original[::2, ::2].copy()
    depth = np.array([[1.123456, np.nan, 3], [4, 0, 7]], dtype=np.float32)
    result = DepthResult(rgb, depth, None, np.eye(3), None, 0.5)
    directory = new_capture_dir(tmp_path)
    preview = save_result(directory, original, result, {"device": "cpu"})
    np.testing.assert_equal(np.load(directory / "depth.npy"), depth)
    np.testing.assert_equal(read_rgb(directory / "rgb_processed.png"), rgb)
    np.testing.assert_equal(read_rgb(directory / "rgb.png"), original)
    assert preview.shape == (2, 6, 3)
    assert json.loads((directory / "metadata.json").read_text())["depth_units"] == "relative"
    with np.load(directory / "prediction.npz", allow_pickle=False) as data:
        assert data.files == ["intrinsics"]


@pytest.mark.parametrize("depth", [np.zeros((3, 4)), np.ones((3, 4)), np.full((3, 4), np.nan)])
def test_colorization_handles_empty_or_constant_depth(depth):
    assert colorize_depth(depth).shape == (3, 4, 3)


def test_camera_converts_bgr_and_releases_on_failure(monkeypatch):
    released = []
    capture = SimpleNamespace(isOpened=lambda: True,
                              read=lambda: (False, None),
                              release=lambda: released.append(True))
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(RuntimeError, match="frame"):
        with USBCamera() as camera:
            camera.read()
    assert released == [True]


def test_capture_only_cli_does_not_load_model(tmp_path, monkeypatch):
    import occlusion.depth
    monkeypatch.setattr(occlusion.depth, "DA3Estimator", lambda *a: pytest.fail("Model loaded"))
    image = tmp_path / "input.png"
    cv2.imwrite(str(image), np.full((20, 30, 3), 128, dtype=np.uint8))
    monkeypatch.setattr("sys.argv", ["occlusion", "--image", str(image), "--capture-only",
                                   "--no-display", "--output", str(tmp_path / "out")])
    main()
    assert len(list((tmp_path / "out").glob("*/rgb.png"))) == 1
