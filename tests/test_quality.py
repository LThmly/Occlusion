import json

import cv2
import numpy as np
import pytest

from occlusion.quality import compare_observations, region_mask


def make_observation(path, size, scale, focal=50):
    path.mkdir()
    width, height = size
    rgb = np.full((48, 64, 3), [50, 100, 150], dtype=np.uint8)
    cv2.imwrite(str(path / "rgb.png"), rgb)
    depth = np.full((height, width), 2 * scale, dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    object_region = (xx * 64 / width >= 25) & (xx * 64 / width < 45) & (yy * 48 / height >= 15) & (yy * 48 / height < 35)
    depth[object_region] = 1.6 * scale
    np.save(path / "depth.npy", depth)
    matrix = np.array([[focal * width / 64, 0, width / 2],
                       [0, focal * height / 48, height / 2], [0, 0, 1]])
    np.savez(path / "prediction.npz", intrinsics=matrix)
    (path / "metadata.json").write_text(json.dumps({"depth_units": "relative", "process_resolution": width}))
    return path


@pytest.fixture
def regions():
    return {"reference_image_size": [64, 48],
            "table_boxes": [[0, 0, 20, 48], [48, 0, 64, 48]],
            "object_boxes": {"box": [[27, 17, 43, 33]]}}


def test_same_image_different_resolution_and_scale_preserve_relief(tmp_path, regions):
    first = make_observation(tmp_path / "low", (32, 24), 1)
    # Deliberately wrong second predicted intrinsics: comparison must share
    # the first observation's rays rather than introduce a focal-length change.
    second = make_observation(tmp_path / "high", (64, 48), 7, focal=500)
    report, maps = compare_observations([first, second], regions)
    assert [m.shape for m in maps] == [(24, 32), (48, 64)]
    for observation in report["observations"]:
        values = observation["objects"]["box"]["normalized_signed_distance_p10_p50_p90"]
        np.testing.assert_allclose(values, [0.2, 0.2, 0.2], atol=1e-6)
        assert observation["table_normalized_abs_residual_p50_p95"][1] < 1e-6
    assert report["units"] == "dimensionless_not_metric_height"
    assert not report["scale_validated"]


def test_comparison_rejects_different_original_image(tmp_path, regions):
    first = make_observation(tmp_path / "first", (32, 24), 1)
    second = make_observation(tmp_path / "second", (64, 48), 1)
    cv2.imwrite(str(second / "rgb.png"), np.zeros((48, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="same original RGB"):
        compare_observations([first, second], regions)


def test_comparison_rejects_sparse_valid_table(tmp_path, regions):
    observation = make_observation(tmp_path / "capture", (32, 24), 1)
    np.save(observation / "depth.npy", np.full((24, 32), np.nan))
    with pytest.raises(ValueError, match="fewer than 30"):
        compare_observations([observation], regions)


def test_regions_use_original_image_coordinates_and_reject_bad_boxes():
    pixels = np.array([[0, 0], [5, 5], [9, 9]])
    np.testing.assert_array_equal(region_mask(pixels, (10, 10), (100, 100), [[40, 40, 60, 60]]),
                                   [False, True, False])
    with pytest.raises(ValueError, match="outside"):
        region_mask(pixels, (10, 10), (100, 100), [[-1, 0, 50, 50]])
