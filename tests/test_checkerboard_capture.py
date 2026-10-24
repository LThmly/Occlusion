import json

import cv2
import numpy as np
import pytest

from occlusion.calibration import write_screen_board
from occlusion.checkerboard_capture import detect_board, draw_feedback, save_board_pair


def test_ipad_target_has_white_border_and_54_detectable_corners(tmp_path):
    path=tmp_path/'ipad.png'
    write_screen_board(path)
    image=cv2.imread(str(path))
    assert image.shape == (1800,2400,3)
    assert (image[:200] == 255).all()
    assert (image[-200:] == 255).all()
    assert (image[:,:200] == 255).all()
    assert (image[:,-200:] == 255).all()
    rgb=cv2.resize(image,(640,480))
    corners=detect_board(rgb,9,6)
    assert corners is not None
    assert corners.shape == (54,1,2)


def board_pair():
    frames={'wrist':np.full((48,64,3),50,dtype=np.uint8),
            'overhead':np.full((48,64,3),150,dtype=np.uint8)}
    corners={name:np.mgrid[0:9,0:6].T.reshape(-1,1,2).astype(np.float32) for name in frames}
    timing={name:{'host_received_ns':time,'exposure_timestamp':None}
            for name,time in [('wrist',1000000),('overhead',2000000)]}
    return frames,corners,timing


def test_board_pair_save_preserves_raw_rgb_pairing_and_rejects_duplicates(tmp_path):
    frames,corners,timing=board_pair()
    seen={name:set() for name in frames}
    identifier=save_board_pair(tmp_path,frames,corners,timing,9,6,seen)
    manifest=json.loads((tmp_path/'pairs'/identifier/'pair.json').read_text())
    assert manifest['host_receive_skew_ms'] == 1
    for name in frames:
        image=cv2.cvtColor(cv2.imread(str(tmp_path/manifest['images'][name])),cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(image,frames[name])
        np.testing.assert_array_equal(np.load(tmp_path/name/identifier/'corners.npy'),corners[name])
    with pytest.raises(ValueError,match='identical'):
        save_board_pair(tmp_path,frames,corners,timing,9,6,seen)


def test_missing_or_incomplete_board_is_not_saved(tmp_path):
    frames,corners,timing=board_pair()
    corners['overhead']=None
    with pytest.raises(ValueError,match='BOTH'):
        save_board_pair(tmp_path,frames,corners,timing,9,6,{name:set() for name in frames})
    assert not (tmp_path/'pairs').exists()
    corners['overhead']=np.zeros((10,1,2))
    with pytest.raises(ValueError,match='complete'):
        save_board_pair(tmp_path,frames,corners,timing,9,6,{name:set() for name in frames})


def test_feedback_can_show_one_missing_board_without_modifying_raw_images():
    frames,corners,_=board_pair()
    before=frames['wrist'].copy()
    corners['overhead']=None
    canvas=draw_feedback(frames,corners,9,6,3,'Hold still')
    assert canvas.shape == (625,1280,3)
    np.testing.assert_array_equal(frames['wrist'],before)
