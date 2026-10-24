import json

import numpy as np
import pytest

from occlusion.alignment import fit_pair, pair_identity
from occlusion.alignment_diagnostics import fixed_span_pass, inspect_alignment, jitter_pixels
from occlusion.depth import DepthResult
from occlusion.outputs import save_result


def test_integer_jitter_is_seeded_bounded_and_does_not_clip_edges():
    pixels = np.array([[10, 10], [20, 20]])
    first = jitter_pixels(pixels, (40, 40), 2, np.random.default_rng(3))
    second = jitter_pixels(pixels, (40, 40), 2, np.random.default_rng(3))
    np.testing.assert_array_equal(first, second)
    assert np.abs(first-pixels).max() <= 2
    np.testing.assert_array_equal(jitter_pixels(pixels, (40,40), 0, np.random.default_rng(0)), pixels)
    with pytest.raises(ValueError, match="outside"):
        jitter_pixels([[-1,0]], (40,40), 0, np.random.default_rng(0))


def test_fixed_span_validation_does_not_hide_error_with_a_larger_trial_span():
    errors = np.array([0.14, 0.01, 0.01, 0.02, 0.01, 0.01, 0.03, 0.01, 0.01])
    training = np.array([1,2,4,5,7,8])
    validation = np.array([0,3,6])
    assert not fixed_span_pass(errors,training,validation,1.0,0.05,0.1)
    assert fixed_span_pass(errors,training,validation,2.0,0.05,0.1)


def test_sensitivity_reproduces_known_fit_and_binds_report_to_capture(tmp_path):
    pair=tmp_path/'pair'
    rgb=np.zeros((32,40,3),dtype=np.uint8)
    matrix=np.array([[40,0,20],[0,40,16],[0,0,1]],dtype=float)
    for name,scale in [('wrist',1),('overhead',2)]:
        directory=pair/name
        directory.mkdir(parents=True)
        save_result(directory,rgb,DepthResult(rgb,np.full((32,40),scale,dtype=float),None,matrix,None,0),{})
    uv=[[5,5],[15,5],[25,5],[34,5],[5,14],[15,14],[25,14],[34,14],[5,25],[15,25],[25,25],[34,25]]
    matches=pair/'matches.json'
    matches.write_text(json.dumps({'schema_version':1,'capture_identity':pair_identity(pair),
                                   'pairs':[{'wrist':p,'overhead':p} for p in uv]}))
    output=tmp_path/'geometry.json'
    fit_pair(pair,matches,output)
    report_path=output.with_suffix('.report.json')
    summary,points,fitted,_,_=inspect_alignment(pair,matches,report_path,trials=1)
    np.testing.assert_allclose(points['overhead'],fitted,atol=1e-12)
    zero=summary['experiments'][0]['summary']
    assert zero['original_span_pass_count'] == 1
    assert zero['attempted_trials'] == 1
    assert len(summary['experiments']) == 4
    assert all(p['error_fraction_of_matched_span'] < 1e-10 for p in summary['per_point'])
    report=json.loads(report_path.read_text())
    report['capture_identity']={}
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='exact saved pair'):
        inspect_alignment(pair,matches,report_path,trials=1)
