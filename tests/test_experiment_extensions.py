"""Fast regression tests for research ablations, payload and protocol."""
from dataclasses import replace
import numpy as np
import pytest
from qr64_certified.msdp import MSDPConfig,embed_msdp,extract_msdp
from qr64_certified.sparse import encode_sparse,extract_sparse
from qr64_certified.research import protocol_hash,aggregate
from qr64_certified.attacks import get_attack_suite,apply_attack

@pytest.fixture(scope='module')
def synthetic():
    rng=np.random.default_rng(151)
    host=rng.integers(35,220,size=(64,64,3),dtype=np.uint8)
    watermark=(rng.integers(0,2,size=(8,8),dtype=np.uint8)*255)
    return host,watermark

@pytest.mark.parametrize('selector',['minimax','fixed','random','max_abs'])
def test_ablation_selectors_are_decodable_and_message_independent(synthetic,selector):
    host,wm=synthetic; cfg=MSDPConfig(pattern_selector=selector)
    marked,key,meta=embed_msdp(host,wm,seed=2026,config=cfg)
    assert np.array_equal(extract_msdp(marked,key),wm)
    _,key_flip,_=embed_msdp(host,255-wm,seed=2026,config=cfg)
    assert key.pattern_codes==key_flip.pattern_codes
    assert meta['certified_integer_constraint_violations']==0

@pytest.mark.parametrize('placement',['random','low_variance','high_variance'])
def test_sparse_recovery_uses_only_compact_key(synthetic,placement):
    host,wm=synthetic
    marked,key,pos,meta=encode_sparse(host,wm,fraction=.5,placement=placement,seed=2026)
    got,positions=extract_sparse(marked,key)
    np.testing.assert_array_equal(positions,pos)
    np.testing.assert_array_equal(got,wm.reshape(-1)[positions])
    assert meta['sparse_pattern_map_bytes'] < 24


def test_incomplete_reporting_and_protocol(synthetic):
    suite=get_attack_suite('common20')
    assert len(suite)==20 and len(set(a.attack_id for a in suite))==20
    assert protocol_hash('common20')==protocol_hash('common20')
    host,_=synthetic
    for attack in suite[:3]:
        out=apply_attack(host,attack)
        assert out.shape==host.shape and out.dtype==np.uint8
    summaries=aggregate([{'variant':'a','status':'error','attack_id':'jpeg'},
                         {'variant':'a','status':'ok','attack_id':'none__clean','clean_psnr_db':50,'clean_nc':1}])
    assert summaries[0]['failed_rows']==1
    assert summaries[0]['mean_attack_nc'] is None


def test_config_validation():
    with pytest.raises(ValueError):MSDPConfig(pattern_selector='bad').validate()


def test_basic20x3_exact_61_protocol(synthetic):
    from collections import Counter
    from qr64_certified.research import measure_trial
    attacks = get_attack_suite('basic20x3')
    assert len(attacks) == 60
    assert len({attack.attack_id for attack in attacks}) == 60
    assert Counter(a.category for a in attacks) == {
        'compression': 12, 'noise': 12, 'filtering': 12,
        'geometric': 12, 'photometric': 12,
    }
    assert Counter(a.severity for a in attacks) == {'mild': 20, 'moderate': 20, 'strong': 20}
    assert all(a.group != 'none' for a in attacks)
    expected = {
        'jpeg': ('quality', (90, 70, 50)),
        'jpeg2000': ('quality_layer', (3.0, 7.0, 10.0)),
        'bit_depth': ('bits', (7, 6, 5)),
        'color_quantization': ('colors', (128, 64, 32)),
        'gaussian_noise': ('variance', (.001, .003, .005)),
        'salt_pepper': ('amount', (.001, .005, .01)),
        'speckle': ('variance', (.001, .005, .01)),
        'poisson': ('peak', (255, 128, 64)),
        'median': ('size', (3, 5, 7)),
        'average': ('size', (3, 5, 7)),
        'gaussian_blur': ('radius', (.5, 1., 1.5)),
        'sharpen': ('factor', (1.25, 1.5, 2.)),
        'direct_rotation': ('degrees', (5., 10., 15.)),
        'resize': ('factor', (.9, .75, .5)),
        'crop_zero_fill': ('keep', (.95, .9, .8)),
        'brightness': ('factor', (.95, .9, .8)),
        'contrast': ('factor', (.95, .9, .8)),
        'gamma': ('gamma', (1.05, 1.2, 1.5)),
        'saturation': ('factor', (.8, .5, .2)),
    }
    for name, (field, vals) in expected.items():
        assert tuple(next(x for x in attacks if x.attack_id == f'basic20x3_{name}_{level}').params[field]
                     for level in ('mild', 'moderate', 'strong')) == vals
    for level, shift in zip(('mild','moderate','strong'), (2, 4, 8)):
        item=next(x for x in attacks if x.attack_id==f'basic20x3_translation_{level}')
        assert item.params['shift_x'] == item.params['shift_y'] == shift
    host, wm = synthetic
    rows = measure_trial(host, wm, config=MSDPConfig(), seed=2026,
                         suite='basic20x3', attacked_quality=False)
    assert len(rows) == 61
    assert sum(r['attack_id'] == 'none__clean' for r in rows) == 1
    assert sum(r['status'] == 'error' for r in rows) == 0
    assert all('attacked_ssim' not in r for r in rows)
    assert 'clean_ssim' in rows[0]


def test_crop_zero_fill_keeps_pixels_not_resize():
    img=np.full((40,60,3),128,dtype=np.uint8)
    attack=next(x for x in get_attack_suite('basic20x3') if x.attack_id=='basic20x3_crop_zero_fill_moderate')
    out=apply_attack(img,attack)
    assert out.shape==img.shape
    assert np.array_equal(out[2:38,3:57],img[2:38,3:57])
    assert np.all(out[0,:,:]==0) and np.all(out[:,-1,:]==0)


def test_trial_journal_resume_and_protocol_guard(tmp_path):
    from qr64_certified.research import TrialJournal
    spec={'suite':'basic20x3','seed':2026}
    j=TrialJournal(tmp_path,spec=spec)
    j.append('trial-A',[{'attack_id':'none__clean','status':'ok'}])
    k=TrialJournal(tmp_path,spec=spec)
    assert k.has('trial-A')
    assert len(k.data)==1
    with pytest.raises(ValueError):
        TrialJournal(tmp_path,spec={'suite':'common20','seed':2026})
