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
