"""Numerical and integration tests for the selected determinant method."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard

from qr64_certified.msdp import (
    MSDPConfig, W, W2, _affine_slack, _means, _normalized_residual,
    _continuous_project, _project_unbounded, _codebook, _round_and_repair,
    embed_msdp, extract_msdp, key_to_jsonable, key_from_jsonable, verify_msdp,
)
from qr64_certified.baselines import ALL_BASELINE_IDS, embed_baseline, extract_baseline
from qr64_certified.common.io import load_host_rgb, load_watermark_binary

ROOT=Path(__file__).resolve().parents[1]


def test_normalized_determinant_equivalence():
    rng=np.random.default_rng(12)
    p=hadamard(64)[3].astype(np.float64)
    mu=0.07
    kappa=mu/math.sqrt(2-mu**2)
    for _ in range(80):
        x=rng.integers(0,256,(64,3)).astype(np.float64)
        u,v=_means(x,p,W)
        residual=_normalized_residual(x,p,W)
        np.testing.assert_allclose(residual,(u-v)/math.hypot(u,v),atol=1e-12)
        for sign in (-1.0,1.0):
            assert (sign*residual>=mu)==(sign*(u-v)>=kappa*(u+v))


def test_unbounded_projection_formula_and_minimax_rule():
    rng=np.random.default_rng(333)
    x=rng.integers(65,180,(64,3)).astype(np.float64)
    patterns=_codebook(8)
    mu=0.015
    k=mu/math.sqrt(2-mu*mu)
    lam=2.0
    floor=0.25
    d=np.array([_means(x,p,W)[0]-_means(x,p,W)[1] for p in patterns])
    selected=int(np.argmin(np.abs(d)))
    costs=[]
    for p in patterns:
        max_cost=0.0
        for s in [-1.0,1.0]:
            y,feasible=_project_unbounded(x,p,s,k,floor,lam,W,W2)
            assert feasible
            assert np.min(y)>=0 and np.max(y)<=255
            assert _affine_slack(y,p,s,k,floor,W)>=-1e-10
            delta=y-x
            obj=0.5*np.sum(delta**2)+lam*64/2*np.sum(delta.mean(axis=0)**2)
            deficit=max(0,floor+k*sum(_means(x,p,W))-s*(_means(x,p,W)[0]-_means(x,p,W)[1]))
            denom=(4/64)*W2*(1+k*k/(1+lam))
            np.testing.assert_allclose(obj,deficit**2/(2*denom),atol=1e-9)
            max_cost=max(max_cost,obj)
        costs.append(max_cost)
    assert selected==int(np.argmin(costs))


def test_box_projection_and_integer_constraints():
    p=_codebook(8)[0]
    x=np.zeros((64,3),dtype=np.float64)
    x[::4]=255.0
    mu=.015
    k=mu/math.sqrt(2-mu*mu)
    for sign in [-1.0,1.0]:
        y,active=_continuous_project(x,p,sign,k,.25,2.0,W,W2)
        assert 0 <= y.min() and y.max()<=255
        assert _affine_slack(y,p,sign,k,.25,W)>-1e-6
        z,ok=_round_and_repair(y,x,p,sign,k,.25,2.0,W)
        assert ok
        assert z.dtype==np.uint8
        assert _affine_slack(z,p,sign,k,.25,W)>=-1e-9
        assert sign*_normalized_residual(z,p,W)>=mu-1e-10


def test_clean_recovery_and_message_independent_map():
    host=load_host_rgb(ROOT/'data/host/Girl.bmp')
    watermark=load_watermark_binary(ROOT/'data/watermark/wm.png')
    marked,key,meta=embed_msdp(host,watermark,seed=2026)
    assert np.array_equal(extract_msdp(marked,key),watermark)
    assert meta['minimum_attained_normalized_residual']>=0.015-1e-10
    assert meta['minimum_attained_signed_determinant']>=.25-1e-10
    assert len(key.pattern_codes)==1536
    assert key_to_jsonable(key)['method']=='msdp_v1'
    assert np.array_equal(extract_msdp(marked,key_from_jsonable(key_to_jsonable(key))),watermark)
    _,flipped_key,_=embed_msdp(host,255-watermark,seed=2026)
    assert key.pattern_codes==flipped_key.pattern_codes
    cover_accuracy=np.mean(extract_msdp(host,key)==watermark)
    assert .45<=cover_accuracy<=.55,cover_accuracy
    assert verify_msdp(marked,key,watermark)['accepted']
    assert not verify_msdp(host,key,watermark)['accepted']


def test_all_literature_baselines_integrated():
    assert len(ALL_BASELINE_IDS)==16
    host=load_host_rgb(ROOT/'data/host/Girl.bmp')
    watermark=load_watermark_binary(ROOT/'data/watermark/wm.png')
    watermarked,key=embed_baseline('dct_dm_qim_chen2001_blind',host,watermark,seed=2026,repeat=1)
    assert np.array_equal(extract_baseline(watermarked,key),watermark)


def test_host_data_excludes_requested_image():
    hosts=sorted((ROOT/'data/host').glob('*.bmp'))
    assert len(hosts)==12
    assert not any('lenna' in f.stem.lower() for f in hosts)
