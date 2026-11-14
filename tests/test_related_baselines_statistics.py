"""Regression checks for newly added reference controls and paired inference."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qr64_certified.baselines import (
    ALL_BASELINE_IDS, CONTROL_BASELINE_IDS, RELATED_ADDITIONAL_BASELINE_IDS,
    get_baseline_spec,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('paired', ROOT/'scripts'/'analyze_paired_statistics.py')
paired = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paired)


def test_published_paper_registry_only():
    assert len(ALL_BASELINE_IDS) == 16
    assert CONTROL_BASELINE_IDS == ()
    assert set(RELATED_ADDITIONAL_BASELINE_IDS) == {
        'dwt_qr_fa_guo2017_keyassisted',
        'dct_ca_qim_mao2024_semiblind',
        'dwt_wht_svd_kumar2024_semiblind',
    }
    for method in RELATED_ADDITIONAL_BASELINE_IDS:
        spec = get_baseline_spec(method)
        assert spec.citation and spec.citation != 'new_control'
        assert spec.fidelity_tier == ('paper_core_common_image_adapter' if 'mao2024' in method else 'paper_guided_implementation')
        assert spec.common_4096_payload
        assert spec.blindness_tier in ('key_assisted_blind', 'semi_blind')


def make_rows(offset: float):
    out={}
    for host in [f'h{i}' for i in range(8)]:
        for mark in ('w1','w2'):
            for attack in ('none__clean','jpeg_mild','median_strong'):
                key=(host,mark,'2026',attack)
                row={
                    'host_id':host, 'watermark_id':mark,'seed':'2026',
                    'attack_id':attack,'attack_category':'compression' if attack.startswith('jpeg') else 'filtering',
                    'attack_protocol_sha256':'same', 'status':'ok','payload_bits':'4096',
                    'clean_psnr_db':str(44.0+offset), 'clean_ssim':'.98',
                    'clean_ber':'0.0', 'clean_nc':'1.0',
                    'attacked_ber':str(.15-offset), 'attacked_nc':str(.85+offset)
                }
                out[key]=row
    return out


def test_host_level_paired_exact_and_coverage():
    reference=make_rows(0.02)
    control=make_rows(0.0)
    summary, overall, breakdown=paired.analyze({'msdp':reference,'control':control},
                                                  'msdp',bootstrap=500,seed=10)
    assert len(summary)==2 and len(overall)==3
    nc=next(row for row in overall if row['metric']=='attacked_nc')
    assert nc['n_hosts']==8 and nc['n_paired_rows']==32
    assert nc['mean_reference_advantage']==pytest.approx(.02)
    assert nc['p_two_sided']==pytest.approx(2/256)
    assert nc['ci95_low']==pytest.approx(.02)
    assert nc['p_holm']==nc['p_two_sided']
    assert breakdown
    broken=control.copy();broken.pop(next(iter(broken)))
    with pytest.raises(ValueError,match='missing'):
        paired.analyze({'msdp':reference,'control':broken},'msdp',bootstrap=200,seed=1)
    bad=make_rows(0.0)
    bad[next(iter(bad))]['status']='error'
    with pytest.raises(ValueError,match='failed'):
        paired.analyze({'msdp':reference,'control':bad},'msdp',bootstrap=200,seed=1)


def test_paper_provenance_audit():
    spec=importlib.util.spec_from_file_location('audit',ROOT/'scripts'/'audit_paper_baselines.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows=module.audit()
    assert len(rows)==8
    assert all(row['doi'].startswith('10.') and row['exact_paper_reproduction'] is False for row in rows)
    assert sum(row['blindness_tier']=='semi_blind' for row in rows)==2
    assert not any('control' in row['method_id'] or 'dummy' in row['method_id'] for row in rows)
