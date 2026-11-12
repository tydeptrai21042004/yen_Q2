#!/usr/bin/env python3
"""Controlled method ablation: FULL vs minimax alternatives and ablated terms."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dataclasses import replace
from qr64_certified.msdp import MSDPConfig
from qr64_certified.common.io import load_host_rgb,load_watermark_binary
from qr64_certified.research import (add_common_args,hosts,watermarks,parse_seeds,
                                     measure_trial,write_result_bundle)

VARIANTS={
 'full':{},
 'fixed_pattern':{'pattern_selector':'fixed'},
 'random_pattern':{'pattern_selector':'random'},
 'anti_minimax':{'pattern_selector':'max_abs'},
 'no_mean_penalty':{'mean_penalty':0.0},
 'no_integer_repair':{'integer_repair':False},
}

def main():
    p=add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--variants',default=','.join(VARIANTS),help='Comma-separated variant names')
    p.add_argument('--normalized-margin',type=float,default=.015)
    p.add_argument('--determinant-floor',type=float,default=.25)
    p.add_argument('--mean-penalty',type=float,default=2.)
    p.add_argument('--pattern-count',type=int,default=8)
    a=p.parse_args()
    names=a.variants.split(',')
    if any(v not in VARIANTS for v in names):p.error(f'Valid variants: {list(VARIANTS)}')
    base=MSDPConfig(normalized_margin=a.normalized_margin,determinant_floor=a.determinant_floor,
                    mean_penalty=a.mean_penalty,pattern_count=a.pattern_count)
    rows=[]
    for h in hosts(a.hosts):
        host=load_host_rgb(h)
        for w in watermarks(a.watermarks):
            wm=load_watermark_binary(w)
            for seed in parse_seeds(a.seeds):
                for name in names:
                    config=replace(base,**VARIANTS[name])
                    print(f'{name} {h.stem} {w.stem} seed={seed}',flush=True)
                    rows+=measure_trial(host,wm,config=config,seed=seed,suite=a.suite,label=name,
                                        host_id=h.stem,watermark_id=w.stem,fail_fast=a.fail_fast)
                    write_result_bundle(rows,a.output,spec={'variants':names,'suite':a.suite,
                                                               'config':base.__dict__})
    if a.fail_fast and any(r['status']!='ok' for r in rows):raise SystemExit(1)

if __name__=='__main__': main()
