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
                                     measure_trial,write_result_bundle,TrialJournal,protocol_hash)

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
    cfgs={name:replace(base,**VARIANTS[name]) for name in names}
    seed_values=parse_seeds(a.seeds)
    file_hosts=hosts(a.hosts);file_wms=watermarks(a.watermarks)
    spec={'experiment':'ablation','hosts':[h.stem for h in file_hosts],
          'watermarks':[w.stem for w in file_wms],'seeds':seed_values,
          'suite':a.suite,'protocol_hash':protocol_hash(a.suite),'configs':{k:v.__dict__ for k,v in cfgs.items()},
          'attacked_quality':not a.skip_attacked_quality}
    journal=TrialJournal(a.output,spec=spec)
    for h in file_hosts:
        host=load_host_rgb(h)
        for w in file_wms:
            wm=load_watermark_binary(w)
            for seed in seed_values:
                for name in names:
                    trial_id=f'{name}|{h.stem}|{w.stem}|{seed}'
                    if journal.has(trial_id):
                        print('Resuming: skip', trial_id,flush=True)
                        continue
                    print(trial_id,flush=True)
                    trial=measure_trial(host,wm,config=cfgs[name],seed=seed,suite=a.suite,label=name,
                                        host_id=h.stem,watermark_id=w.stem,fail_fast=a.fail_fast,
                                        attacked_quality=not a.skip_attacked_quality)
                    journal.append(trial_id,trial)
    write_result_bundle(journal.data,a.output,spec=spec)
    if a.fail_fast and any(r['status']!='ok' for r in journal.data):raise SystemExit(1)

if __name__=='__main__': main()
