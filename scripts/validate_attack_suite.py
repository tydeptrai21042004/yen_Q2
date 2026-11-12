#!/usr/bin/env python3
"""Audit the exact unified attack protocol, codec support and determinism."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.attacks import apply_attack,get_attack_suite,list_attack_suites
from qr64_certified.research import csv_write,json_write,protocol_hash

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suites',default='common20',help='Comma-separated suite names or all')
    p.add_argument('--probe',action='store_true',help='Actually execute each attack twice on fixed image')
    p.add_argument('--fail-on-error',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    names=list(list_attack_suites()) if a.suites=='all' else a.suites.split(',')
    rng=np.random.default_rng(889)
    sample=rng.integers(0,256,size=(512,512,3),dtype=np.uint8)
    rows=[]
    for name in names:
        attacks=get_attack_suite(name)
        if len({v.attack_id for v in attacks})!=len(attacks):
            raise ValueError(f'duplicate attack IDs in {name}')
        for attack in attacks:
            row={'suite':name,'suite_sha256':protocol_hash(name),**asdict(attack),
                 'probe_status':'not_requested'}
            if a.probe:
                try:
                    one=apply_attack(sample,attack)
                    two=apply_attack(sample,attack)
                    if one.shape!=sample.shape or one.dtype!=np.uint8:
                        raise AssertionError('invalid shape or type')
                    if not np.array_equal(one,two):
                        raise AssertionError('attack not deterministic on repeated input')
                    row.update({'probe_status':'ok','mean_abs_change':float(np.mean(np.abs(one.astype(float)-sample)))})
                except Exception as ex:
                    row.update({'probe_status':'error','error':repr(ex)})
            rows.append(row)
            print(name,attack.attack_id,row['probe_status'],flush=True)
    out=Path(a.output)
    csv_write(rows,out/'attack_manifest.csv')
    json_write({'suites':{name:{'count':len(get_attack_suite(name)),
                              'sha256':protocol_hash(name)} for name in names},
                 'failure_count':sum(r['probe_status']=='error' for r in rows),
                 'rules':'Same named config per method. JPEG2000 requires actual codec; no silent fallback. '
                         'Simulation attacks are not physical camera/print-scan measurements.'},
               out/'attack_protocol.json')
    if a.fail_on_error and any(r['probe_status']=='error' for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
