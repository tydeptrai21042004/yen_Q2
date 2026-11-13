#!/usr/bin/env python3
"""OFAT or factorial MSDP hyperparameter sweep with frozen attack protocol."""
import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.msdp import MSDPConfig
from qr64_certified.common.io import load_host_rgb,load_watermark_binary
from qr64_certified.research import (add_common_args,hosts,watermarks,parse_seeds,
                                     measure_trial,write_result_bundle,csv_write,TrialJournal,protocol_hash)

def floats(x):return [float(t) for t in x.split(',')]
def ints(x):return [int(t) for t in x.split(',')]

def configs(a):
    base=MSDPConfig()
    groups={'normalized_margin':floats(a.margins),
            'determinant_floor':floats(a.floors),
            'mean_penalty':floats(a.penalties),
            'pattern_count':ints(a.pattern_counts)}
    output=[]
    if a.mode=='ofat':
        output.append(base)
        for field,values in groups.items():
            output.extend(replace(base,**{field:v}) for v in values)
    else:
        output=[MSDPConfig(*v) for v in product(*groups.values())]
    configs=list(dict.fromkeys(output))
    for cfg in configs:cfg.validate()
    if len(configs)>a.max_configs:
        raise ValueError(f'{len(configs)} configurations exceeds --max-configs {a.max_configs}; narrow ranges')
    return configs

def main():
    p=add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--mode',choices=('ofat','grid'),default='ofat')
    p.add_argument('--margins',default='.005,.01,.015,.02,.03')
    p.add_argument('--floors',default='.05,.1,.25,.5,1.0')
    p.add_argument('--penalties',default='0,.5,1,2,5')
    p.add_argument('--pattern-counts',default='2,4,8,16,32')
    p.add_argument('--max-configs',type=int,default=150)
    a=p.parse_args(); candidates=configs(a)
    pairs=[(h,w) for h in hosts(a.hosts) for w in watermarks(a.watermarks)]
    seed_values=parse_seeds(a.seeds)
    spec={'experiment':'parameter_sweep','mode':a.mode,'suite':a.suite,
          'protocol_hash':protocol_hash(a.suite),'pairs':[(h.stem,w.stem) for h,w in pairs],
          'seeds':seed_values,'attacked_quality':not a.skip_attacked_quality,
          'configs':{f'config_{j:03d}':c.__dict__ for j,c in enumerate(candidates)}}
    journal=TrialJournal(a.output,spec=spec)
    for idx,cfg in enumerate(candidates):
        name=f'config_{idx:03d}'
        print(name,cfg,flush=True)
        for h,w in pairs:
            for seed in seed_values:
                trial_id=f'{name}|{h.stem}|{w.stem}|{seed}'
                if journal.has(trial_id):
                    print('Resuming: skip', trial_id, flush=True)
                    continue
                host=load_host_rgb(h);wm=load_watermark_binary(w)
                trial=measure_trial(host,wm,config=cfg,seed=seed,suite=a.suite,label=name,
                      host_id=h.stem,watermark_id=w.stem,fail_fast=a.fail_fast,
                      attacked_quality=not a.skip_attacked_quality)
                journal.append(trial_id,trial)
    summary=write_result_bundle(journal.data,a.output,spec=spec)
    # Descriptive Pareto frontier only: clean PSNR and attacked NC maximize.
    available=[r for r in summary if r['failed_rows']==0 and
               r['mean_clean_psnr_db'] is not None and r['mean_attack_nc'] is not None]
    frontier=[]
    for row in available:
        dominated=any(other is not row and
                      other['mean_clean_psnr_db']>=row['mean_clean_psnr_db'] and
                      other['mean_attack_nc']>=row['mean_attack_nc'] and
                      (other['mean_clean_psnr_db']>row['mean_clean_psnr_db'] or
                       other['mean_attack_nc']>row['mean_attack_nc']) for other in available)
        if not dominated:frontier.append(row)
    csv_write(frontier,Path(a.output)/'pareto_complete_cases_only.csv')

if __name__=='__main__':main()
