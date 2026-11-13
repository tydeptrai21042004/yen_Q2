#!/usr/bin/env python3
"""Study effect of watermark content, bit balance, and independent seeds."""
import argparse
from pathlib import Path
import hashlib
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.msdp import MSDPConfig
from qr64_certified.common.io import load_host_rgb,load_watermark_binary
from qr64_certified.research import hosts,measure_trial,write_result_bundle,TrialJournal,protocol_hash

def generate(mode,shape,rng,p):
    r,c=np.indices(shape)
    if mode=='bernoulli':return np.where(rng.random(shape)<p,255,0).astype(np.uint8)
    if mode=='checkerboard':return (((r+c)%2)*255).astype(np.uint8)
    if mode=='stripes':return (((c//4)%2)*255).astype(np.uint8)
    if mode=='constant_zero':return np.zeros(shape,dtype=np.uint8)
    if mode=='constant_one':return np.full(shape,255,dtype=np.uint8)
    raise ValueError(mode)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hosts',default='Girl')
    parser.add_argument('--seeds',default='2026')
    parser.add_argument('--probabilities',default='.1,.25,.5,.75,.9')
    parser.add_argument('--modes',default='bernoulli,checkerboard,stripes,constant_zero,constant_one,logo')
    parser.add_argument('--suite',default='basic20x3')
    parser.add_argument('--skip-attacked-quality',action='store_true')
    parser.add_argument('--output',required=True)
    a=parser.parse_args()
    cfg=MSDPConfig()
    host_files=hosts(a.hosts);seed_values=[int(v) for v in a.seeds.split(',')]
    modes=a.modes.split(',');probabilities=[float(x) for x in a.probabilities.split(',')]
    spec={'experiment':'watermark_content','hosts':[h.stem for h in host_files],
          'seeds':seed_values,'modes':modes,'probabilities':probabilities,
          'suite':a.suite,'protocol_hash':protocol_hash(a.suite),
          'attacked_quality':not a.skip_attacked_quality}
    journal=TrialJournal(a.output,spec=spec)
    for h in host_files:
        image=load_host_rgb(h);shape=(image.shape[0]//8,image.shape[1]//8)
        for seed in seed_values:
            for mode in modes:
                values=probabilities if mode=='bernoulli' else [.5]
                for prob in values:
                    name=f'{mode}_p{prob:g}'
                    trial_id=f'{h.stem}|{seed}|{name}'
                    if journal.has(trial_id):
                        continue
                    derived=int.from_bytes(hashlib.sha256(f'{mode}:{prob:g}'.encode()).digest()[:4],'big')
                    rng=np.random.default_rng(np.random.SeedSequence([seed,derived]))
                    wm=load_watermark_binary(ROOT/'data/watermark/wm.png') if mode=='logo' else generate(mode,shape,rng,prob)
                    result=measure_trial(image,wm,config=cfg,seed=seed,suite=a.suite,
                                          label=name,host_id=h.stem,watermark_id=name,
                                          attacked_quality=not a.skip_attacked_quality)
                    balance=float(np.mean(wm>127))
                    for row in result:row['actual_one_fraction']=balance
                    journal.append(trial_id,result)
    write_result_bundle(journal.data,a.output,keys=('variant',),spec=spec)
    print('Recorded',len(journal.data),'rows')
if __name__=='__main__':main()
