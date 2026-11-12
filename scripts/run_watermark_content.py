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
from qr64_certified.research import hosts,measure_trial,write_result_bundle

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
    parser.add_argument('--seeds',default='2026,2027,2028')
    parser.add_argument('--probabilities',default='.1,.25,.5,.75,.9')
    parser.add_argument('--modes',default='bernoulli,checkerboard,stripes,constant_zero,constant_one,logo')
    parser.add_argument('--suite',default='sanity')
    parser.add_argument('--output',required=True)
    a=parser.parse_args();rows=[]
    cfg=MSDPConfig()
    for h in hosts(a.hosts):
        image=load_host_rgb(h);shape=(image.shape[0]//8,image.shape[1]//8)
        for seed in [int(v) for v in a.seeds.split(',')]:
            for mode in a.modes.split(','):
                values=[float(x) for x in a.probabilities.split(',')] if mode=='bernoulli' else [.5]
                for p in values:
                    derived=int.from_bytes(hashlib.sha256(f'{mode}:{p:g}'.encode()).digest()[:4],'big')
                    rng=np.random.default_rng(np.random.SeedSequence([seed,derived]))
                    wm=load_watermark_binary(ROOT/'data/watermark/wm.png') if mode=='logo' else generate(mode,shape,rng,p)
                    name=f'{mode}_p{p:g}'
                    result=measure_trial(image,wm,config=cfg,seed=seed,suite=a.suite,
                                          label=name,host_id=h.stem,watermark_id=name)
                    balance=float(np.mean(wm>127))
                    for row in result:row['actual_one_fraction']=balance
                    rows+=result
                    write_result_bundle(rows,a.output,keys=('variant',),spec={
                        'design':'Independent random watermarks or controlled bit patterns',
                        'seed_definition':'seed controls embed mask and separate watermark RNG'})
    print('Recorded',len(rows),'rows')
if __name__=='__main__':main()
