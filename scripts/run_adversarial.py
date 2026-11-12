#!/usr/bin/env python3
"""Key-exposed targeted affine-boundary attack, collusion, and overwrite tests.

Do not misrepresent access to the private detector map as a keyless attack.
"""
import argparse
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.msdp import (W,W2,N,_blockize,_unblockize,_codebook,_unpack,
    _means,_continuous_project,_round_and_repair,embed_msdp,extract_msdp,MSDPConfig)
from qr64_certified.common.io import load_host_rgb,load_watermark_binary
from qr64_certified.common.metrics import psnr,ssim,watermark_metrics
from qr64_certified.research import hosts,watermarks,parse_seeds,csv_write,json_write

def affine_targeted(image,key, fraction:float, selection:str, *, seed:int=1234):
    blocks,shape=_blockize(image)
    count=len(blocks)
    indices=_unpack(key.pattern_codes,key.pattern_count,count)
    patterns=_codebook(key.pattern_count)
    signed=np.einsum('nmc,nm,c->n',blocks.astype(np.float64),patterns[indices],W)*(2./N)
    radius=np.abs(signed)/(2.*math.sqrt(1./N)*math.sqrt(W2))
    qty=max(1,int(round(fraction*count)))
    if selection=='cheapest':chosen=np.argsort(radius)[:qty]
    else:chosen=np.random.default_rng(seed).permutation(count)[:qty]
    out=blocks.copy()
    for i in chosen:
        p=patterns[indices[i]]
        sign=-1. if signed[i]>=0 else 1.
        x=blocks[i].astype(np.float64)
        y,_=_continuous_project(x,p,sign,0.,.05,0.,W,W2)
        changed,valid=_round_and_repair(y,x,p,sign,0.,.05,0.,W)
        if valid:out[i]=changed
    return _unblockize(out,shape),radius,chosen

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hosts',default='Girl')
    p.add_argument('--watermarks',default='wm')
    p.add_argument('--seeds',default='2026')
    p.add_argument('--fractions',default='.01,.05,.10,.25')
    p.add_argument('--copies',default='2,4,8')
    p.add_argument('--selection',choices=('cheapest','random','both'),default='both')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    rows=[];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for h in hosts(a.hosts):
        host=load_host_rgb(h)
        for w in watermarks(a.watermarks):
            wm=load_watermark_binary(w)
            for seed in parse_seeds(a.seeds):
                marked,key,_=embed_msdp(host,wm,seed=seed)
                for frac in (float(x) for x in a.fractions.split(',')):
                    for selection in ('cheapest','random') if a.selection=='both' else (a.selection,):
                        attacked,radii,chosen=affine_targeted(marked,key,frac,selection,seed=seed+13)
                        m=watermark_metrics(wm,extract_msdp(attacked,key))
                        rows.append({'kind':'targeted_key_exposed','host':h.stem,'watermark':w.stem,'seed':seed,
                                     'fraction':frac,'selection':selection,'actual_targeted_blocks':len(chosen),
                                     'median_theoretical_L2_radius':float(np.median(radii[chosen])),
                                     'psnr_db':psnr(host,attacked),'ssim':ssim(host,attacked),
                                     'ber':m['ber'],'nc':m['nc'],'status':'ok'})
                        csv_write(rows,out/'adversarial_results.csv')
                rng=np.random.default_rng(np.random.SeedSequence([seed,609]))
                for ncopies in (int(x) for x in a.copies.split(',')):
                    if ncopies<2:continue
                    marked_copies=[marked];keys=[key]
                    for j in range(ncopies-1):
                        random_wm=(rng.integers(0,2,size=wm.shape,dtype=np.uint8)*255)
                        mi,ki,_=embed_msdp(host,random_wm,seed=seed+100+j)
                        marked_copies.append(mi);keys.append(ki)
                    stack=np.stack(marked_copies).astype(np.float32)
                    for combine,mixed in [('mean',np.rint(np.mean(stack,axis=0)).astype(np.uint8)),
                                          ('median',np.rint(np.median(stack,axis=0)).astype(np.uint8))]:
                        m=watermark_metrics(wm,extract_msdp(mixed,key))
                        rows.append({'kind':'collusion','host':h.stem,'watermark':w.stem,'seed':seed,
                                     'copies':ncopies,'combine':combine,'psnr_db':psnr(host,mixed),
                                     'ssim':ssim(host,mixed),'ber':m['ber'],'nc':m['nc'],'status':'ok'})
                    overwritten=marked_copies[-1]
                    m=watermark_metrics(wm,extract_msdp(overwritten,key))
                    rows.append({'kind':'replacement_overwrite','host':h.stem,'watermark':w.stem,'seed':seed,
                                 'copies':ncopies,'psnr_db':psnr(host,overwritten),
                                 'ssim':ssim(host,overwritten),'ber':m['ber'],'nc':m['nc'],
                                 'status':'ok'})
                    # Sequential second embedding into an ALREADY marked cover:
                    new_wm=(rng.integers(0,2,size=wm.shape,dtype=np.uint8)*255)
                    sequential,new_key,_=embed_msdp(marked,new_wm,seed=seed+900+ncopies)
                    old_sc=watermark_metrics(wm,extract_msdp(sequential,key))
                    new_sc=watermark_metrics(new_wm,extract_msdp(sequential,new_key))
                    rows.append({'kind':'sequential_overwrite','host':h.stem,'watermark':w.stem,
                                 'seed':seed,'copies':ncopies,
                                 'psnr_db':psnr(host,sequential),'ssim':ssim(host,sequential),
                                 'ber':old_sc['ber'],'nc':old_sc['nc'],
                                 'new_mark_nc':new_sc['nc'],'status':'ok'})
                    csv_write(rows,out/'adversarial_results.csv')
    json_write({'description':'Key exposed attack; collusion copies use independently embedded marks. '
                               'Replacement is a separately watermarked copy, not sequential overwrite.',
                'trials':len(rows)},out/'protocol.json')
    print('Wrote',len(rows),'trials')

if __name__=='__main__':main()
