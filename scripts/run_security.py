#!/usr/bin/env python3
"""MSDP detection study: wrong-key, wrong-map, unmarked, unrelated-mark and ROC.

Statistical diagnostics only, NOT cryptographic authentication proofs.
Threshold is declared BEFORE scoring; ROC is descriptive, not threshold tuning.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.msdp import (MSDPConfig,embed_msdp,extract_msdp,_pack,verify_msdp)
from qr64_certified.common.io import load_host_rgb,load_watermark_binary
from qr64_certified.research import hosts,watermarks,parse_seeds,csv_write,json_write

def corr(image,key,reference):
    got=extract_msdp(image,key)
    return 2.*float(np.mean((got>127)==(reference>127)))-1.

def auc(scores):
    p=np.array([r['correlation'] for r in scores if r['label']==1]);
    n=np.array([r['correlation'] for r in scores if r['label']==0]);
    if not len(p) or not len(n):return None
    return float(np.mean((p[:,None]>n[None,:])+0.5*(p[:,None]==n[None,:])))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hosts',default='Girl,pepper')
    p.add_argument('--watermarks',default='wm')
    p.add_argument('--seeds',default='2026')
    p.add_argument('--negative-trials',type=int,default=100)
    p.add_argument('--threshold',type=float,default=.15)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if not 0<a.threshold<=1 or a.negative_trials<1:p.error('invalid threshold or negative count')
    images=[(h.stem,load_host_rgb(h)) for h in hosts(a.hosts)]
    rows=[]
    for w in watermarks(a.watermarks):
        wm=load_watermark_binary(w)
        for image_id,host in images:
            for seed in parse_seeds(a.seeds):
                print(f'security: {image_id} {w.stem} seed={seed}',flush=True)
                marked,key,_=embed_msdp(host,wm,seed=seed)
                def push(kind,y,k,reference,trial,positive=0):
                    score=corr(y,k,reference)
                    rows.append({'host_id':image_id,'watermark_id':w.stem,'seed':seed,
                                 'experiment':kind,'trial':trial,'label':positive,
                                 'correlation':score,'bit_accuracy':(score+1)/2,
                                 'accepted':bool(score>=a.threshold),'threshold':a.threshold})
                push('correct_key_marked',marked,key,wm,0,1)
                push('correct_key_unmarked',host,key,wm,0)
                for foreign_id,foreign_host in images:
                    if foreign_id!=image_id:
                        push('other_host_same_key',foreign_host,key,wm,foreign_id)
                rng=np.random.default_rng(np.random.SeedSequence([seed,811]))
                n_blocks=wm.size
                for i in range(a.negative_trials):
                    false_seed=int(rng.integers(0,2**31))
                    if false_seed==seed:false_seed^=0xA5A5
                    push('wrong_mask_seed',marked,replace(key,seed=false_seed),wm,i)
                    indices=rng.integers(0,key.pattern_count,size=n_blocks,dtype=np.int64)
                    push('wrong_pattern_map',marked,replace(key,pattern_codes=_pack(indices,key.pattern_count)),wm,i)
                    random_wm=(rng.integers(0,2,size=wm.shape,dtype=np.uint8)*255)
                    push('unrelated_reference_mark',marked,key,random_wm,i)
    csv_write(rows,Path(a.output)/'detection_trials.csv')
    breakdown=[]
    for kind in sorted(set(r['experiment'] for r in rows)):
        part=[r for r in rows if r['experiment']==kind]
        breakdown.append({'experiment':kind,'n_trials':len(part),
                          'accept_count':sum(r['accepted'] for r in part),
                          'accept_rate':float(np.mean([r['accepted'] for r in part])),
                          'mean_correlation':float(np.mean([r['correlation'] for r in part]))})
    csv_write(breakdown,Path(a.output)/'security_by_scenario.csv')
    thresholds=np.linspace(-1,1,201)
    roc=[]
    p_rows=[r for r in rows if r['label']==1]
    n_rows=[r for r in rows if r['label']==0]
    for t in thresholds:
        roc.append({'threshold':float(t),'tpr':float(np.mean([r['correlation']>=t for r in p_rows])),
                    'fpr':float(np.mean([r['correlation']>=t for r in n_rows]))})
    csv_write(roc,Path(a.output)/'roc.csv')
    pos=sum(r['accepted'] for r in p_rows);false=sum(r['accepted'] for r in n_rows)
    # 95% one-sided Clopper-Pearson upper bound (for zero observed false positives,
    # 1 - 0.05**(1/n)); trials are dependent; NOT a population assurance.
    upper=float(beta.ppf(.95,false+1,len(n_rows)-false)) if false<len(n_rows) else 1.
    summary={'threshold':a.threshold,'n_positive':len(p_rows),'true_positive':pos,
             'n_negative':len(n_rows),'false_positive':false,
             'observed_tpr':pos/len(p_rows),'observed_fpr':false/len(n_rows),
             'nominal_95pct_one_sided_fpr_upper_if_iid':upper,
             'auc_descriptive':auc(rows),
             'warning':'Negative trials share images/keys; independence assumption for binomial bound is not justified. '
                       'Do not make cryptographic security or population confidence claims.'}
    json_write(summary,Path(a.output)/'security_summary.json')
    print(summary)

if __name__=='__main__':main()
