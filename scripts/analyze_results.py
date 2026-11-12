#!/usr/bin/env python3
"""Failure-aware attack-group summaries and paired host-level bootstrap CIs."""
import argparse
import csv
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.research import csv_write,json_write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='per_attack.csv from benchmark/ablation/sweep')
    p.add_argument('--method-column',default='auto',help='method_id or variant, auto detects')
    p.add_argument('--reference',default=None)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=451)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    rows=list(csv.DictReader(open(a.input,encoding='utf-8')))
    if not rows:p.error('empty input')
    method=a.method_column if a.method_column!='auto' else ('method_id' if 'method_id' in rows[0] else 'variant')
    if method not in rows[0]:p.error('method field missing')
    groups=defaultdict(list)
    for r in rows:
        category=r.get('attack_category') or r.get('attack_group') or 'unknown'
        groups[(r.get(method,''),category)].append(r)
    summaries=[]
    for (m,category),items in sorted(groups.items()):
        usable=[r for r in items if r.get('status')=='ok' and r.get('attacked_nc') not in ('',None)]
        summaries.append({method:m,'category':category,'total':len(items),
                          'ok':len(usable),'failures':len(items)-len(usable),
                          'mean_nc_success_only':float(np.mean([float(r['attacked_nc']) for r in usable])) if usable else None,
                          'mean_ber_success_only':float(np.mean([float(r['attacked_ber']) for r in usable if r.get('attacked_ber') not in ('',None)])) if usable else None})
    output=Path(a.output)
    csv_write(summaries,output/'by_attack_category.csv')
    if a.reference:
        # Same host/watermark/seed/attack and no duplicates: zero unsafe matching.
        def make_key(r):return (r.get('host_id',''),r.get('watermark_id',''),r.get('seed',''),r.get('attack_id',''))
        by_method=defaultdict(dict)
        for r in rows:
            k=make_key(r);m=r.get(method,'')
            if k in by_method[m]:
                raise ValueError(f'Non-unique trial {k} for {m}; analyze one config per run')
            by_method[m][k]=r
        ref=by_method.get(a.reference)
        if not ref:p.error('Unknown reference')
        rng=np.random.default_rng(a.seed);comp=[]
        for other,mapping in by_method.items():
            if other==a.reference:continue
            paired=[]
            for k in ref.keys()&mapping.keys():
                r=ref[k];o=mapping[k]
                if (r.get('status')==o.get('status')=='ok'
                    and r.get('attacked_nc') not in ('',None)
                    and o.get('attacked_nc') not in ('',None)
                    and k[-1]!='none__clean'):
                    paired.append((k,float(o['attacked_nc'])-float(r['attacked_nc'])))
            clusters=defaultdict(list)
            for k,delta in paired:clusters[(k[0],k[1],k[2])].append(delta)
            # Bootstrap on independent host/watermark/seed clusters, NOT attack rows.
            cluster_means=np.array([np.mean(vals) for vals in clusters.values()])
            if len(cluster_means):
                boots=[np.mean(rng.choice(cluster_means,size=len(cluster_means),replace=True))
                       for _ in range(a.bootstrap)] if a.bootstrap else []
                low,high=np.quantile(boots,[.025,.975]) if boots else (None,None)
                diff=float(np.mean(cluster_means))
            else:diff=low=high=None
            comp.append({'reference':a.reference,'method':other,'paired_attack_rows':len(paired),
                         'matched_clusters':len(clusters),'mean_paired_delta_nc':diff,
                         'bootstrap_ci_low':low,'bootstrap_ci_high':high,
                         'reference_rows_missing_or_failed':len(ref)-len(paired)})
        csv_write(comp,output/'paired_bootstrap.csv')
    json_write({'input':a.input,'input_rows':len(rows),'method_field':method,
                'warning':'Means exclude failed rows and are NOT directly comparable if coverage differs. '
                          'Bootstrap CIs resample host-watermark-seed clusters, assuming those are independent.'},
                output/'analysis_protocol.json')

if __name__=='__main__':main()
