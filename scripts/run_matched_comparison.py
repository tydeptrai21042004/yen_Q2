#!/usr/bin/env python3
"""Disjoint-host PSNR calibration followed by baseline-vs-MSDP test comparisons.

PSNR matching is approximate and never adjusts the native baseline method.
Only compare same-payload, same-key-assumption methods in primary tables.
"""
import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.msdp import MSDPConfig,embed_msdp,extract_msdp
from qr64_certified.baselines import embed_baseline,extract_baseline,get_baseline_spec
from qr64_certified.attacks import apply_attack,get_attack_suite
from qr64_certified.common.io import load_host_rgb,load_watermark_binary
from qr64_certified.common.metrics import psnr,ssim,watermark_metrics
from qr64_certified.research import hosts,watermarks,csv_write,json_write

def embed(method,host,wm,seed,margin,opts):
    if method=='msdp':return embed_msdp(host,wm,seed=seed,config=MSDPConfig(normalized_margin=margin))[:2]
    return embed_baseline(method,host,wm,seed=seed,repeat=opts.get('repeat',1),step=opts.get('step'),
                          method_params=opts.get('method_params',{}))

def extract(method,marked,key,host):
    if method=='msdp':return extract_msdp(marked,key)
    return extract_baseline(marked,key,original_host=host if get_baseline_spec(method).requires_original_host else None)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',default='dct_dm_qim_chen2001_blind')
    p.add_argument('--calibration-hosts',default='Girl')
    p.add_argument('--test-hosts',default='pepper')
    p.add_argument('--watermarks',default='wm')
    p.add_argument('--margins',default='.005,.01,.015,.02,.03,.05')
    p.add_argument('--suite',default='basic20x3')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--psnr-tolerance',type=float,default=1.5)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    cal=hosts(a.calibration_hosts);test=hosts(a.test_hosts)
    if {p.stem for p in cal}&{p.stem for p in test}:
        p.error('Calibration and test hosts MUST NOT overlap')
    methods=('msdp',a.baseline)
    get_baseline_spec(a.baseline)
    baselines=json.loads((ROOT/'configs/baseline_parameters.json').read_text())
    opts=baselines.get(a.baseline,{})
    wms=[load_watermark_binary(w) for w in watermarks(a.watermarks)]
    calib=[]
    for h in cal:
        host=load_host_rgb(h)
        for wm in wms:
            marked,_=embed(a.baseline,host,wm,a.seed,.015,opts)
            calib.append({'method':a.baseline,'margin':None,'psnr':psnr(host,marked)})
            for margin in [float(v) for v in a.margins.split(',')]:
                marked,_=embed('msdp',host,wm,a.seed,margin,{})
                calib.append({'method':'msdp','margin':margin,'psnr':psnr(host,marked)})
    base_psnr=float(np.mean([r['psnr'] for r in calib if r['method']!= 'msdp']))
    candidates=sorted(set(r['margin'] for r in calib if r['method']=='msdp'))
    selected=min(candidates,key=lambda m:abs(np.mean([r['psnr'] for r in calib if r['margin']==m])-base_psnr))
    achieved=float(np.mean([r['psnr'] for r in calib if r['margin']==selected]))
    out=Path(a.output);csv_write(calib,out/'calibration_psnr.csv')
    manifest={'baseline':a.baseline,'reference_calibration_psnr':base_psnr,
              'selected_margin':selected,'msdp_calibration_psnr':achieved,
              'absolute_calibration_gap':abs(base_psnr-achieved),
              'matched_within_tolerance':abs(base_psnr-achieved)<=a.psnr_tolerance,
              'calibration_hosts':[p.stem for p in cal],
              'test_hosts':[p.stem for p in test],
              'caution':'If matched_within_tolerance=false do not label the comparison PSNR-matched. '
                        'Baseline capacity and extraction assumptions must be disclosed.'}
    json_write(manifest,out/'calibration.json')
    trials=[]
    attacks=[] if a.suite=='clean' else get_attack_suite(a.suite)
    for h in test:
        host=load_host_rgb(h)
        for wm in wms:
            for method in methods:
                try:
                    t0=time.perf_counter()
                    marked,key=embed(method,host,wm,a.seed,selected,opts)
                    elapsed=time.perf_counter()-t0
                    rec=extract(method,marked,key,host)
                    clean=watermark_metrics(wm,rec)
                    base={'method':method,'host':h.stem,'seed':a.seed,
                          'margin_if_msdp':selected if method=='msdp' else None,
                          'clean_psnr_db':psnr(host,marked),'clean_ssim':ssim(host,marked),
                          'clean_nc':clean['nc'],'embed_seconds':elapsed}
                    trials.append({**base,'attack_id':'none__clean','status':'ok','nc':clean['nc'],'ber':clean['ber']})
                    for attack in attacks:
                        if attack.group=='none':continue
                        try:
                            attacked=apply_attack(marked,attack)
                            score=watermark_metrics(wm,extract(method,attacked,key,host))
                            trials.append({**base,'attack_id':attack.attack_id,'status':'ok',
                                           'attack_group':attack.group,'nc':score['nc'],'ber':score['ber']})
                        except Exception as ex:
                            trials.append({**base,'attack_id':attack.attack_id,'status':'error','error':repr(ex)})
                except Exception as ex:
                    trials.append({'method':method,'host':h.stem,'attack_id':'embedding','status':'error','error':repr(ex)})
    csv_write(trials,out/'test_comparison.csv')
    if not manifest['matched_within_tolerance']:
        print('WARNING: unmatched PSNR on calibration hosts; see calibration.json',flush=True)
    if any(r.get('status')=='error' for r in trials):print('WARNING: failed trials; see CSV',flush=True)

if __name__=='__main__':main()
