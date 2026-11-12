#!/usr/bin/env python3
"""Optional publication-style PDF figures from research CSVs; no fabrication."""
import argparse
import csv
from pathlib import Path
import numpy as np

def read(path):
    with open(path,encoding='utf-8',newline='') as fh:return list(csv.DictReader(fh))
def value(x):
    try:return float(x)
    except (TypeError,ValueError):return np.nan

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--kind',choices=('sweep','ablation','attack','security','payload'),required=True)
    p.add_argument('--output',required=True,help='Output PDF filename')
    a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.bbox':'tight','pdf.fonttype':42})
    rows=read(a.input)
    if not rows:raise ValueError('Empty CSV')
    fig,ax=plt.subplots(figsize=(5.6,3.7))
    if a.kind=='sweep':
        good=[r for r in rows if r.get('failed_rows')=='0' and r.get('mean_attack_nc') not in (None,'')]
        if not good:raise ValueError('No complete sweep results: use suite other than clean')
        xs=np.array([value(r['mean_clean_psnr_db']) for r in good]);ys=np.array([value(r['mean_attack_nc']) for r in good])
        ax.scatter(xs,ys,s=26,color='black')
        for x,y,r in zip(xs,ys,good):ax.annotate(r.get('variant',''),(x,y),xytext=(3,3),textcoords='offset points',fontsize=6)
        ax.set(xlabel='Clean PSNR (dB)',ylabel='Attacked NC',title='MSDP parameter sensitivity')
    elif a.kind=='ablation':
        values=[r for r in rows if r.get('mean_attack_nc') not in ('',None)]
        if not values:raise ValueError('Ablation needs an attacked suite')
        labels=[r['variant'].replace('_',' ') for r in values]
        ax.barh(labels,[value(r['mean_attack_nc']) for r in values],color='0.4')
        ax.set(xlabel='Mean attacked NC (successful trials)',title='Controlled ablations',xlim=(0,1))
    elif a.kind=='security':
        ax.plot([value(r['fpr']) for r in rows],[value(r['tpr']) for r in rows],color='black',lw=1.6)
        ax.plot([0,1],[0,1],'--',color='0.6',lw=.8)
        ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Detection ROC',xlim=(0,1),ylim=(0,1))
    elif a.kind=='payload':
        used=[r for r in rows if r.get('attack_id')=='none__clean' and r.get('status')=='ok']
        if not used:raise ValueError('No payload clean rows')
        labels=sorted(set(r['placement'] for r in used))
        for placement in labels:
            part=[r for r in used if r['placement']==placement]
            points={}
            for r in part:points.setdefault(value(r['payload_bpp']),[]).append(value(r['clean_psnr_db']))
            x=sorted(points)
            ax.plot(x,[np.mean(points[t]) for t in x],marker='o',lw=1,label=placement)
        ax.legend(frameon=False)
        ax.set(xlabel='Payload (bits per host pixel)',ylabel='Clean PSNR (dB)',title='Sparse placement trade-off')
    else:
        labels=[r['category'] for r in rows if r.get('mean_nc_success_only') not in ('',None)]
        values=[value(r['mean_nc_success_only']) for r in rows if r.get('mean_nc_success_only') not in ('',None)]
        ax.barh(labels,values,color='0.4')
        ax.set(xlabel='Mean NC (successful trials)',title='Robustness by attack category',xlim=(0,1))
    ax.grid(axis='x' if a.kind in ('ablation','attack') else 'y',alpha=.2)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(out);plt.close(fig)
    print('Saved',out)

if __name__=='__main__':main()
