#!/usr/bin/env python3
"""Synthetic extremes: saturation, gray levels, flat and checkerboard blocks."""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.msdp import MSDPConfig,embed_msdp,extract_msdp
from qr64_certified.common.metrics import psnr,watermark_metrics
from qr64_certified.research import csv_write,json_write

def images(size,seed):
    rng=np.random.default_rng(seed)
    grid=np.indices((size,size)).sum(axis=0)%2
    out={
       'black':np.zeros((size,size,3),dtype=np.uint8),
       'white':np.full((size,size,3),255,dtype=np.uint8),
       'mid_gray':np.full((size,size,3),128,dtype=np.uint8),
       'checkerboard':np.broadcast_to((grid*255)[...,None],(size,size,3)).astype(np.uint8).copy(),
       'gradient':np.broadcast_to(np.arange(size,dtype=np.float64)[None,:,None]*255/(size-1),
                                  (size,size,3)).astype(np.uint8).copy(),
       'random':rng.integers(0,256,size=(size,size,3),dtype=np.uint8),
       'saturated_rgb':np.stack([grid*255,(1-grid)*255,grid*255],axis=-1).astype(np.uint8),
    }
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-size',type=int,default=64)
    p.add_argument('--margins',default='.005,.015,.03')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.image_size%8 or a.image_size<8:p.error('--image-size must be a multiple of 8')
    rng=np.random.default_rng(a.seed+33)
    wm=(rng.integers(0,2,(a.image_size//8,a.image_size//8),dtype=np.uint8)*255)
    rows=[]
    for name,image in images(a.image_size,a.seed).items():
        for margin in [float(t) for t in a.margins.split(',')]:
            try:
                marked,key,meta=embed_msdp(image,wm,seed=a.seed,
                                         config=MSDPConfig(normalized_margin=margin))
                score=watermark_metrics(wm,extract_msdp(marked,key))
                row={'image_type':name,'margin':margin,'status':'ok',
                     'clean_psnr_db':psnr(image,marked),'clean_nc':score['nc'],
                     'constraint_violations':meta['certified_integer_constraint_violations'],
                     'minimum_slack':meta['minimum_signed_affine_slack'],
                     'box_active_blocks':meta['active_box_projection_blocks']}
            except Exception as ex:
                row={'image_type':name,'margin':margin,'status':'error','error':repr(ex)}
            rows.append(row)
            csv_write(rows,Path(a.output)/'numerical_stress.csv')
    json_write({'total':len(rows),'failures':sum(r['status']=='error' for r in rows),
                'caution':'Synthetic image stress tests are not natural-image robustness evidence.'},
               Path(a.output)/'numerical_summary.json')

if __name__=='__main__':main()
