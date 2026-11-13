#!/usr/bin/env python3
"""Payload placement/quality/overhead study, with standalone sparse extractor."""
import argparse
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.sparse import encode_sparse,extract_sparse
from qr64_certified.common.io import load_host_rgb,load_watermark_binary,save_image
from qr64_certified.common.metrics import psnr,ssim,watermark_metrics
from qr64_certified.research import hosts,watermarks,parse_seeds,csv_write,json_write,TrialJournal,protocol_hash
from qr64_certified.attacks import apply_attack,get_attack_suite

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hosts',default='Girl')
    p.add_argument('--watermarks',default='wm')
    p.add_argument('--seeds',default='2026')
    p.add_argument('--fractions',default='.125,.25,.5,1')
    p.add_argument('--placements',default='random,low_variance,high_variance')
    p.add_argument('--suite',default='basic20x3')
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);saved_example=(out/'example_sparse_key.json').exists()
    host_files=hosts(a.hosts);wm_files=watermarks(a.watermarks);seed_values=parse_seeds(a.seeds)
    fractions=[float(x) for x in a.fractions.split(',')];placements=a.placements.split(',')
    spec={'experiment':'payload','hosts':[h.stem for h in host_files],
          'watermarks':[w.stem for w in wm_files],'seeds':seed_values,
          'fractions':fractions,'placements':placements,'suite':a.suite,
          'protocol_hash':protocol_hash(a.suite)}
    journal=TrialJournal(out,spec=spec)
    attacks=[] if a.suite=='clean' else get_attack_suite(a.suite)
    for h in host_files:
        host=load_host_rgb(h)
        for w in wm_files:
            wm=load_watermark_binary(w)
            for seed in seed_values:
                for fraction in fractions:
                    for placement in placements:
                        trial_id=f'{h.stem}|{w.stem}|{seed}|{fraction:g}|{placement}'
                        if journal.has(trial_id):
                            continue
                        trial=[]
                        marked,key,pos,meta=encode_sparse(host,wm,fraction=fraction,
                                                placement=placement,seed=seed)
                        recovered,actual=extract_sparse(marked,key)
                        if not np.array_equal(pos,actual):raise AssertionError('Sparse decoding positions mismatch')
                        source=wm.reshape(-1)[pos]
                        clean=watermark_metrics(source,recovered)
                        shared={'host_id':h.stem,'watermark_id':w.stem,'seed':seed,
                                'fraction':fraction,'placement':placement,
                                'payload_bits':len(pos),'payload_bpp':len(pos)/(host.shape[0]*host.shape[1]),
                                'pattern_map_bytes':meta['sparse_pattern_map_bytes'],
                                'selection_map_bytes':meta['selection_mask_bytes'],
                                'total_map_bytes_excluding_seed_and_json':meta['sparse_pattern_map_bytes']+meta['selection_mask_bytes'],
                                'clean_psnr_db':psnr(host,marked),'clean_ssim':ssim(host,marked),
                                'clean_nc':clean['nc'],'clean_ber':clean['ber']}
                        trial.append({**shared,'attack_id':'none__clean','attack_category':'clean',
                                     'status':'ok','attacked_nc':clean['nc'],'attacked_ber':clean['ber']})
                        for attack in attacks:
                            if attack.group=='none':continue
                            try:
                                attacked=apply_attack(marked,attack)
                                got,pos2=extract_sparse(attacked,key)
                                score=watermark_metrics(wm.reshape(-1)[pos2],got)
                                trial.append({**shared,'attack_id':attack.attack_id,
                                             'attack_category':attack.category,'status':'ok',
                                             'attacked_nc':score['nc'],'attacked_ber':score['ber']})
                            except Exception as ex:
                                trial.append({**shared,'attack_id':attack.attack_id,
                                             'attack_category':attack.category,'status':'error','error':repr(ex)})
                        journal.append(trial_id,trial)
                        if not saved_example:
                            json_write(key,out/'example_sparse_key.json')
                            save_image(out/'example_sparse_marked.png',marked)
                            np.save(out/'example_sparse_extracted_bits.npy',recovered)
                            np.save(out/'example_sparse_active_positions.npy',pos)
                            saved_example=True
    csv_write(journal.data,out/'payload_trials.csv')
    json_write({'method':'sparse block placement (one bit per active block)',
                 'disclosure':'The encoder currently embeds full image first, then copies active blocks. '
                 'Sparse embedding runtime cannot be inferred; independently extracted compact keys are tested. '
                 'Selection overhead for adaptive placement is counted.','trials':len(journal.data)},out/'protocol.json')
    print('Saved',len(journal.data),'rows')

if __name__=='__main__':main()
