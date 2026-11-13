"""Shared, explicit protocol for reproducible MSDP research experiments.

All wrappers call the same embedder, extractor, metrics, and attack registry as
run_benchmark.py; failure rows are preserved and never silently imputed.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
from .attacks import apply_attack, get_attack_suite
from .common.io import load_host_rgb, load_watermark_binary
from .common.metrics import psnr, ssim, watermark_metrics
from .msdp import MSDPConfig, embed_msdp, extract_msdp

ROOT = Path(__file__).resolve().parents[2]


def select_files(directory: str, selection: str, suffix: str) -> list[Path]:
    folder = ROOT / 'data' / directory
    if selection == 'all':
        files = sorted(folder.glob('*'+suffix))
    else:
        files = []
        for name in selection.split(','):
            name=name.strip()
            path=folder / name
            if path.suffix.lower()!=suffix:
                path=folder/(name+suffix)
            files.append(path)
    if not files or any(not p.is_file() for p in files):
        raise FileNotFoundError(f'Missing {directory} files: {[str(f) for f in files]}')
    return files


def hosts(value: str) -> list[Path]:
    return select_files('host', value, '.bmp')


def watermarks(value: str) -> list[Path]:
    return select_files('watermark', value, '.png')


def safe(v: Any) -> Any:
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return safe(float(v))
    if isinstance(v,float) and not math.isfinite(v): return None
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,np.ndarray): return safe(v.tolist())
    return v


def csv_write(rows: list[dict], filename: str | Path) -> None:
    filename=Path(filename)
    filename.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        filename.write_text('',encoding='utf-8')
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with filename.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader()
        w.writerows([safe(row) for row in rows])


def json_write(obj: Any, filename: str | Path) -> None:
    filename=Path(filename)
    filename.parent.mkdir(parents=True,exist_ok=True)
    filename.write_text(json.dumps(safe(obj),ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def protocol_hash(suite: str) -> str:
    attack_configs=[] if suite=='clean' else [asdict(a) for a in get_attack_suite(suite)]
    data=json.dumps(attack_configs,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(data.encode()).hexdigest()


def get_attacks(suite: str):
    return () if suite=='clean' else get_attack_suite(suite)


def measure_trial(host: np.ndarray, wm: np.ndarray, *, config: MSDPConfig,
                  seed: int, suite: str='sanity', label: str='full',
                  host_id: str='', watermark_id: str='', fail_fast: bool=False,
                  attacked_quality: bool=True) -> list[dict]:
    base={'variant':label,'host_id':host_id,'watermark_id':watermark_id,'seed':seed,
          'normalized_margin':config.normalized_margin,'determinant_floor':config.determinant_floor,
          'mean_penalty':config.mean_penalty,'pattern_count':config.pattern_count,
          'pattern_selector':config.pattern_selector,'integer_repair':config.integer_repair,
          'suite':suite,'attack_protocol_sha256':protocol_hash(suite)}
    rows=[]
    try:
        start=time.perf_counter()
        marked,key,meta=embed_msdp(host,wm,config=config,seed=seed)
        embed_s=time.perf_counter()-start
        start=time.perf_counter()
        recovered=extract_msdp(marked,key)
        extract_s=time.perf_counter()-start
        score=watermark_metrics(wm,recovered)
        common={**base,'status':'ok','clean_psnr_db':psnr(host,marked),'clean_ssim':ssim(host,marked),
                'clean_nc':score['nc'],'clean_ber':score['ber'],
                'embed_seconds':embed_s,'clean_extract_seconds':extract_s,
                'pattern_map_bytes':meta['pattern_map_bytes'],
                'integer_constraint_violations':meta['certified_integer_constraint_violations'],
                'minimum_signed_affine_slack':meta['minimum_signed_affine_slack'],
                'minimum_signed_determinant':meta['minimum_attained_signed_determinant'],
                'minimum_normalized_residual':meta['minimum_attained_normalized_residual'],
                'active_box_projection_blocks':meta['active_box_projection_blocks'],
                'squared_distortion':meta['total_squared_distortion']}
        rows.append({**common,'attack_id':'none__clean','attack_group':'none',
                     'attack_category':'clean','severity':'none','attacked_nc':score['nc'],
                     'attacked_ber':score['ber'],'attacked_psnr_db':psnr(host,marked)})
    except Exception as ex:
        if fail_fast: raise
        return [{**base,'status':'error','attack_id':'embedding','error':repr(ex)}]
    for attack in get_attacks(suite):
        if attack.group=='none': continue
        try:
            start=time.perf_counter()
            attacked=apply_attack(marked,attack)
            attack_s=time.perf_counter()-start
            start=time.perf_counter()
            got=extract_msdp(attacked,key)
            extract_s=time.perf_counter()-start
            sc=watermark_metrics(wm,got)
            row={**common,'attack_id':attack.attack_id,'attack_group':attack.group,
                         'attack_category':attack.category,'severity':attack.severity,
                         'attacked_nc':sc['nc'],'attacked_ber':sc['ber'],
                         'attack_seconds':attack_s,'extract_seconds':extract_s}
            if attacked_quality:
                row['attacked_psnr_db']=psnr(host,attacked)
                row['attacked_ssim']=ssim(host,attacked)
            rows.append(row)
        except Exception as ex:
            rows.append({**common,'status':'error','attack_id':attack.attack_id,
                         'attack_group':attack.group,'attack_category':attack.category,
                         'severity':attack.severity,'error':repr(ex)})
            if fail_fast: raise
    return rows


def aggregate(rows: list[dict], keys=('variant',)) -> list[dict]:
    groups: dict[tuple,list[dict]]={}
    for r in rows:
        groups.setdefault(tuple(r.get(k,'') for k in keys),[]).append(r)
    out=[]
    for label,gr in sorted(groups.items(),key=lambda a:str(a[0])):
        good=[r for r in gr if r.get('status')=='ok']
        attacked=[r for r in good if r.get('attack_id')!='none__clean']
        clean=[r for r in good if r.get('attack_id')=='none__clean']
        def avg(vals):
            vals=[float(v) for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None
        out.append({**dict(zip(keys,label)), 'total_rows':len(gr),
                    'failed_rows':len(gr)-len(good),'clean_trials':len(clean),
                    'attack_trials':len(attacked),
                    'mean_clean_psnr_db':avg(r.get('clean_psnr_db') for r in clean),
                    'mean_clean_ssim':avg(r.get('clean_ssim') for r in clean),
                    'mean_clean_nc':avg(r.get('clean_nc') for r in clean),
                    'mean_attack_nc':avg(r.get('attacked_nc') for r in attacked),
                    'mean_attack_ber':avg(r.get('attacked_ber') for r in attacked),
                    'mean_embed_seconds':avg(r.get('embed_seconds') for r in clean),
                    'mean_pattern_map_bytes':avg(r.get('pattern_map_bytes') for r in clean),
                    'mean_constraint_violations':avg(r.get('integer_constraint_violations') for r in clean)})
    return out


def write_result_bundle(rows: list[dict], out: str | Path, *, keys=('variant',), spec=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary=aggregate(rows,keys=keys)
    csv_write(rows,out/'per_attack.csv')
    csv_write(summary,out/'summary.csv')
    json_write({'experiment':spec or {},'summary':summary},out/'summary.json')
    return summary


class TrialJournal:
    """Resumable, append-only complete-trial journal.

    A JSONL record is committed only after all attacks in a trial finish. The
    run fingerprint prevents accidental resumption with incompatible options.
    Partial final lines (e.g., interrupted notebook) are discarded on resume.
    """
    def __init__(self, output: str | Path, *, spec: dict, resume: bool = True):
        self.out = Path(output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / 'trials.jsonl'
        self.meta = self.out / 'trial_journal_protocol.json'
        self.spec = safe(spec)
        self.keys: set[str] = set()
        self.data: list[dict] = []
        if self.meta.exists():
            previous = json.loads(self.meta.read_text(encoding='utf-8'))
            if previous != self.spec:
                raise ValueError(f'Existing journal protocol differs: {self.meta}; use new --output')
            if not resume:
                raise FileExistsError(f'Existing journal at {self.path}, use new --output')
        elif self.path.exists():
            raise ValueError(f'Journal exists without protocol: {self.path}')
        else:
            json_write(self.spec, self.meta)
        if self.path.exists():
            with self.path.open('rb+') as fh:
                contents = fh.read()
                end = contents.rfind(b'\n')
                if len(contents) and end + 1 != len(contents):
                    fh.seek(end + 1)
                    fh.truncate()
                for line in contents[:end+1].splitlines():
                    record = json.loads(line)
                    key = record['trial_id']
                    if key in self.keys:
                        raise ValueError(f'Duplicate journal trial ID: {key}')
                    self.keys.add(key)
                    self.data.extend(record['rows'])

    def has(self, trial_id: str) -> bool:
        return trial_id in self.keys

    def append(self, trial_id: str, rows: list[dict]) -> None:
        if trial_id in self.keys:
            raise ValueError(f'Trial already saved: {trial_id}')
        record = {'trial_id': trial_id, 'rows': safe(rows)}
        with self.path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(',',':')) + '\n')
            fh.flush()
            os.fsync(fh.fileno())
        self.keys.add(trial_id)
        self.data.extend(rows)


def add_common_args(parser, *, default_suite='basic20x3'):
    parser.add_argument('--hosts',default='Girl')
    parser.add_argument('--watermarks',default='wm')
    parser.add_argument('--suite',default=default_suite,
                        choices=('clean', *sorted(__import__('qr64_certified.attacks',fromlist=['list_attack_suites']).list_attack_suites())))
    parser.add_argument('--seeds',default='2026',help='Comma-separated integer seeds')
    parser.add_argument('--output',required=True)
    parser.add_argument('--fail-fast',action='store_true')
    parser.add_argument('--skip-attacked-quality', action='store_true',
                        help='Skip attacked PSNR/SSIM (retain clean PSNR/SSIM and attacked NC/BER)')
    return parser


def parse_seeds(text: str) -> list[int]:
    seeds=[int(p) for p in text.split(',')]
    if not seeds: raise ValueError('Empty seeds')
    return seeds
