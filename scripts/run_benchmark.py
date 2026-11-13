#!/usr/bin/env python3
"""Run MSDP and/or the 16 preserved literature baselines on identical inputs.

Usage (from repository root):
  python scripts/run_benchmark.py --methods msdp --hosts Girl --suite basic20x3
  python scripts/run_benchmark.py --methods msdp,primary_blind --hosts Girl --suite basic20x3
  python scripts/run_benchmark.py --methods msdp,all_baselines --hosts all --suite basic20x3

Every method is compared on the same host, watermark and attacks; published
native/adapted fidelity and blindness distinctions are recorded, not erased.
"""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from qr64_certified.attacks import apply_attack, get_attack_suite, list_attack_suites
from qr64_certified.baselines import (
    ALL_BASELINE_IDS, PRIMARY_BLIND_BASELINE_IDS, embed_baseline,
    extract_baseline, get_baseline_spec, list_baselines,
)
from qr64_certified.common.io import load_host_rgb, load_watermark_binary, save_image
from qr64_certified.common.metrics import psnr, ssim, watermark_metrics
from qr64_certified.msdp import (
    MSDPConfig, embed_msdp, extract_msdp, key_to_jsonable,
)
from qr64_certified.research import protocol_hash, TrialJournal


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return str(obj)


def _resolve_methods(arg: str) -> list[str]:
    result: list[str] = []
    for value in arg.lower().split(','):
        value = value.strip()
        if value == 'all_baselines':
            result.extend(ALL_BASELINE_IDS)
        elif value == 'primary_blind':
            result.extend(PRIMARY_BLIND_BASELINE_IDS)
        elif value == 'msdp':
            result.append('msdp')
        elif value in ALL_BASELINE_IDS:
            result.append(value)
        else:
            raise ValueError(f"Unknown method selection {value!r}; use msdp, all_baselines, primary_blind or full baseline ID")
    return list(dict.fromkeys(result))


def _resolve_hosts(selection: str) -> list[Path]:
    folder = ROOT / 'data' / 'host'
    if selection.lower() == 'all':
        return sorted(folder.glob('*.bmp'))
    output = []
    for name in selection.split(','):
        p = folder / name.strip()
        if not p.exists():
            p = p.with_suffix('.bmp')
        if not p.is_file():
            raise FileNotFoundError(f"Host {name!r} missing in {folder}")
        output.append(p)
    return output


def _resolve_watermarks(selection: str) -> list[Path]:
    folder = ROOT / 'data' / 'watermark'
    if selection.lower() == 'all':
        return sorted(folder.glob('*.png'))
    output = []
    for name in selection.split(','):
        p = folder / name.strip()
        if not p.exists():
            p = p.with_suffix('.png')
        if not p.is_file():
            raise FileNotFoundError(f"Watermark {name!r} missing in {folder}")
        output.append(p)
    return output


def evaluate_one(method_id: str, host_path: Path, watermark_path: Path,
                 attacks: tuple, args: argparse.Namespace,
                 baseline_parameters: dict) -> list[dict[str, Any]]:
    host = load_host_rgb(host_path)
    watermark = load_watermark_binary(watermark_path)
    seed = args.seed
    basis = {
        'method_id': method_id, 'host_id': host_path.stem,
        'watermark_id': watermark_path.stem, 'seed': seed,
        'payload_bits': int(watermark.size),
        'payload_bpp': float(watermark.size / (host.shape[0] * host.shape[1])),
    }
    rows: list[dict[str, Any]] = []
    protocol_sha = protocol_hash(args.suite)
    basis['attack_protocol_sha256'] = protocol_sha
    meta = {}
    is_msdp = method_id == 'msdp'
    if is_msdp:
        basis.update({
            'method_kind': 'proposal', 'blindness_tier': 'key_assisted_blind',
            'requires_original_host': False,
            'fidelity_tier': 'new_unpublished_research_implementation',
            'cover_dependent_key': True,
        })
    else:
        spec = get_baseline_spec(method_id)
        basis.update({
            'method_kind': 'literature_baseline', 'blindness_tier': spec.blindness_tier,
            'requires_original_host': spec.requires_original_host,
            'fidelity_tier': spec.fidelity_tier,
            'cover_dependent_key': 'see_disclosure',
        })
    try:
        t0 = time.perf_counter()
        if is_msdp:
            config = MSDPConfig(normalized_margin=args.normalized_margin, determinant_floor=args.determinant_floor,
                                mean_penalty=args.mean_penalty, pattern_count=args.pattern_count,
                                pattern_selector=args.pattern_selector,
                                integer_repair=not args.disable_integer_repair)
            marked, key, meta = embed_msdp(host, watermark, seed=seed, config=config)
        else:
            opts = baseline_parameters.get(method_id, {})
            marked, key = embed_baseline(method_id, host, watermark, seed=seed,
                repeat=opts.get('repeat', 1), step=opts.get('step', None),
                method_params=opts.get('method_params', {}))
        embed_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        recovered = extract_msdp(marked, key) if is_msdp else extract_baseline(
            marked, key, original_host=host if basis['requires_original_host'] else None)
        extract_time = time.perf_counter() - t1
        quality = {'clean_psnr_db': psnr(host, marked), 'clean_ssim': ssim(host, marked)}
        clean = watermark_metrics(watermark, recovered)
        common = {**basis, **quality,
                  'clean_nc': clean['nc'], 'clean_ber': clean['ber'],
                  'clean_bit_accuracy': clean['bit_accuracy'],
                  'embed_seconds': embed_time, 'clean_extract_seconds': extract_time,
                  'pattern_map_bytes': meta.get('pattern_map_bytes'),
                  'minimum_attained_normalized_residual': meta.get('minimum_attained_normalized_residual'),
                  'minimum_attained_signed_determinant': meta.get('minimum_attained_signed_determinant'),
                  'active_box_projection_blocks': meta.get('active_box_projection_blocks'),
                  'integer_constraint_violations': meta.get('certified_integer_constraint_violations'),
                  'pattern_selector': meta.get('pattern_selector'),
                  'integer_repair':meta.get('integer_repair'),
                 }
        if is_msdp:
            on_cover = extract_msdp(host, key)
            cover_stat = watermark_metrics(watermark, on_cover)
            common['cover_only_bit_accuracy'] = cover_stat['bit_accuracy']
            common['cover_only_nc'] = cover_stat['nc']
        else:
            common['cover_only_bit_accuracy'] = None
            common['cover_only_nc'] = None
        rows.append({**common, 'attack_id': 'none__clean', 'attack_group':'none',
                     'status':'ok', 'attacked_nc':clean['nc'],
                     'attacked_ber':clean['ber'],
                     'attacked_bit_accuracy':clean['bit_accuracy']})
        if args.save_images and is_msdp:
            image_dir = Path(args.output) / 'images'
            image_dir.mkdir(parents=True, exist_ok=True)
            save_image(image_dir / f'{host_path.stem}_{watermark_path.stem}_msdp.png', marked)
            save_image(image_dir / f'{host_path.stem}_{watermark_path.stem}_recovered.png', recovered)
            (image_dir / f'{host_path.stem}_{watermark_path.stem}_key.json').write_text(
                json.dumps(key_to_jsonable(key), indent=2), encoding='utf-8')
            (image_dir / f'{host_path.stem}_{watermark_path.stem}_embed_metadata.json').write_text(
                json.dumps(_json_safe(meta), indent=2), encoding='utf-8')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        # Preserve one row per planned attack, including failed embed trials.
        rows.append({**basis, 'attack_id':'none__clean','attack_group':'none',
                     'attack_category':'clean','status':'error','error':error})
        for attack in attacks:
            if attack.attack_id == 'none__clean':continue
            rows.append({**basis,'attack_id':attack.attack_id,
                         'attack_group':attack.group,'attack_category':attack.category,
                         'attack_severity':attack.severity,'status':'error',
                         'error':'embedding failed: '+error})
        if args.fail_fast:
            raise
        return rows

    for attack in attacks:
        if attack.attack_id == 'none__clean':
            continue
        try:
            t2 = time.perf_counter()
            attacked = apply_attack(marked, attack)
            attack_time = time.perf_counter() - t2
            t3 = time.perf_counter()
            output = extract_msdp(attacked, key) if is_msdp else extract_baseline(
                attacked, key, original_host=host if basis['requires_original_host'] else None)
            extract_time = time.perf_counter() - t3
            met = watermark_metrics(watermark, output)
            rows.append({**common, 'attack_id':attack.attack_id,
                         'attack_group':attack.group,'attack_category':attack.category,
                         'attack_severity':attack.severity,
                         'attack_params':json.dumps(attack.params,sort_keys=True), 'status':'ok',
                         'attack_seconds':attack_time, 'extract_seconds':extract_time,
                         'attacked_nc':met['nc'], 'attacked_ber':met['ber'],
                         'attacked_bit_accuracy':met['bit_accuracy'],
                         'attacked_psnr_db':psnr(host, attacked),
                         'attacked_ssim':ssim(host, attacked)})
        except Exception as exc:
            rows.append({**common, 'attack_id':attack.attack_id,
                         'attack_group':attack.group,'attack_category':attack.category,
                         'attack_severity':attack.severity,
                         'attack_params':json.dumps(attack.params,sort_keys=True),'status':'error',
                         'error':f'{type(exc).__name__}: {exc}'})
            if args.fail_fast:
                raise
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--methods', default='msdp', help='msdp, primary_blind, all_baselines, or comma-separated IDs')
    parser.add_argument('--hosts', default='Girl', help='host name(s), without .bmp, or all')
    parser.add_argument('--watermarks', default='wm', help='watermark name(s), without .png, or all')
    parser.add_argument('--suite', default='basic20x3', choices=('clean', *list_attack_suites()))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--normalized-margin', type=float, default=0.015)
    parser.add_argument('--determinant-floor', type=float, default=0.25)
    parser.add_argument('--mean-penalty', type=float, default=2.0)
    parser.add_argument('--pattern-count', type=int, default=8)
    parser.add_argument('--output', default='results/latest')
    parser.add_argument('--save-images', action='store_true', help='Save MSDP PNGs + JSON extraction key')
    parser.add_argument('--pattern-selector',choices=('minimax','fixed','random','max_abs'),default='minimax',
                        help='Controlled ablation; minimax preserves original MSDP')
    parser.add_argument('--disable-integer-repair',action='store_true',
                        help='Diagnostic ablation; does not guarantee integer margin')
    parser.add_argument('--strict',action='store_true',help='Exit nonzero after saving if any experiment fails')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--list-baselines', action='store_true')
    args = parser.parse_args()

    if args.list_baselines:
        for spec in list_baselines():
            print(spec['canonical_id'], '|', spec['blindness_tier'], '|', spec['fidelity_tier'])
        return
    methods = _resolve_methods(args.methods)
    hosts = _resolve_hosts(args.hosts)
    marks = _resolve_watermarks(args.watermarks)
    attacks = () if args.suite == 'clean' else get_attack_suite(args.suite)
    baseline_parameters = json.loads((ROOT / 'configs' / 'baseline_parameters.json').read_text(encoding='utf-8'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    spec={'experiment':'benchmark','methods':methods,'hosts':[h.stem for h in hosts],
          'watermarks':[m.stem for m in marks],'seed':args.seed,'suite':args.suite,
          'protocol_hash':protocol_hash(args.suite),
          'msdp_params':{k:getattr(args,k) for k in ('normalized_margin','determinant_floor',
                          'mean_penalty','pattern_count','pattern_selector','disable_integer_repair')},
          'baseline_parameters':baseline_parameters}
    journal=TrialJournal(out,spec=spec)
    for method in methods:
        for host in hosts:
            for mark in marks:
                trial_id=f'{method}|{host.stem}|{mark.stem}|{args.seed}'
                if journal.has(trial_id):
                    print('Resuming: skip',trial_id,flush=True)
                    continue
                print(f'Running {method} | host={host.stem} | watermark={mark.stem} | suite={args.suite}', flush=True)
                trial = evaluate_one(method, host, mark, attacks, args, baseline_parameters)
                journal.append(trial_id,trial)
                clean = next((r for r in trial if r['attack_id']=='none__clean'), None)
                if clean and clean.get('status')=='ok':
                    print(f"  clean PSNR={clean['clean_psnr_db']:.2f} dB  "
                          f"NC={clean['clean_nc']:.4f}  BER={clean['clean_ber']:.4f}", flush=True)
                else:
                    print('  ERROR',trial[0].get('error'),flush=True)
    rows=journal.data
    _write_csv(rows, out / 'per_attack.csv')
    summarized: list[dict[str, Any]] = []
    for method in methods:
        group = [row for row in rows if row.get('method_id')==method]
        ok = [row for row in group if row.get('status')=='ok']
        attacked = [row for row in ok if row.get('attack_id')!='none__clean']
        clean = [row for row in ok if row.get('attack_id')=='none__clean']
        summarized.append({
            'method_id': method,
            'total_rows':len(group), 'failed_rows':len(group)-len(ok),
            'clean_trials':len(clean), 'attacked_trials':len(attacked),
            'mean_clean_psnr_db':float(np.mean([r['clean_psnr_db'] for r in clean])) if clean else None,
            'mean_clean_nc':float(np.mean([r['clean_nc'] for r in clean])) if clean else None,
            'mean_attacked_nc':float(np.mean([r['attacked_nc'] for r in attacked])) if attacked else None,
            'mean_attacked_ber':float(np.mean([r['attacked_ber'] for r in attacked])) if attacked else None,
        })
    _write_csv(summarized, out / 'summary.csv')
    (out / 'summary.json').write_text(json.dumps(_json_safe({
        'experiment': {'methods':methods, 'hosts':[p.stem for p in hosts],
                       'watermarks':[p.stem for p in marks], 'suite':args.suite,
                       'seed':args.seed, 'normalized_margin':args.normalized_margin,
                       'determinant_floor':args.determinant_floor,
                       'mean_penalty':args.mean_penalty,
                       'pattern_count':args.pattern_count,
                       'pattern_selector':args.pattern_selector,
                       'integer_repair':not args.disable_integer_repair,
                       'attack_protocol_sha256':protocol_hash(args.suite),
                       },
        'summaries': summarized,
    }), indent=2), encoding='utf-8')
    manifest=[{'attack_id':x.attack_id,'group':x.group,'params':x.params,
               'category':x.category,'severity':x.severity} for x in attacks]
    (out/'attack_protocol.json').write_text(json.dumps(_json_safe({
        'suite':args.suite,'sha256':protocol_hash(args.suite),'attacks':manifest}),indent=2),encoding='utf-8')
    print(f'Wrote {len(rows)} rows to {out.resolve()}', flush=True)
    failed=sum(r.get('status')!='ok' for r in rows)
    if failed:print(f'WARNING: {failed} failed rows; see per_attack.csv',flush=True)
    if args.strict and failed:raise SystemExit(1)


if __name__ == '__main__':
    main()
