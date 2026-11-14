#!/usr/bin/env python3
"""Host-cluster paired bootstrap and exact paired sign-flip inference.

Inputs MUST use identical host/watermark/seed/attack keys, payloads, and
attack protocol SHA256 across methods. Any failed/missing/duplicate row is an
ERROR; it is never dropped to make the method look better.

Independence level: host image (two watermarks and 60 attacks within one host
are repeated observations). The eight fixed hosts are NOT a random population
sample; inferential p-values are conditional/exploratory, not population claims.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import numpy as np

METRICS = {
    'attacked_ber': ('attacked_ber', 'lower'),
    'attacked_nc': ('attacked_nc', 'higher'),
    'clean_psnr_db': ('clean_psnr_db', 'higher'),
}
CLEAN = 'none__clean'
KEY_FIELDS = ('host_id', 'watermark_id', 'seed', 'attack_id')


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        if not rows:
            f.write('')
            return
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def load_data(paths: list[str], method_field: str) -> tuple[dict, str, dict]:
    data: dict[str, dict[tuple[str, ...], dict]] = defaultdict(dict)
    per_method_info = {}
    kind = method_field
    for path in paths:
        with Path(path).open('r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            if kind == 'auto':
                kind = 'method_id' if 'method_id' in (reader.fieldnames or []) else 'variant'
            required = set(KEY_FIELDS) | {kind, 'status', 'attack_protocol_sha256'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{path} missing fields: {sorted(required-set(reader.fieldnames or []))}')
            for row in reader:
                method = row[kind]
                if not method:
                    raise ValueError(f'{path} has empty method name')
                key = tuple(row[x] for x in KEY_FIELDS)
                if key in data[method]:
                    raise ValueError(f'Duplicate method/key {method}: {key}')
                data[method][key] = row
                extra = (row.get('blindness_tier',''), row.get('fidelity_tier',''))
                if method in per_method_info and per_method_info[method] != extra:
                    raise ValueError(f'Inconsistent method metadata for {method}')
                per_method_info[method] = extra
    if not data:
        raise ValueError('No rows found')
    return dict(data), kind, per_method_info


def validate(data: dict, reference: str) -> tuple[list[tuple], list[str]]:
    if reference not in data:
        raise ValueError(f'Reference {reference!r} absent; available: {list(data)}')
    expected = set(data[reference])
    if not expected:
        raise ValueError('Reference contains no results')
    for method, rows in data.items():
        keys = set(rows)
        if keys != expected:
            raise ValueError(f'{method}: {len(expected-keys)} missing and {len(keys-expected)} extra paired keys')
        for key, row in rows.items():
            if row['status'] != 'ok':
                raise ValueError(f'{method} failed at {key}: {row.get("error", "")!r}')
            orig = data[reference][key]
            if row['attack_protocol_sha256'] != orig['attack_protocol_sha256']:
                raise ValueError(f'Attack protocol mismatch at {method}, {key}')
            if (row.get('payload_bits') and orig.get('payload_bits')
                    and row.get('payload_bits') != orig.get('payload_bits')):
                raise ValueError(f'Payload mismatch at {method}, {key}')
    hosts = sorted({k[0] for k in expected})
    for k in expected:
        if not all(k):
            raise ValueError(f'Empty pairing field: {k}')
    return sorted(expected), hosts


def parse_float(row: dict, field: str, method: str, key: tuple) -> float:
    try:
        value = float(row[field])
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f'Missing/non-numeric {field} for {method} {key}') from e
    if not np.isfinite(value):
        raise ValueError(f'Nonfinite {field} for {method} {key}')
    return value


def signflip_pvalue(d: np.ndarray) -> tuple[float, str]:
    n = len(d)
    observed = abs(float(np.mean(d)))
    if n <= 16:
        signs = np.array(list(itertools.product((-1., 1.), repeat=n)), dtype=np.float64)
        means = np.abs(np.mean(signs * d[None, :], axis=1))
        return float(np.mean(means >= observed - 1e-12)), 'exact_host_sign_flip'
    # Beyond 16 hosts, use deterministic Monte Carlo with plus-one correction.
    rng = np.random.default_rng(1703)
    signs = rng.choice((-1.0, 1.0), size=(100000, n))
    means = np.abs(np.mean(signs * d[None, :], axis=1))
    return float((np.count_nonzero(means >= observed-1e-12)+1)/100001), 'mc_host_sign_flip'


def paired_inference(d: np.ndarray, rng: np.random.Generator, b: int) -> dict:
    n = len(d)
    if n < 2:
        raise ValueError('At least two host images are needed for paired inference')
    indices = rng.integers(0, n, size=(b, n))
    samples = np.mean(d[indices], axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    sd = float(np.std(d, ddof=1))
    p, test = signflip_pvalue(d)
    return {'mean_reference_advantage': float(np.mean(d)),
            'ci95_low': float(lo), 'ci95_high': float(hi),
            'paired_cohen_dz': float(np.mean(d)/sd) if sd > 1e-14 else None,
            'p_two_sided': p, 'test': test,
            'n_hosts': n, 'bootstrap_replicates': b,
            'host_advantages_json': json.dumps([float(x) for x in d])}


def holm(rows: list[dict]) -> None:
    """Holm correction separately within each metric across all baselines."""
    for metric in METRICS:
        selected = sorted([r for r in rows if r['metric'] == metric],
                          key=lambda r: r['p_two_sided'])
        prev = 0.0
        for index, row in enumerate(selected):
            raw = min(1.0, row['p_two_sided']*(len(selected)-index))
            prev = max(prev, raw)
            row['p_holm'] = prev


def analyze(data: dict, reference: str, *, bootstrap: int, seed: int) -> tuple[list, list, list]:
    keys, hosts = validate(data, reference)
    rng = np.random.default_rng(seed)
    summary, overall, category_rows = [], [], []
    attacked_keys = [k for k in keys if k[3] != CLEAN]
    clean_keys = [k for k in keys if k[3] == CLEAN]
    if not attacked_keys or not clean_keys:
        raise ValueError('Requires both attacked and clean paired results')
    for method, rows in sorted(data.items()):
        summary.append({
            'method': method, 'n_hosts': len(hosts),
            'n_clean_rows': len(clean_keys), 'n_attacked_rows': len(attacked_keys),
            'clean_psnr_db': float(np.mean([parse_float(rows[k], 'clean_psnr_db',method,k) for k in clean_keys])),
            'clean_nc': float(np.mean([parse_float(rows[k], 'clean_nc',method,k) for k in clean_keys])),
            'clean_ber': float(np.mean([parse_float(rows[k], 'clean_ber',method,k) for k in clean_keys])),
            'clean_ssim': float(np.mean([parse_float(rows[k], 'clean_ssim',method,k) for k in clean_keys])),
            'attacked_nc': float(np.mean([parse_float(rows[k], 'attacked_nc',method,k) for k in attacked_keys])),
            'attacked_ber': float(np.mean([parse_float(rows[k], 'attacked_ber',method,k) for k in attacked_keys])),
        })
    for method in sorted(data):
        if method == reference:
            continue
        for metric, (col, direction) in METRICS.items():
            selected = clean_keys if metric.startswith('clean_') else attacked_keys
            grouped: dict[str, list] = defaultdict(list)
            for k in selected:
                ref_value = parse_float(data[reference][k], col, reference, k)
                alt_value = parse_float(data[method][k], col, method, k)
                advantage = (ref_value-alt_value) if direction == 'higher' else (alt_value-ref_value)
                grouped[k[0]].append(advantage)
            host_values = np.array([np.mean(grouped[h]) for h in hosts], dtype=np.float64)
            stats = paired_inference(host_values, rng, bootstrap)
            overall.append({'reference': reference, 'baseline': method, 'metric': metric,
                            'positive_means_reference_better': True,
                            'n_paired_rows': len(selected), **stats})
        categories = sorted({(data[reference][k].get('attack_category') or
                              data[reference][k].get('attack_group') or 'unknown')
                             for k in attacked_keys})
        for category in categories:
            selected = [k for k in attacked_keys if
                        (data[reference][k].get('attack_category') or
                         data[reference][k].get('attack_group') or 'unknown') == category]
            for metric in ('attacked_ber', 'attacked_nc'):
                direction = METRICS[metric][1]
                deltas = [((parse_float(data[reference][k],metric,reference,k) -
                            parse_float(data[method][k],metric,method,k))
                           if direction == 'higher' else
                           (parse_float(data[method][k],metric,method,k) -
                            parse_float(data[reference][k],metric,reference,k))) for k in selected]
                category_rows.append({'reference':reference, 'baseline':method,
                                      'category':category,'metric':metric,
                                      'reference_advantage':float(np.mean(deltas)),
                                      'n_paired_rows':len(selected),
                                      'inference':'descriptive_only'})
    holm(overall)
    return summary, overall, category_rows


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--method-column', default='auto', choices=('auto','method_id','variant'))
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    if args.bootstrap < 100:
        parser.error('bootstrap must be at least 100')
    data, column, metadata = load_data(args.inputs, args.method_column)
    summary, overall, categories = analyze(data, args.reference,
                                           bootstrap=args.bootstrap, seed=args.seed)
    out=Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for row in summary:
        row['blindness_tier'], row['fidelity_tier'] = metadata.get(row['method'], ('', ''))
    for row in overall:
        row['baseline_blindness_tier'], row['baseline_fidelity_tier'] = metadata.get(row['baseline'], ('', ''))
        row['interpretation_note'] = 'exploratory; embedding strengths not PSNR-matched; compare blindness tiers separately'
    for row in categories:
        row['baseline_blindness_tier'], row['baseline_fidelity_tier'] = metadata.get(row['baseline'], ('', ''))
    write_csv(out/'method_summary.csv', summary)
    write_csv(out/'paired_overall.csv', overall)
    write_csv(out/'paired_attack_categories.csv', categories)
    write_csv(out/'summary_by_extraction_tier.csv', summary)
    manifest={'inputs': args.inputs, 'method_column':column, 'reference':args.reference,
              'methods':sorted(data), 'method_metadata': metadata,
              'bootstrap':args.bootstrap, 'seed':args.seed,
              'unit_of_independence':'host_image',
              'paired_on':KEY_FIELDS,
              'coverage':'identical sets of host/watermark/seed/attack keys required; no failed rows allowed',
              'p_values':'two-sided host-level sign-flip tests, Holm corrected within each metric',
              'confidence_intervals':'95% paired host bootstrap percentile; NOT multiplicity adjusted',
              'inference_limitations':(
                'Eight fixed hosts are not a random population sample. Single seed and two '
                'watermarks per host do not support population-level significance. '
                'Effects are exploratory and conditional on this benchmark.'),
              'blindness_warning':'Blind/key-assisted-blind and semi-blind methods are DIFFERENT extraction assumptions; do not present an undifferentiated leaderboard.',
              'provenance_warning':'Baseline DOI and adaptation status must be read from paper_baseline_audit; not official full-paper reproductions.',
              'psnr_warning':'Embedding strengths are NOT necessarily PSNR matched; do not interpret robustness '
                             'difference as an embedding-efficiency advantage.'}
    (out/'protocol.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Validated {len(data)} methods and {summary[0]["n_hosts"]} hosts. '
          f'Wrote {len(overall)} paired metric comparisons to {out}',flush=True)

if __name__=='__main__':
    main()
