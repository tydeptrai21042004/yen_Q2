#!/usr/bin/env python3
"""Fail-closed audit of the default paper-derived baseline comparison.

A valid DOI establishes a published reference; this audit does NOT certify
line-by-line fidelity to the full article. Check `adaptations` in the report.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.baselines import get_baseline_spec, RELATED_ADDITIONAL_BASELINE_IDS, PRIMARY_BLIND_BASELINE_IDS

def audit():
    registry=json.loads((ROOT/'configs/paper_baseline_provenance.json').read_text())['methods']
    expected=set(PRIMARY_BLIND_BASELINE_IDS)|set(RELATED_ADDITIONAL_BASELINE_IDS)
    if set(registry)!=expected:
        raise ValueError(f'Paper-manifest and default comparison mismatch: missing={expected-set(registry)}, unexpected={set(registry)-expected}')
    rows=[]
    for method_id in sorted(expected):
        data=registry[method_id]
        spec=get_baseline_spec(method_id)
        if not data['doi'].startswith('10.') or not data['reference_url'].startswith('https://doi.org/10.'):
            raise ValueError('Missing peer-reviewed publication DOI for '+method_id)
        if not data['paper_component'] or not data['adaptations']:
            raise ValueError('Missing paper-to-code mapping and adaptations for '+method_id)
        if not (ROOT/data['implementation']).is_file():
            raise ValueError('Implementation file missing for '+method_id)
        if 'control' in method_id or 'dummy' in method_id or 'new_control' in spec.citation:
            raise ValueError('Non-literature control selected: '+method_id)
        if spec.blindness_tier!=data['tier']:
            raise ValueError('Blindness tier mismatch for '+method_id)
        if spec.fidelity_tier in {'controlled_reference_not_literature','major_algorithm_adaptation','conceptual_iwt_svd_qim_adapter'}:
            raise ValueError('Method not eligible for strict default comparison: '+method_id)
        rows.append({'method_id':method_id, 'citation':spec.citation,
                     'year':data['year'], 'doi':data['doi'], 'paper_title':data['title'],
                     'blindness_tier':spec.blindness_tier,'fidelity_tier':spec.fidelity_tier,
                     'paper_component':data['paper_component'],
                     'implementation':data['implementation'],
                     'adaptations':data['adaptations'],
                     'exact_paper_reproduction':False})
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    rows=audit()
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'paper_baseline_audit.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (folder/'paper_baseline_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f'PASS: {len(rows)} paper-derived methods; 0 synthetic controls; '+
          'adaptations disclosed; no claim of exact reproduction',flush=True)

if __name__=='__main__':main()
