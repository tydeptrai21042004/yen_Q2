#!/usr/bin/env python3
"""One-command orchestrator for ALL experiment families (no network needed).

quick: lightweight verification. standard: multi-host/seed science protocol.
full: extensive, potentially very slow; baseline all is explicit opt-in.
Each child writes its own outputs, failures are logged and surfaced.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('quick','standard','full'),default='quick')
    p.add_argument('--output',default='results/research_run')
    p.add_argument('--include-baselines',action='store_true',
                   help='Full baseline comparison including known unsupported/slow baselines')
    p.add_argument('--skip-theory-solver',action='store_true')
    p.add_argument('--strict-trials',action='store_true',
                   help='Exit nonzero if any child reports failed trials')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=(root/a.output).resolve() if not Path(a.output).is_absolute() else Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    quick=a.profile=='quick'; full=a.profile=='full'
    hosts='Girl' if quick else ('all' if full else 'Girl,pepper,house,sailboat')
    wm='wm' if quick else 'all'
    seeds='2026' if quick else ('2026,2027,2028,2029,2030' if full else '2026,2027,2028')
    attack='sanity' if quick else ('publication' if full else 'common20')
    neg=12 if quick else (500 if full else 100)
    tasks=[]
    def task(name,script,*args):
        tasks.append((name,[sys.executable,str(root/'scripts'/script),*map(str,args),
                            '--output',str(out/name)]))
    task('attack_protocol','validate_attack_suite.py','--suites','common20' if quick else 'publication','--probe')
    task('ablation','run_ablation.py','--hosts',hosts,'--watermarks',wm,'--seeds',seeds,
         '--suite',attack,*(['--variants','full,fixed_pattern,random_pattern,no_mean_penalty'] if quick else []))
    task('parameter_sweep','run_parameter_sweep.py','--hosts',hosts,'--watermarks',wm,
         '--seeds',seeds,'--suite',attack,
         *(['--margins','.015,.02','--floors','.25','--penalties','2','--pattern-counts','8'] if quick else []))
    task('security','run_security.py','--hosts',hosts,'--watermarks',wm,'--seeds',seeds,
         '--negative-trials',neg)
    task('adversarial','run_adversarial.py','--hosts',hosts,'--watermarks',wm,
         '--seeds',seeds,'--fractions','.01' if quick else '.01,.05,.10,.25',
         '--copies','2' if quick else '2,4,8','--selection','cheapest' if quick else 'both')
    task('payload','run_payload.py','--hosts',hosts,'--watermarks',wm,'--seeds',seeds,
         '--fractions','.25,.5' if quick else '.125,.25,.5,1',
         '--placements','random' if quick else 'random,low_variance,high_variance',
         '--suite','clean' if quick else attack)
    task('watermark_content','run_watermark_content.py','--hosts',hosts,'--seeds',seeds,
         '--modes','bernoulli,logo' if quick else 'bernoulli,checkerboard,stripes,constant_zero,constant_one,logo',
         '--probabilities','.5' if quick else '.1,.25,.5,.75,.9',
         '--suite','clean' if quick else attack)
    task('theory','run_theory_validation.py','--hosts',hosts,
         '--blocks-per-host','1' if quick else '4',
         *([] if a.skip_theory_solver else ['--independent-solver']))
    task('numerical_stress','run_numerical_stress.py','--image-size','64' if quick else '128',
         '--margins','.015' if quick else '.005,.015,.03')
    task('matched_comparison','run_matched_comparison.py',
         '--calibration-hosts','Girl', '--test-hosts','pepper',
         '--margins','.01,.015' if quick else '.005,.01,.015,.02,.03,.05',
         '--suite','clean' if quick else attack)
    if a.include_baselines:
        task('all_baselines','run_benchmark.py','--methods','msdp,all_baselines',
             '--hosts',hosts,'--watermarks',wm,'--suite',attack)
    report=[]
    for index,(name,command) in enumerate(tasks,1):
        print(f'[{index}/{len(tasks)}] {name}',flush=True)
        started=time.perf_counter();log=out/(name+'_stdout.log')
        with log.open('w',encoding='utf-8') as fh:
            proc=subprocess.run(command,cwd=root,stdout=fh,stderr=subprocess.STDOUT)
        failure_counts={}
        for report_csv in (out/name).glob('*.csv'):
            try:
                with report_csv.open(encoding='utf-8',newline='') as fh:
                    failed=sum(r.get('status')=='error' for r in csv.DictReader(fh))
                if failed:failure_counts[report_csv.name]=failed
            except Exception as exc:
                failure_counts[report_csv.name+'_parse_error']=repr(exc)
        status={'name':name,'exit_code':proc.returncode,'trial_failure_counts':failure_counts,
                'seconds':time.perf_counter()-started,
                'command':command,'log':str(log.relative_to(out))}
        report.append(status)
        (out/'execution_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(f'  exit={proc.returncode} log={log}',flush=True)
    if any(r['exit_code']!=0 for r in report):
        raise SystemExit('One or more experiments failed; inspect execution_report.json')
    failed=sum(sum(v for v in r['trial_failure_counts'].values() if isinstance(v,int)) for r in report)
    print(f'All experiment runners completed; failed trial rows detected: {failed}.')
    if a.strict_trials and failed:raise SystemExit(1)

if __name__=='__main__':main()
