#!/usr/bin/env python3
"""One-seed MSDP experiment orchestrator using the exact 60+clean protocol.

Default ``one_seed`` profile uses disjoint validation/test hosts and five
primary blind baselines. It never runs separate adversarial image attacks.
All robustness experiments use the same fixed 60 attacks + 1 clean case.
"""
from __future__ import annotations
import argparse
import csv
import os
import platform
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('quick','standard','full','one_seed'),default='one_seed')
    p.add_argument('--seed',type=int,default=2026,help='ONE experiment seed for every runner')
    p.add_argument('--output',default='results/MSDP_BASIC_61_ONE_SEED')
    p.add_argument('--max-hours',type=float,default=11.25,
                   help='Stop before session timeout; unfinished trials remain resumable')
    p.add_argument('--no-result-zip',action='store_true',help='Disable atomic ZIP export after execution')
    p.add_argument('--all-baselines','--include-baselines',dest='all_baselines',action='store_true',
                   help='All 16 literature baselines instead of five primary blind baselines; slow')
    p.add_argument('--skip-theory-solver',action='store_true')
    p.add_argument('--strict-trials',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=(root/a.output).resolve() if not Path(a.output).is_absolute() else Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    def version_or_missing(package):
        try:
            return version(package)
        except PackageNotFoundError:
            return 'not installed'
    try:
        memory_gib=round(os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')/1024**3,2)
    except (AttributeError,ValueError,OSError):
        memory_gib=None
    hardware={'python':sys.version,'platform':platform.platform(),
              'processor':platform.processor(),'cpu_count':os.cpu_count(),
              'total_memory_gib':memory_gib,
              'numpy':version_or_missing('numpy'),'scipy':version_or_missing('scipy'),
              'pillow':version_or_missing('Pillow'),
              'numba':version_or_missing('numba'),
              'opencv_python_headless':version_or_missing('opencv-python-headless')}
    (out/'hardware_software.json').write_text(json.dumps(hardware,indent=2)+'\n',encoding='utf-8')
    print('HARDWARE / SOFTWARE:',json.dumps(hardware),flush=True)

    protocol='basic20x3'  # Frozen paper protocol: 60 attacked + one clean row.
    seed=str(a.seed)
    quick=a.profile=='quick'
    full=a.profile=='full'
    if quick:
        tuning_hosts='Girl';test_hosts='airplane'
        tuning_wm='wm';test_wm='wm'
    elif full:
        tuning_hosts='all';test_hosts='all'
        tuning_wm='all';test_wm='all'
    else:
        # No validation/test image overlap for the recommended one-seed run.
        tuning_hosts='Girl,pepper,house,sailboat'
        test_hosts='airplane,athens,baboon,butfly,manhatan,milkdrop,safari,tiffany'
        tuning_wm='wm';test_wm='all'
    neg=12 if quick else (60 if full else 30)
    tasks=[]
    def task(name,script,*args):
        tasks.append((name,[sys.executable,str(root/'scripts'/script),*map(str,args),
                            '--output',str(out/name)]))

    task('attack_protocol','validate_attack_suite.py','--suites',protocol,'--probe','--fail-on-error')
    task('parameter_sweep','run_parameter_sweep.py','--hosts',tuning_hosts,
         '--watermarks',tuning_wm,'--seeds',seed,'--suite',protocol,'--skip-attacked-quality',
         *(['--margins','.015,.02','--floors','.25','--penalties','2',
            '--pattern-counts','8'] if quick else []))
    task('ablation','run_ablation.py','--hosts',tuning_hosts,'--watermarks',tuning_wm,
         '--seeds',seed,'--suite',protocol,'--skip-attacked-quality',
         *(['--variants','full,fixed_pattern,random_pattern,no_mean_penalty'] if quick else []))
    task('payload','run_payload.py','--hosts',tuning_hosts,'--watermarks',tuning_wm,
         '--seeds',seed,'--fractions','.25,.5' if quick else '.125,.25,.5,1',
         '--placements','random' if quick else 'random,low_variance,high_variance',
         '--suite',protocol)
    task('watermark_content','run_watermark_content.py','--hosts',tuning_hosts,
         '--seeds',seed,'--modes','bernoulli,logo' if quick else
         'bernoulli,checkerboard,stripes,constant_zero,constant_one,logo',
         '--probabilities','.5' if quick else '.1,.25,.5,.75,.9',
         '--suite',protocol,'--skip-attacked-quality')
    task('primary_baselines','run_benchmark.py','--methods',
         'msdp,all_baselines' if a.all_baselines else 'msdp,primary_blind',
         '--hosts',test_hosts,'--watermarks',test_wm,'--seed',seed,'--suite',protocol)
    task('matched_comparison','run_matched_comparison.py',
         '--calibration-hosts','Girl','--test-hosts',
         'airplane' if not full else 'athens',
         '--watermarks',tuning_wm,'--seed',seed,'--suite',protocol,
         '--margins','.01,.015' if quick else '.005,.01,.015,.02,.03,.05')
    task('security','run_security.py','--hosts',tuning_hosts,'--watermarks',tuning_wm,
         '--seeds',seed,'--negative-trials',str(neg))
    task('theory','run_theory_validation.py','--hosts',tuning_hosts,
         '--blocks-per-host','1' if quick else '4','--seed',seed,
         *([] if a.skip_theory_solver else ['--independent-solver']))
    task('numerical_stress','run_numerical_stress.py','--image-size','64' if quick else '128',
         '--margins','.015' if quick else '.005,.015,.03','--seed',seed)
    # The adversarial attack runner is intentionally NOT part of this protocol.

    run_spec={'profile':a.profile,'seed':a.seed,'suite':protocol,'conditions_per_trial':61,
              'tuning_hosts':tuning_hosts,'test_hosts':test_hosts,
              'tuning_watermarks':tuning_wm,'test_watermarks':test_wm,
              'all_baselines':a.all_baselines,'skip_theory_solver':a.skip_theory_solver,
              'tasks':{name:cmd for name,cmd in tasks}}
    spec_path=out/'run_protocol.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding='utf-8'))!=run_spec:
            raise SystemExit('Existing output has a different protocol. Choose a new --output folder.')
    else:
        spec_path.write_text(json.dumps(run_spec,indent=2)+'\n',encoding='utf-8')
    report_path=out/'execution_report.json'
    old=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else []
    report={entry['name']:entry for entry in old}
    start=time.monotonic()
    deadline=start+max(0.,a.max_hours)*3600
    for index,(name,command) in enumerate(tasks,1):
        if name in report and report[name].get('exit_code')==0:
            print(f'[{index}/{len(tasks)}] {name}: already completed; skip',flush=True)
            continue
        remaining=deadline-time.monotonic()
        if remaining<=10:
            print('Session time budget reached. Saved completed trials; rerun same command to resume.',flush=True)
            break
        print(f'[{index}/{len(tasks)}] {name}, remaining budget={remaining/3600:.2f} h',flush=True)
        started=time.monotonic();log=out/(name+'_stdout.log')
        timeout=False
        with log.open('a',encoding='utf-8') as fh:
            fh.write(f'\n===== START / RESUME {name} =====\n')
            fh.flush()
            try:
                proc=subprocess.run(command,cwd=root,stdout=fh,stderr=subprocess.STDOUT,
                                    timeout=max(1.,remaining-5.))
                returncode=proc.returncode
            except subprocess.TimeoutExpired:
                timeout=True
                returncode=124
                fh.write('\nExceeded session budget; any complete trial checkpoints retained.\n')
        failures={}
        if (out/name).is_dir():
            for report_csv in (out/name).glob('*.csv'):
                try:
                    with report_csv.open(encoding='utf-8',newline='') as fh:
                        failed=sum(r.get('status')=='error' for r in csv.DictReader(fh))
                    if failed:failures[report_csv.name]=failed
                except Exception as exc:
                    failures[report_csv.name+'_parse_error']=repr(exc)
        report[name]={'name':name,'exit_code':returncode,'timed_out':timeout,
                      'trial_failure_counts':failures,
                      'seconds_this_attempt':time.monotonic()-started,
                      'command':command,'log':str(log.relative_to(out))}
        report_path.write_text(json.dumps([report[k] for k,_ in tasks if k in report],indent=2)+'\n',encoding='utf-8')
        print(f'  exit={returncode}, saved log={log.name}',flush=True)
        if returncode!=0:
            print('Stopped: this stage failed or timed out; check its log. Retry same command to resume.',flush=True)
            break
        if a.strict_trials and failures:
            print('Stopped: failed trial rows detected; check output before publication.',flush=True)
            break
    complete=all(name in report and report[name].get('exit_code')==0 for name,_ in tasks)
    failures=sum(sum(v for v in r.get('trial_failure_counts',{}).values() if isinstance(v,int)) for r in report.values())
    completion={'complete':complete,'profile':a.profile,'seed':a.seed,'suite':protocol,
                'total_tasks':len(tasks),'completed_tasks':sum(r.get('exit_code')==0 for r in report.values()),
                'failed_trial_rows_recorded':failures,'remaining_tasks':[name for name,_ in tasks
                                           if name not in report or report[name].get('exit_code')!=0],
                'note':'If incomplete, do not claim all experiment results are complete. Rerun with same output to resume.'}
    (out/'completion_status.json').write_text(json.dumps(completion,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(completion,indent=2),flush=True)
    if not a.no_result_zip:
        suffix='COMPLETE' if complete and not failures else 'PARTIAL'
        final_zip=out.parent/(out.name+'_'+suffix+'.zip')
        temporary_zip=out.parent/(out.name+'.zip.tmp')
        try:
            with zipfile.ZipFile(temporary_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,
                                 allowZip64=True) as archive:
                for child in sorted(out.rglob('*')):
                    if child.is_file():
                        archive.write(child,arcname=str(child.relative_to(out.parent)))
            with zipfile.ZipFile(temporary_zip,'r') as archive:
                damaged=archive.testzip()
                if damaged is not None:
                    raise RuntimeError(f'ZIP integrity check failed on {damaged}')
            temporary_zip.replace(final_zip)
            print('Verified results archive:',final_zip,flush=True)
        finally:
            if temporary_zip.exists():
                temporary_zip.unlink()
    if not complete or (a.strict_trials and failures):
        raise SystemExit(1)


if __name__=='__main__':main()
