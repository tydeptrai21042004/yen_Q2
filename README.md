# MSDP — one-seed, 61-condition reproducible experiments

This GitHub-ready research package contains the MSDP implementation, 16 inherited literature baseline implementations, 12 BMP hosts, 2 PNG watermarks, tests, and experiment scripts. **Lenna is not included.** No experiment results are bundled.

## Run all primary studies with ONE seed (recommended)

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_all_experiments.py \
  --profile one_seed \
  --seed 2026 \
  --max-hours 11.25 \
  --strict-trials \
  --output results/MSDP_BASIC_61_ONE_SEED
```

**Use one seed only**: `2026` is the single embedding/evaluation seed for all experiment families. Negative-key security trials and different watermark contents are distinct test inputs, **not additional replicate seeds**.

**Exact benchmark**: 20 types × 3 levels = **60 attacks**, plus one clean/no-attack condition created automatically in every trial = **61 total conditions**. The `basic20x3` suite matches the table in `docs/BASIC61_ONE_SEED.md`. The primary orchestration never uses `publication`, `extended`, simulated real-world attacks, or the separate `run_adversarial.py` suite.

**Under 12 hours**: The recommended `one_seed` profile evaluates 4 tuning hosts/1 watermark for ablation, OFAT sweep, payload and watermark content, and 8 disjoint test hosts/2 watermarks for the fixed baseline comparison. It runs the 5 primary blind baselines, a PSNR-matched comparison, security diagnostics, theory validation, and synthetic numerical verification. All attacks use the same 61-condition protocol. The 11.25-hour cap preserves completed checkpoints and creates a verified `*_PARTIAL.zip` if a longer stage cannot finish. **Completion inside the limit is an objective, not a guarantee**; refer to `completion_status.json`.

To resume after an interrupted or budget-limited run, execute the **exact same command with the same output directory**. A trial journal prevents recomputation of finished ablation, sweep, payload, watermark-content, and baseline trials. Completed stages are skipped. Changing the protocol requires a new output folder.

To run **all 12 hosts and both watermarks in each experiment** (much slower), select `--profile full`. To run all 16 literature baselines (may exceed 12h), add `--all-baselines`. These options change coverage but never add more experiment seeds or image attacks.

### Kaggle (after uploading the modified files to GitHub)

```python
import os, sys, subprocess
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
repo = Path('/kaggle/working/yen_Q2')
subprocess.run(['git','clone','--depth','1','https://github.com/tydeptrai21042004/yen_Q2.git',str(repo)],check=True)
subprocess.run([sys.executable,'-m','pip','install','-r','requirements.txt'],cwd=repo,check=True)
subprocess.run([sys.executable,'-m','pytest','-q'],cwd=repo,check=True)
subprocess.run([sys.executable,'scripts/run_all_experiments.py','--profile','one_seed',
    '--seed','2026','--max-hours','11.25','--strict-trials',
    '--output','results/MSDP_BASIC_61_ONE_SEED'],cwd=repo,check=False)
print('Results and verified ZIP files:',list((repo/'results').glob('*')))
```

See `docs/BASIC61_ONE_SEED.md` for details and exact attack parameters. The repository retains historical attack definitions for compatibility but does not execute them in the default pipeline.
