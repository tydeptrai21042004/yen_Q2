# MSDP — runnable experiment package (no results included)

This ZIP contains **only the inputs and files necessary to run and verify the experiments**: the MSDP algorithm, all 16 literature baseline implementations, attack definitions, experiment runners, configurations, unit tests, 12 host images and two watermarks. It contains **no saved results, plots, logs, marked images, recovered images, or cached outputs**. Lenna is not included.

## Install (Ubuntu, Python 3.10+)

Extract this archive and enter its root folder `MSDP_Watermarking`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Quick smoke run (all experiment families)

```bash
python scripts/run_all_experiments.py --profile quick --strict-trials --output results/research_quick
```

## Standard and full studies

```bash
python scripts/run_all_experiments.py --profile standard --output results/research_standard
python scripts/run_all_experiments.py --profile full --output results/research_full
```

The `full` profile can be very slow. Include the inherited baseline comparison explicitly using `--include-baselines`; some baselines have known capacity or runtime limitations.

## Direct benchmark and other experiments

```bash
# Single-host proposed method with all sanity attacks
python scripts/run_benchmark.py --methods msdp --hosts Girl --suite sanity --save-images --output results/sanity

# Unified attack set for proposed method and all supported baselines
python scripts/run_benchmark.py --methods msdp,all_baselines --hosts all --suite publication --output results/full_baselines

# List supported baselines
python scripts/run_benchmark.py --list-baselines
```

Other experiments (ablation, hyperparameter sweep, security, collusion, adversarial removal, payload, theoretical validation, PSNR matching, attack audit, statistical analysis and plotting) are documented with individual runnable commands in **`docs/EXPERIMENTS.md`**.

The scripts create `results/` and output files *when you execute them*; no output files are shipped. To create figures from your own measurements, install `matplotlib` and run `scripts/plot_results.py` using the generated CSV files. Review all `status` and error columns; a baseline failure must not be treated as successful recovery.

## Contents

- `src/qr64_certified/` — MSDP, baselines, shared utilities and attacks
- `scripts/` — benchmark and extended experiment runners
- `configs/` — algorithm and baseline parameters
- `data/host/` — 12 original 512×512 BMP host images (no Lenna)
- `data/watermark/` — two watermark PNG images
- `tests/` — runnable unit and integration tests
- `requirements.txt`, `pyproject.toml` — Python dependencies
- `docs/EXPERIMENTS.md` — all experimental commands and caveats
- `LICENSE_BASE_CODE` — original license notice
- `PACKAGE_MANIFEST.sha256` — integrity hashes for all non-manifest files
