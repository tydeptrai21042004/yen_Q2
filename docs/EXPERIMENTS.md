# MSDP experiments — current protocol

Use the unified, one-seed 61-condition benchmark documented in **[BASIC61_ONE_SEED.md](BASIC61_ONE_SEED.md)**.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_all_experiments.py --profile one_seed --seed 2026 --max-hours 11.25 --strict-trials --output results/MSDP_BASIC_61_ONE_SEED
```

The default command runs attack validation, 17-configuration hyperparameter sweep, six controlled ablations, payload study, watermark-content study, MSDP plus five primary blind baselines, PSNR-matched comparison, security diagnostics, theory checks and synthetic numerical verification. The independent adversarial-attack runner is **not** run. The existing source files implementing extra attack suites are kept only for compatibility and are not used by the unified benchmark.

For an individual experiment, always use `--suite basic20x3` and `--seeds 2026` (or `--seed 2026` where supported). A clean condition is automatically included in addition to the 60 attacks. Do not use the older `publication`, `extended`, or `common20` protocols when preparing results for the 20×3 paper table.

The run produces `execution_report.json`, `completion_status.json`, per-stage logs and result directories, plus a verified final ZIP. Trials are checkpointed and resume with the same output path. Trials reporting errors are not silently counted as successes. `--strict-trials` flags such errors. Use `--profile full` or `--all-baselines` only with sufficient time budget; these do not add experiment seeds but may take much longer than the default.
