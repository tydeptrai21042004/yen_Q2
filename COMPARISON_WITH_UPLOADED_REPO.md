# Comparison with yen_Q2-main(3).zip

The latest uploaded repository is SHA-256 identical file-by-file to yen_Q2-main(2).zip.
This means it does not contain the previously proposed paper-only / paired-statistics changes.
This ZIP contains all changed and added files needed to apply that revision. It is NOT
an entire replacement repository; the current original source/data remain in place.

## Changes included
- Eight paper-derived baseline methods in default comparison: five existing primary
  paper-based methods plus Guo et al. (2017), Mao et al. (2024), Kumar et al. (2024).
- No synthetic Hadamard-QIM controls registered (the old Revision-1 controls are removed).
- New paper-baseline provenance/audit with explicit adaptation and blindness categories.
- Expanded one-seed orchestrator and compatible benchmark selection.
- Host-level paired bootstrap CIs, exact sign-flip p-values, Holm adjusted p-values;
  ablation comparisons and attack-category differences.
- Original four tuning/eight test hosts, two watermarks, 60 attacks + clean, and seed 2026.
- Full repository PACKAGE_MANIFEST.sha256 updated for reproducibility.
- No independent host-image generalization experiment included.

## Fidelity notes
Paper-linked means the implementation follows identifiable paper methods, with a
common image/payload adaptation. This is NOT proof of a line-by-line or original-
parameter reproduction. Guo FA optimization is not used in common evaluation;
Mao and Kumar require semi-blind information. See provenance JSON.

## How to apply
1. Unzip this archive at the ROOT of `yen_Q2`, overwriting existing files.
2. `python scripts/apply_paper_only_patch.py` (removes only obsolete Revision-1 control).
3. `python -m pip install -r requirements.txt -e .`
4. `python -m pytest -q`
5. `python scripts/audit_paper_baselines.py --output results/paper_baseline_audit`
6. `python scripts/run_all_experiments.py --profile one_seed --seed 2026 --max-hours 11.1 --strict-trials --output results/MSDP_PAPER_BASELINES_STATISTICS_ONE_SEED`

Use a new results directory, not the older primary comparison output folder.

## Validation on the newly uploaded repository
- Exact base-to-previous-base match: 113 of 113 regular files identical.
- After merging this patch: 21/21 pytest cases passed using `PYTHONPATH=src`.
- Paper provenance audit: 8 paper-derived methods, zero synthetic controls.
- Whole 14-stage run is NOT claimed to have executed from this freshly merged ZIP.
- Local editable pip install was blocked by internet/DNS access; pytest ran from source.
