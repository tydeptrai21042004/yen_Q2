# Revision 2 — paper-derived baselines ONLY

This patch **replaces** the prior Revision 1 synthetic Hadamard-QIM controls
with published-paper-derived methods already implemented in the repository.
It includes no results and makes no claim of exact native-paper reproduction.
Each default comparison method has a verified published-paper DOI, a source
file, implementation stages, and an explicit adaptations description.

## Apply to `yen_Q2` after cloning or pulling current GitHub repo

1. Extract the contents of this ZIP **into repository root** and OVERWRITE
   existing files while preserving the relative directories.
2. Run `python scripts/apply_paper_only_patch.py` to remove the old orphan
   `src/qr64_certified/baselines/implementations/spatial/hadamard_qim.py`
   if a previous patch created that file. No other files are deleted.
3. Run `python -m pip install -r requirements.txt -e .`.
4. Run `python -m pytest -q`.
5. Run `python scripts/audit_paper_baselines.py --output results/paper_baseline_audit`.
6. Run `python scripts/run_all_experiments.py --profile one_seed --seed 2026
   --max-hours 11.1 --strict-trials --output results/MSDP_PAPER_BASELINES_STATISTICS_ONE_SEED`.

The default 14-stage one-seed run keeps four tuning/eight test images,
two binary marks, 60 attacks + clean and the original statistical
analysis. It has six primary methods (MSDP + 5 paper baselines) and three
extra paper-derived methods: Guo et al. (2017) DWT-QR, Mao et al. (2024)
CA-QIM, and Kumar et al. (2024) DWT-WHT-SVD. Guo is key-assisted blind;
Mao and Kumar are semi-blind and **should not be presented as having equal
access assumptions**. Their code includes adaptations, documented in
`configs/paper_baseline_provenance.json`.

## Run only added methods, then paired statistics against existing results

```bash
python scripts/run_benchmark.py --methods related_additions \
  --hosts airplane,athens,baboon,butfly,manhatan,milkdrop,safari,tiffany \
  --watermarks all --seed 2026 --suite basic20x3 --strict \
  --skip-attacked-quality --output results/paper_related_baselines

python scripts/analyze_paired_statistics.py \
  --inputs results/primary_baselines/per_attack.csv \
           results/paper_related_baselines/per_attack.csv \
  --reference msdp --bootstrap 10000 --seed 451 \
  --output results/paired_statistics
```

The existing primary result file must use the exact same protocol, host
and mark IDs. No false results are manufactured to fill gaps. The added
3-method/1-host 60+clean smoke test passed; the full multi-host default
experiment has not been executed to completion within this patch creation.

More detail: `docs/REVISION_BASELINES_STATISTICS.md`.
