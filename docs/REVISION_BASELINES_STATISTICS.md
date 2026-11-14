# Revision 2 — published-paper baselines and paired statistics

## No synthetic/dummy baseline entries

The two previous fixed-pattern and minimax-pattern Hadamard-QIM controls have
been **removed** from the baseline registry, executable methods, and benchmark
configuration. The old `spatial/hadamard_qim.py` file must be deleted if the
previous patch was applied; run `python scripts/apply_paper_only_patch.py`
after extracting the new ZIP. MSDP ablations (`fixed_pattern`, `random_pattern`,
`anti_minimax`, `no_mean_penalty`, `no_integer_repair`) remain **ablations**,
not separate published-paper baselines.

This revision uses eight peer-reviewed paper-derived methods in comparisons:

| Group | Method | DOI | Extraction assumption | Implementation alignment |
|---|---|---|---|---|
| Primary | Chen & Wornell 2001, DCT-DM-QIM | 10.1109/18.923725 | blind | Published QIM/dither modulation with DCT image wrapper |
| Primary | Chen & Wornell 2001, DCT-STDM-QIM | 10.1109/18.923725 | blind | Paper-inspired spread-transform/dither realization; not native full reproduction |
| Primary | Malvar & Florêncio 2003, DCT-ISS | 10.1109/TSP.2003.809385 | blind | Published ISS modulation with DCT image wrapper |
| Primary | Nha et al. 2023, Hessenberg | 10.1007/s10489-023-04903-y | blind | Published decomposition principle with common payload adapter |
| Primary | Zhang et al. 2022, QWT-QSVD | 10.1016/j.jfranklin.2021.11.027 | blind | Published decomposition scheme, adapted blind key and payload |
| Added | Guo et al. 2017, DWT-QR | 10.1049/iet-ipr.2016.0515 | key-assisted blind | Paper-guided DWT/QR; common mode uses FIXED strength instead of Firefly optimization |
| Added | Mao et al. 2024, CA-QIM | 10.1109/TIFS.2023.3342612 | semi-blind | Paper A2 lattice and canonical labeling; DCT image wrapper |
| Added | Kumar et al. 2024, DWT-WHT-SVD | 10.1007/s11042-023-16059-5 | semi-blind | Paper-guided YCbCr-DWT-WHT-SVD; saved host components |

For exact citation, method-to-source mapping, and deviations see
`configs/paper_baseline_provenance.json` and output
`paper_baseline_audit/paper_baseline_audit.csv`. These are **paper-derived
research implementations** and **not certified exact reproductions** of full
original settings. The Guo (2017) code includes a Firefly strength search,
but the default common benchmark does not run it. The two semi-blind additions
require different side information than a blind detector; keep their results
in separate interpretation strata.

No new independent host-image generalization experiment was added. The frozen
four tuning hosts, eight held-out hosts, two watermarks, seed 2026 and
`basic20x3` 60-attacks-plus-one-clean protocol are unchanged.

## Statistical comparison

`scripts/analyze_paired_statistics.py` pairs identical
`(host_id, watermark_id, seed, attack_id)` and rejects missing/failed rows,
protocol mismatches and differing payload sizes. It reports paired overall
advantages in attacked BER, attacked NC and clean PSNR; exact host-level
sign-flip p-values, 10,000 host-cluster bootstrap replicates, 95% CIs, Cohen
dz and Holm multiplicity correction within metric.

Outputs:

- `paper_baseline_audit/paper_baseline_audit.csv` — DOI/source/implementation/adaptation audit.
- `primary_baselines/per_attack.csv` — MSDP and five original blind methods.
- `related_baselines/per_attack.csv` — the three additional published methods.
- `paired_statistics/method_summary.csv`, `paired_overall.csv`, `paired_attack_categories.csv`, `summary_by_extraction_tier.csv` — descriptive and paired results with extraction-tier labels.
- `ablation_statistics/paired_overall.csv` — original six-variant ablation inference, not external baselines.

**Interpretation:** The eight selected hosts are fixed, not an independent
random image sample; single-seed p-values are exploratory. The 60 conditions
and two watermarks under one host are clustered, not treated as independent
n=960. The strengths are NOT PSNR matched. Do not claim statistically
significant superiority, identical paper-native settings, or identical blind
access for semi-blind methods. No numerical results are included in the patch.

## Apply and run

Extract files at the root of your GitHub repository, then:

```bash
python scripts/apply_paper_only_patch.py
python -m pip install -r requirements.txt -e .
python -m pytest -q
python scripts/audit_paper_baselines.py --output results/paper_baseline_audit
python scripts/run_all_experiments.py --profile one_seed --seed 2026 \
  --max-hours 11.1 --strict-trials \
  --output results/MSDP_PAPER_BASELINES_STATISTICS_ONE_SEED
```

Use a **new output directory**. This orchestration now has 14 stages:
original 10 stages, a paper-DOI audit, one three-paper additional method
stage, and two paired-statistics stages. If a session stops at its time limit,
rerun the same command to resume. A complete result needs all 14 stages,
all required paired keys, and zero failed trial rows.

DO NOT rerun with the old Revision 1 output directory: its trial fingerprints
and method roster differ. The patch contains code, tests, metadata,
documentation, and no host or watermark image files.
