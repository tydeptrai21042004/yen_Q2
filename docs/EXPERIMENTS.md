# MSDP experimental research protocol (expanded runnable project)

This package retains the original MSDP method, all 16 inherited literature baseline source implementations, 12 original hosts (NO Lenna), two watermarks, without archived benchmark outputs. The extension adds reproducible experimental families. **The presence of a script is not experimental evidence. Only generated measurements in output directories are observations.**

## 1. Installation, tests, and one-command execution

From `MSDP_Watermarking` (Python 3.10+):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_all_experiments.py --profile quick --output results/research_quick
# For plots: python -m pip install matplotlib
```

`quick` runs all ten experiment families on small, reproducible selections. `standard` runs several hosts/seeds, and `full` runs broad sweeps on the full host set. Full runs can require hours or longer, particularly the nonblind/classical baselines, the adversarial collusion copies, and the SciPy reference optimizer. `--include-baselines` adds the 16 inherited methods and **may produce legitimate capacity failures** (especially DEW) or slow QWT-QSVD runs. `execution_report.json` and child logs expose failures. Do not interpret success of orchestrator processes as zero failed *trials*: inspect every CSV `status` column.

The `common20` suite remains frozen for direct comparisons with previously recorded MSDP results. The output includes a SHA-256 of the exact ordered attack configurations. For genuine JPEG2000 the encoder must support JPEG2000; it is no longer silently replaced by JPEG. Dependencies and codec support vary across machines.

## 2. Experiment matrix

| Study | Script | Reproducible output and limitations |
|---|---|---|
| Original/baseline unified protocol | `run_benchmark.py` | `per_attack.csv`, `summary.csv`, `summary.json`, `attack_protocol.json`; 16 baselines retained; run `--strict` to fail on incomplete trials |
| Minimax ablation | `run_ablation.py` | Full, fixed balanced pattern, randomized pattern, anti-minimax pattern, no channel-mean penalty, no integer repair; no-repair is **diagnostic only** |
| Margin / floor / channel penalty / pattern count | `run_parameter_sweep.py` | One-factor-at-a-time or factorial, raw trials, summaries and complete-case Pareto frontier; choose parameters using validation images, not test images |
| Wrong-key / false positives | `run_security.py` | Correct key positives, host-only, wrong seed, wrong pattern map, unrelated reference, foreign image; `roc.csv`; correlated trials make binomial IID confidence bounds non-justified |
| Adversarial/key-exposed contrast removal | `run_adversarial.py` | Targeted affine boundary attack at multiple budgets; averaging/median of independently marked copies; replacement by a new copy and sequential overwrite; attacker has true pattern map for targeted attack |
| Sparse payload/selection overhead | `run_payload.py` | 1-bit-per-active-block rates, random/low-/high-variance placement; directly usable sparse extraction key; adaptive selection-mask overhead included; encoder timing NOT representative of sparse implementation |
| Watermark balance / structure | `run_watermark_content.py` | Independent random Bernoulli watermarks, checkerboard, stripes, constant marks and supplied logo; report watermark bit balance |
| Theory / numerical optimizer | `run_theory_validation.py` | Signed affine constraint, box clipping, integer repair and optional independent SLSQP objective gap; this is not an analytic optimality proof |
| Numerical/saturation stress | `run_numerical_stress.py` | Flat black/white, midgray, 8-bit RGB saturation, random, gradient, checkerboard. Synthetic feasibility ≠ natural-image robustness |
| PSNR calibration + held-out comparison | `run_matched_comparison.py` | Tune only MSDP normalized margin on CALIBRATION hosts, evaluate both methods on DISJOINT TEST hosts; explicitly mark unmatched results |
| Unified attack manifest and probe | `validate_attack_suite.py` | `attack_manifest.csv`, `attack_protocol.json`, deterministic repeat check and operation/codec failure reports |
| Failure-aware summaries + paired CIs | `analyze_results.py` | Per-category success-only NC and errors, matched-trial differences and host-watermark-seed cluster bootstrap |
| Optional paper figures | `plot_results.py` | PDF charts from **real CSV**, no prefilled/synthetic measurements |

## 3. Minimal individual commands

```bash
python scripts/run_ablation.py --hosts Girl,pepper --seeds 2026,2027,2028 --suite common20 --output results/ablation
python scripts/run_parameter_sweep.py --hosts Girl,pepper --seeds 2026,2027 --suite common20 --output results/sweep
python scripts/run_parameter_sweep.py --mode grid --hosts Girl --margins .01,.02 --floors .1,.25 --penalties 1,2 --pattern-counts 4,8 --suite sanity --output results/grid
python scripts/run_security.py --hosts Girl,pepper --negative-trials 500 --output results/security
python scripts/run_adversarial.py --hosts Girl,pepper --fractions .01,.05,.1,.25 --copies 2,4,8 --output results/adversarial
python scripts/run_payload.py --hosts Girl --fractions .125,.25,.5,1 --placements random,low_variance,high_variance --suite common20 --output results/payload
python scripts/run_watermark_content.py --hosts Girl,pepper --suite common20 --output results/wm_content
python scripts/run_theory_validation.py --hosts Girl,pepper --blocks-per-host 4 --independent-solver --output results/theory
python scripts/run_numerical_stress.py --output results/numerical
python scripts/validate_attack_suite.py --suites publication --probe --fail-on-error --output results/attack_validation
python scripts/run_matched_comparison.py --baseline dct_dm_qim_chen2001_blind --calibration-hosts Girl,airplane --test-hosts pepper,house --margins .005,.01,.015,.02,.03,.05 --suite common20 --output results/matched
python scripts/run_benchmark.py --methods msdp,primary_blind --hosts Girl,pepper --suite common20 --strict --output results/primary_baselines
python scripts/run_benchmark.py --methods msdp,all_baselines --hosts all --suite publication --output results/full_comparison
```

## 4. Analyze and plot

```bash
python scripts/analyze_results.py --input results/primary_baselines/per_attack.csv --reference msdp --bootstrap 1000 --output results/summary_analysis
python -m pip install matplotlib
python scripts/plot_results.py --kind ablation --input results/ablation/summary.csv --output results/fig_ablation.pdf
python scripts/plot_results.py --kind sweep --input results/sweep/summary.csv --output results/fig_sweep.pdf
python scripts/plot_results.py --kind security --input results/security/roc.csv --output results/fig_roc.pdf
python scripts/plot_results.py --kind payload --input results/payload/payload_trials.csv --output results/fig_payload.pdf
python scripts/plot_results.py --kind attack --input results/summary_analysis/by_attack_category.csv --output results/fig_attacks.pdf
```

## 5. Rules for defensible comparisons

1. Freeze host, watermark, attack settings, key disclosure, and payload across compared methods. A script's shared `attack_id` is necessary but **not sufficient** for comparison fairness: some baselines need the original host and some are adapted rather than exact paper-native reproductions.
2. A failed embedding or attack must count as **failure**, not as success or random NC. Raw rows are retained with `status=error`. The benchmark summary's mean is conditional on success and **must** be paired with the failure count. The `--strict` flag turns any failure into a nonzero exit code.
3. Keep calibration/validation/test host sets disjoint. The default 12 hosts are a modest sample. Use more independently sourced, provenance-tracked hosts only after verifying sharing rights; this ZIP already contains all its necessary input images and never downloads replacements.
4. The normalized-determinant condition implies an affine contrast constraint for nonnegative channel means. SLSQP experiments check implementation accuracy, not novelty of an arbitrary second-order-cone solver. The message-independent minimax selector is exact for the unbounded surrogate, not universally for box-constrained objectives.
5. The detector is **key-assisted blind**: it requires a cover-dependent, message-independent pattern map; the NumPy pseudorandom mask is not cryptographic encryption. Wrong-key tests are not cryptographic proofs. The detection threshold is prespecified (default 0.15); ROC is descriptive, and trials sharing keys/hosts are statistically dependent.
6. A targeted attack that uses the true map explicitly models **key exposure**. The radius `|u-v| sqrt(N)/(2||W||)` is the unbounded Euclidean distance to the affine sign boundary for a given pattern, not a certified radius against arbitrary geometric operations, integer constraints or side-channel adversaries.
7. `run_payload.py` uses independent-block compositing: it applies full embedding and retains the selected blocks, whose output is mathematically the same as directly embedding only those active blocks. Extractor reads a serialized compact map without the host; however, do **not** quote its full-embedding wall time as the cost of a sparse-native encoder.
8. Do not call simulated `print_scan` or `screen_capture` a real-world physical attack; no physical capture images are included.
9. JPEG2000 is **strict** unless an attack explicitly sets `fallback_to_jpeg=True`; if that option is ever used, call it a JPEG substitution, not a JPEG2000 result.
10. Security ROC/false-positive experiments require many independently sourced host-key pairs to support generalizable claims. Repeated wrong seeds on a single cover alone do not prove low operational FPR.

## 6. Study-specific statistical design

For the paper, use five families of figures/tables: (i) clean and attack-category quality with failure counts; (ii) minimax/fixed/random ablations at matched payload, (iii) margin–distortion and key-map overhead sensitivity, (iv) detection ROC and wrong-key/mapping distributions, and (v) theoretical contrast-radius and key-exposed removal curves. Plot different baseline blindness tiers in separate panels or annotate them prominently. Use host-cluster confidence intervals, not independent attack-row intervals.

## 7. Backward compatibility and validation status

The default `MSDPConfig()` preserves the original core embedding parameters and its standard output. New `pattern_selector` and `integer_repair` settings only activate for controlled ablation. The original tests remain present; additional extension tests protect message-independent maps, sparse-key recovery, parameter validation and attack manifests.

No precomputed results are included in this distribution. All output CSV, JSON, logs, plots and images are generated by running the scripts. Rerun tests after installation.
