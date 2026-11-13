# One-seed 61-condition paper benchmark

All primary robustness results use **exactly 60 basic attacks** and **one clean condition** per embedded host/watermark/method. The clean condition is added by each evaluator; it is **NOT** an entry in the `basic20x3` registry. Clean results are reported separately from means over attacked NC/BER.

| Category | Attack | Mild | Moderate | Strong |
|---|---|---|---|---|
| Compression | JPEG | quality=90 | quality=70 | quality=50 |
| Compression | JPEG2000 | quality_layer=3 | quality_layer=7 | quality_layer=10 |
| Compression | Bit depth | 7 bits | 6 bits | 5 bits |
| Compression | Color quantization | 128 colors | 64 colors | 32 colors |
| Noise | Gaussian, variance on [0,1] | 0.001 | 0.003 | 0.005 |
| Noise | Salt and pepper | amount=.001 | amount=.005 | amount=.01 |
| Noise | Speckle | variance=.001 | variance=.005 | variance=.01 |
| Noise | Poisson | peak=255 | peak=128 | peak=64 |
| Filtering | Median | size=3 | size=5 | size=7 |
| Filtering | Average | size=3 | size=5 | size=7 |
| Filtering | Gaussian blur | radius=.5 | radius=1.0 | radius=1.5 |
| Filtering | Sharpen | factor=1.25 | factor=1.5 | factor=2.0 |
| Geometry | Direct rotation | 5° | 10° | 15° |
| Geometry | Resize and restore | factor=.9 | factor=.75 | factor=.5 |
| Geometry | Center crop, zero-filled original canvas | keep=.95 | keep=.90 | keep=.80 |
| Geometry | Translation | (2,2) px | (4,4) px | (8,8) px |
| Photometric | Brightness | factor=.95 | factor=.90 | factor=.80 |
| Photometric | Contrast | factor=.95 | factor=.90 | factor=.80 |
| Photometric | Gamma | 1.05 | 1.20 | 1.50 |
| Photometric | Saturation | factor=.80 | factor=.50 | factor=.20 |

The implementation uses 60 unique attack IDs and asserts the count. Rotation is **direct** (no inverse alignment); resized images are restored to original dimensions; the crop retains a centered square portion by linear width/height fraction and sets everything else to 0, without enlarging the crop. Noise generation uses deterministic attack-local RNG seed=123, fixed across comparison methods. The experiment seed=2026 controls embedding and the other relevant method RNGs.

**JPEG2000 naming:** The Pillow/OpenJPEG implementation passes each number as `quality_layers=[3.0]`, `[7.0]` or `[10.0]`. These are **numeric JPEG2000 quality-layer parameters**, *not a demand for 3, 7 or 10 encoded layers*. To maintain precise terminology in the manuscript, replace “layer 3/7/10” with “JPEG2000 quality-layer parameter 3/7/10”, or adopt and document a separate encoded-layer-count implementation. A missing JPEG2000 codec is reported as failure; JPEG substitution is prohibited.

## One-seed coverage

- Validation/tuning hosts: `Girl`, `pepper`, `house`, `sailboat` (one watermark: `wm.png`).
- Held-out test hosts: `airplane`, `athens`, `baboon`, `butfly`, `manhatan`, `milkdrop`, `safari`, `tiffany` (both supplied watermark images).
- Single experiment seed: `2026`. The protocol uses no 3- or 5-seed sweep.
- Retain all 6 ablation variants, 17 OFAT sweep configurations, 12 payload settings, and 10 generated watermark-content configurations.
- Evaluate MSDP plus 5 primary blind literature baselines on held-out hosts. Opt-in `--all-baselines` enables all 16 and can take longer.
- Additional non-image-attack experiments: wrong-key security diagnostics, PSNR calibration/matched comparison, theory checks, numerical feasibility tests.
- The separate adversarial image-attack runner is excluded. The primary `run_all_experiments.py` does not call non-basic image attacks.

### Estimated evaluation counts

| Family | Embedded trials | Rows (61 each) |
|---|---:|---:|
| Sweep | 17 × 4 = 68 | 4,148 |
| Ablation | 6 × 4 = 24 | 1,464 |
| Payload | 12 × 4 = 48 | 2,928 |
| Watermark content | 10 × 4 = 40 | 2,440 |
| MSDP + 5 primary baselines | 6 × 8 × 2 = 96 | 5,856 |
| **Total of above** | **276** | **16,836** |

The matched comparison adds two methods × one held-out host × one watermark × 61 = 122 more rows. Security, theoretical validation and synthetic numerical studies do not use the 61-image-attack protocol because they are independent test types; they do not execute additional image attacks. Clean is counted once in each of the families above.

### Runtime optimizations

- **ONE seed** (`--seed 2026`), not three or five.
- 4 hosts for parameter selection and 8 disjoint hosts for final baseline comparisons.
- Clean PSNR/SSIM and attacked NC/BER retained for ablation, sweep, and watermark-content studies; **attacked** PSNR/SSIM skipped only there, reducing expensive metric calculations. Final baseline comparisons still compute full attacked metrics.
- Append-only, fsync'd complete-trial journals (`trials.jsonl`) prevent repeated cumulative CSV writes and allow resume. Summary CSVs are written at completed stage boundaries; incomplete stages are identifiable by journal and `completion_status.json`.
- An 11.25-hour execution budget avoids launching stages beyond the deadline; completed stage outputs and journal entries remain available. The archive is written to a temporary file, validated via `ZipFile.testzip()`, then atomically renamed to `..._COMPLETE.zip` or `..._PARTIAL.zip`.
- The **12-hour target is not guaranteed**, especially if a baseline, independent solver, codec, or Kaggle hardware is slower than expected. Use `completion_status.json` to distinguish complete vs partial.

### Run

```bash
python scripts/validate_attack_suite.py --suites basic20x3 --probe --fail-on-error --output results/attack_probe
python scripts/run_all_experiments.py --profile one_seed --seed 2026 --max-hours 11.25 --strict-trials --output results/MSDP_BASIC_61_ONE_SEED
```

Re-running the exact same command after timeout safely skips completed stages/trials. To avoid reporting single-seed experiments as repeated stochastic replications, describe the observations as results for a fixed seed and do not claim multi-seed variance or significance.
