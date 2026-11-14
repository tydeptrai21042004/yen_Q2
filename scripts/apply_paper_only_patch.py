#!/usr/bin/env python3
"""Safely remove the previous synthetic baseline's orphan source file.

Run after extracting this patch to the repository root. Never touches images,
experiment output files or previously published implementations.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
obsolete=(ROOT/'src/qr64_certified/baselines/implementations/spatial/hadamard_qim.py',)
for path in obsolete:
    if path.exists():
        path.unlink()
        print('Removed non-paper control:', path.relative_to(ROOT))
    else:
        print('Already clean:', path.relative_to(ROOT))
print('Paper-based baseline configuration is ready.')
