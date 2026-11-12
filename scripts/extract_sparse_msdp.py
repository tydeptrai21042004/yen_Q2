#!/usr/bin/env python3
"""Standalone extractor for the compact sparse MSDP experimental format."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.sparse import extract_sparse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True,help='RGB watermarked PNG')
    p.add_argument('--key',required=True,help='example_sparse_key.json')
    p.add_argument('--output',required=True,help='Extracted active bits .npy file (0 or 255)')
    a=p.parse_args()
    image=np.asarray(Image.open(a.image).convert('RGB'),dtype=np.uint8)
    bits,indices=extract_sparse(image,json.loads(Path(a.key).read_text()))
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    np.save(dest,bits)
    np.save(dest.with_name(dest.stem+'_block_indices.npy'),indices)
    print('Extracted',len(bits),'active bits; indices in',dest.with_name(dest.stem+'_block_indices.npy'))
if __name__=='__main__':main()
