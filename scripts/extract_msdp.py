#!/usr/bin/env python3
"""Recover watermark from an MSDP-marked RGB image and its extraction key."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.common.io import load_host_rgb,save_image
from qr64_certified.msdp import extract_msdp,key_from_jsonable

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True)
    p.add_argument('--key',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    key=key_from_jsonable(json.loads(Path(args.key).read_text(encoding='utf-8')))
    wm=extract_msdp(load_host_rgb(args.image,size=key.image_shape[0]),key)
    save_image(args.output,wm)
    print('Saved recovered watermark to',args.output)
if __name__=='__main__':main()
