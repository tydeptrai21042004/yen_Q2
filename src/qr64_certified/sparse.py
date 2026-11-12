"""Sparse payload placement experiment and independently usable compact key.

Encoder uses the full independent-block MSDP embedder, then retains only chosen
blocks. Because MSDP has no inter-block coupling, these blocks are the same as
embedding only the selected bits. Embedding timing measured this way is NOT a
sparse-encoder runtime and must not be reported as such.
"""
from __future__ import annotations
import base64
import numpy as np
from .msdp import (_blockize,_unblockize,_pack,_unpack,_codebook,W,N,
                   MSDPConfig,embed_msdp)

def positions_from_seed(count:int,k:int,seed:int)->np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(count,size=k,replace=False))

def select_blocks(host:np.ndarray,k:int,selection:str,seed:int)->np.ndarray:
    blocks,_=_blockize(host)
    if selection=='random':return positions_from_seed(len(blocks),k,seed)
    variation=np.var(blocks.astype(float),axis=(1,2))
    order=np.argsort(variation,kind='stable')
    return np.sort(order[:k] if selection=='low_variance' else order[-k:])

def encode_sparse(host:np.ndarray, watermark:np.ndarray, *, fraction:float,
                  placement:str, seed:int, selection_seed:int=314159,
                  cfg:MSDPConfig|None=None):
    if not 0<fraction<=1:raise ValueError('fraction must be in (0,1]')
    if placement not in ('random','low_variance','high_variance'):
        raise ValueError('Unknown placement')
    marked,key,meta=embed_msdp(host,watermark,seed=seed,config=cfg)
    original_blocks,shape=_blockize(host)
    marked_blocks,_=_blockize(marked)
    n=len(original_blocks)
    k=max(1,min(n,int(round(n*fraction))))
    pos=select_blocks(host,k,placement,selection_seed)
    out=original_blocks.copy();out[pos]=marked_blocks[pos]
    pattern_indices=_unpack(key.pattern_codes,key.pattern_count,n)[pos]
    codes=_pack(pattern_indices,key.pattern_count)
    # Explicit full-size selection bitmask is required for content-adaptive
    # placements; independent-seed random placement requires only the seed.
    active_bytes=None
    if placement!='random':
        flags=np.zeros(n,dtype=np.uint8);flags[pos]=1
        active_bytes=np.packbits(flags).tobytes()
    compact={'method':'msdp_sparse_experiment_v1','shape':list(shape),
             'watermark_shape':list(watermark.shape),'pattern_count':key.pattern_count,
             'pattern_codes_base64':base64.b64encode(codes).decode(),
             'mask_seed':seed,'selection_seed':selection_seed,'selection_mode':placement,
             'selected_blocks':k,'active_mask_base64':base64.b64encode(active_bytes).decode() if active_bytes else None}
    meta.update({'selected_blocks':k,'placement':placement,'sparse_pattern_map_bytes':len(codes),
                 'selection_mask_bytes':len(active_bytes) if active_bytes else 0,
                 'nominal_payload_bpp':k/(shape[0]*shape[1])})
    return _unblockize(out,shape),compact,pos,meta

def extract_sparse(image:np.ndarray,compact:dict)->tuple[np.ndarray,np.ndarray]:
    if compact['method']!='msdp_sparse_experiment_v1':raise ValueError('Wrong key type')
    blocks,shape=_blockize(image)
    if list(shape)!=compact['shape']:raise ValueError('Shape mismatch')
    n=len(blocks);k=compact['selected_blocks']
    if compact['selection_mode']=='random':
        pos=positions_from_seed(n,k,compact['selection_seed'])
    else:
        flags=np.unpackbits(np.frombuffer(base64.b64decode(compact['active_mask_base64']),
                                          dtype=np.uint8),count=n)
        pos=np.flatnonzero(flags)
    if len(pos)!=k:raise ValueError('Sparse map length mismatch')
    indices=_unpack(base64.b64decode(compact['pattern_codes_base64']),compact['pattern_count'],k)
    patterns=_codebook(compact['pattern_count'])[indices]
    d=np.einsum('nmc,nm,c->n',blocks[pos].astype(np.float64),patterns,W)*(2./N)
    mask=np.random.default_rng(compact['mask_seed']).integers(0,2,size=n,dtype=np.int8)[pos]
    return np.where(np.where(d>=0,1,-1)*(1-2*mask)>0,255,0).astype(np.uint8),pos
