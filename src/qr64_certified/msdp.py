"""Minimax Spatial Determinant Projection (MSDP).

Selected method only; direct-spatial, no DCT/QR factorization in computation.
The carrier is exactly the *normalized determinant residual* of
    A_p(X) = [[u_p(X), 1], [v_p(X), 1]]:
    R_p(X) = det(A_p) / sqrt(u_p**2 + v_p**2).

For a message sign s, the continuous embedder solves a strictly convex
quadratic projection with a normalized-determinant guarantee and pixel bounds.
Since u_p,v_p are nonnegative RGB partition means, the apparent conic constraint
is EXACTLY the affine inequality
    s(u-v) >= kappa(u+v),  kappa=mu/sqrt(2-mu**2).
We add a conservative >0 determinant floor delta via
    s(u-v) >= kappa(u+v) + delta.
Thus the actual optimization is a box-constrained quadratic projection onto a
halfspace, not a novel general second-order-cone theorem. No mean equality is
imposed: a quadratic channel-mean penalty retains intensity dependence.

Message-independent patterns minimize the *unconstrained* worst-bit projection
cost; pixel-box constraints can change the actual best pattern. Extractor uses
sign(det(A_p)) plus a message-independent, host-dependent bit-packed pattern map.
Continuous QP solutions are optimal within the specified feasible set, while
integer rounding/repair is feasible but not claimed integer-optimal.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit
from scipy.linalg import hadamard

BLOCK = 8
N = BLOCK * BLOCK
W = np.array((0.299, 0.587, 0.114), dtype=np.float64)
W2 = float(np.dot(W, W))


class MSDPInfeasibleError(ValueError):
    """Numerical solver could not certify a valid integer encoding."""


@dataclass(frozen=True)
class MSDPConfig:
    normalized_margin: float = 0.015
    determinant_floor: float = 0.25
    mean_penalty: float = 2.0
    pattern_count: int = 8
    block_size: int = BLOCK
    # Research-only controlled ablations; defaults preserve the original method.
    pattern_selector: str = 'minimax'  # minimax, fixed, random, max_abs
    integer_repair: bool = True

    def validate(self) -> None:
        if self.block_size != BLOCK:
            raise ValueError('Current MSDP requires 8x8 blocks')
        if not (0.0 < self.normalized_margin < 1.0):
            raise ValueError('normalized_margin must be in (0,1)')
        if not (0.0 < self.determinant_floor < 10.0):
            raise ValueError('determinant_floor must be in (0,10)')
        if not (0.0 <= self.mean_penalty <= 100.0):
            raise ValueError('mean_penalty must be in [0,100]')
        if not (2 <= self.pattern_count <= 63):
            raise ValueError('pattern_count must be in [2,63]')
        if self.pattern_selector not in ('minimax', 'fixed', 'random', 'max_abs'):
            raise ValueError('Unknown pattern_selector')
        if not isinstance(self.integer_repair, bool):
            raise ValueError('integer_repair must be a boolean')


@dataclass(frozen=True)
class MSDPKey:
    image_shape: tuple[int, int]
    watermark_shape: tuple[int, int]
    pattern_count: int
    pattern_codes: bytes
    seed: int
    normalized_margin: float
    determinant_floor: float
    mean_penalty: float
    method: str = 'msdp_v1'


def _blockize(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image HxWx3')
    h, w = a.shape[:2]
    if h % BLOCK or w % BLOCK:
        raise ValueError('Image dimensions must be divisible by 8')
    blocks = a.reshape(h//BLOCK, BLOCK, w//BLOCK, BLOCK, 3).transpose(0,2,1,3,4)
    return blocks.reshape(-1,N,3).copy(), (h,w)


def _unblockize(blocks: np.ndarray, shape: tuple[int,int]) -> np.ndarray:
    h,w = shape
    return blocks.reshape(h//BLOCK,w//BLOCK,BLOCK,BLOCK,3).transpose(0,2,1,3,4).reshape(h,w,3)


def _codebook(count: int) -> np.ndarray:
    return hadamard(N)[1:1+count].astype(np.float64)


def _pack(indices: np.ndarray, count: int) -> bytes:
    nbits = (count-1).bit_length()
    shifts = np.arange(nbits-1,-1,-1,dtype=np.int64)
    return np.packbits(((indices[:,None]>>shifts)&1).astype(np.uint8).ravel()).tobytes()


def _unpack(codes: bytes, count: int, n: int) -> np.ndarray:
    nbits = (count-1).bit_length()
    if len(codes) != (n*nbits+7)//8:
        raise ValueError('Pattern-map length does not match image')
    b = np.unpackbits(np.frombuffer(codes,dtype=np.uint8),count=n*nbits).reshape(n,nbits)
    i = b.astype(np.int64)@(1<<np.arange(nbits-1,-1,-1))
    if (i>=count).any():
        raise ValueError('Invalid selected pattern')
    return i


@njit(cache=True)
def _means(x, p, weights):
    u,v = 0.0,0.0
    for j in range(N):
        luma=0.0
        for c in range(3):
            luma+=weights[c]*float(x[j,c])
        if p[j]>0.0:
            u+=luma
        else:
            v+=luma
    return (2.0/N)*u,(2.0/N)*v


@njit(cache=True)
def _affine_slack(x, p, s, kappa, floor, weights):
    u,v = _means(x,p,weights)
    return s*(u-v)-kappa*(u+v)-floor


@njit(cache=True)
def _normalized_residual(x, p, weights):
    u,v=_means(x,p,weights)
    d=math.sqrt(u*u+v*v)
    return (u-v)/d if d>0 else 0.0


@njit(cache=True)
def _project_unbounded(x,p,s,kappa,floor,lam,weights,norm2):
    """Exact unconstrained continuous minimizer; returned iff box-feasible."""
    u,v=_means(x,p,weights)
    deficit=floor+kappa*(u+v)-s*(u-v)
    if deficit<=0.0:
        return x.copy(),True
    denom=4.0/N*norm2*(1.0+kappa*kappa/(1.0+lam))
    multiplier=deficit/denom
    result=np.empty((N,3),dtype=np.float64)
    feasible=True
    for j in range(N):
        for c in range(3):
            direction=2.0/N*(s*p[j]-kappa/(1.0+lam))*weights[c]
            value=x[j,c]+multiplier*direction
            result[j,c]=value
            if value < -1e-10 or value>255.0+1e-10:
                feasible=False
    return result,feasible


@njit(cache=True)
def _fixed_dual_projection(x,p,s,kappa,alpha,lam,weights):
    """Unique minimizer of Q(Y)-alpha * [s(u-v)-kappa(u+v)] over RGB box.

    A scalar shift b_c solves b_c=-lam*(mean(Y_c)-mean(X_c)); bisection
    uses monotonicity of b+lam*(mean(clip(...))-mean(X)).
    """
    y=np.empty((N,3),dtype=np.float64)
    for c in range(3):
        original_mean=0.0
        for j in range(N):
            original_mean+=x[j,c]/N
        lo=-lam*255.0-1.0
        hi=lam*255.0+1.0
        for _ in range(34):
            shift=(lo+hi)*0.5
            summ=0.0
            for j in range(N):
                a=2.0/N*(s*p[j]-kappa)*weights[c]
                z=x[j,c]+alpha*a+shift
                if z<0.0:z=0.0
                if z>255.0:z=255.0
                summ+=z/N
            fun=shift+lam*(summ-original_mean)
            if fun>0.0:hi=shift
            else:lo=shift
        shift=(lo+hi)*0.5
        for j in range(N):
            a=2.0/N*(s*p[j]-kappa)*weights[c]
            z=x[j,c]+alpha*a+shift
            y[j,c]=max(0.0,min(255.0,z))
    return y


@njit(cache=True)
def _continuous_project(x,p,s,kappa,floor,lam,weights,norm2):
    """Exact continuous convex QP via scalar-dual monotone root (tolerance)."""
    unconstrained,valid=_project_unbounded(x,p,s,kappa,floor,lam,weights,norm2)
    if valid:
        return unconstrained,False
    low=0.0
    high=1.0
    upper=_fixed_dual_projection(x,p,s,kappa,high,lam,weights)
    for _ in range(55):
        if _affine_slack(upper,p,s,kappa,floor,weights)>=-1e-8:
            break
        high*=2.0
        upper=_fixed_dual_projection(x,p,s,kappa,high,lam,weights)
    if _affine_slack(upper,p,s,kappa,floor,weights)<-1e-7:
        raise ValueError('Cone/box constraint appears numerically infeasible')
    for _ in range(36):
        mid=0.5*(low+high)
        y=_fixed_dual_projection(x,p,s,kappa,mid,lam,weights)
        if _affine_slack(y,p,s,kappa,floor,weights)>=-1e-8:
            upper=y
            high=mid
        else:
            low=mid
    return upper,True


@njit(cache=True)
def _round_and_repair(y,x,p,s,kappa,floor,lam,weights):
    """Bounded 8-bit quantizer plus smallest incremental-cost greedy repairs.

    Integer global optimality is NOT claimed. Correctness is checked using
    the exact affine implication of the normalized determinant constraint.
    """
    out=np.empty((N,3),dtype=np.uint8)
    for j in range(N):
        for c in range(3):
            z=math.floor(y[j,c]+0.5)
            out[j,c]=max(0,min(255,int(z)))
    mdelta=np.zeros(3,dtype=np.float64)
    for c in range(3):
        for j in range(N):
            mdelta[c]+=(float(out[j,c])-x[j,c])/N
    for _ in range(2500):
        if _affine_slack(out,p,s,kappa,floor,weights)>=-1e-9:
            return out,True
        best_ratio=1e100
        bestj=-1
        bestc=-1
        bestdir=0
        for j in range(N):
            for c in range(3):
                a=2.0/N*(s*p[j]-kappa)*weights[c]
                direction=1 if a>0.0 else -1
                z=int(out[j,c])
                if (direction>0 and z>=255) or (direction<0 and z<=0):
                    continue
                # Exact objective increase Q(y+unit)-Q(y), including mean penalty.
                diff=z-x[j,c]
                gain=direction*diff+0.5+lam*direction*mdelta[c]+lam/(2.0*N)
                score=gain/abs(a)
                if score<best_ratio:
                    best_ratio=score
                    bestj=j
                    bestc=c
                    bestdir=direction
        if bestj<0:
            return out,False
        out[bestj,bestc]=int(out[bestj,bestc])+bestdir
        mdelta[bestc]+=bestdir/N
    return out,False


def embed_msdp(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    config: MSDPConfig | None = None,
) -> tuple[np.ndarray, MSDPKey, dict[str, Any]]:
    cfg = config or MSDPConfig()
    cfg.validate()
    original,shape=_blockize(host_rgb)
    wm=np.asarray(watermark_binary)
    expected=(shape[0]//BLOCK,shape[1]//BLOCK)
    if wm.shape!=expected:
        raise ValueError(f'Expected watermark {expected}, got {wm.shape}')
    if not np.all((wm==0)|(wm==1)|(wm==255)):
        raise ValueError('Watermark must be binary')
    patterns=_codebook(cfg.pattern_count)
    # Exact unconstrained minimax choice: maximize over both signs, not the
    # actual watermark bit. Note: active box constraints can change ordering.
    d=np.einsum('nmc,pm,c->np',original.astype(np.float64),patterns,W)*(2.0/N)
    if cfg.pattern_selector == 'minimax':
        selected=np.argmin(np.abs(d),axis=1).astype(np.int64)
    elif cfg.pattern_selector == 'fixed':
        selected=np.zeros(len(original), dtype=np.int64)
    elif cfg.pattern_selector == 'random':
        # Independent stream, avoiding coupling to bit-masking RNG state.
        selected=np.random.default_rng(np.random.SeedSequence([int(seed), 7919])).integers(
            cfg.pattern_count, size=len(original), dtype=np.int64)
    else:  # max_abs: deliberately anti-minimax, diagnostic only.
        selected=np.argmax(np.abs(d),axis=1).astype(np.int64)
    codes=_pack(selected,cfg.pattern_count)
    raw_sign=np.where(wm.reshape(-1)>0,1,-1).astype(np.int8)
    mask=np.random.default_rng(int(seed)).integers(0,2,size=len(original),dtype=np.int8)
    signs=(raw_sign*(1-2*mask)).astype(np.int8)
    kappa=cfg.normalized_margin/math.sqrt(2.0-cfg.normalized_margin**2)
    source=original.astype(np.float64)
    output=np.empty_like(original)
    box_active=0
    minimum_margin=float('inf')
    minimum_normalized=float('inf')
    minimum_determinant=float('inf')
    failed_margin_blocks=0
    total_squared_distortion=0.0
    for i in range(len(source)):
        x=source[i]
        p=patterns[selected[i]]
        sign=float(signs[i])
        y,active=_continuous_project(x,p,sign,kappa,cfg.determinant_floor,cfg.mean_penalty,W,W2)
        if cfg.integer_repair:
            out,ok=_round_and_repair(y,x,p,sign,kappa,cfg.determinant_floor,cfg.mean_penalty,W)
            if not ok:
                raise MSDPInfeasibleError(f'Integer margin repair failed for block {i}')
        else:
            # Diagnostic only: feasible continuous solution, uncorrected 8-bit rounding.
            out=np.clip(np.floor(y+0.5),0,255).astype(np.uint8)
        if active:
            box_active+=1
        output[i]=out
        slack=float(_affine_slack(out,p,sign,kappa,cfg.determinant_floor,W))
        failed_margin_blocks += int(slack < -1e-8)
        total_squared_distortion += float(np.sum((out.astype(np.float64)-x)**2))
        residual=float(_normalized_residual(out,p,W))
        u,v=_means(out,p,W)
        minimum_margin=min(minimum_margin,slack)
        minimum_normalized=min(minimum_normalized,sign*residual)
        minimum_determinant=min(minimum_determinant,sign*(u-v))
    key=MSDPKey(
        image_shape=shape,watermark_shape=wm.shape,
        pattern_count=cfg.pattern_count,pattern_codes=codes,seed=int(seed),
        normalized_margin=cfg.normalized_margin,
        determinant_floor=cfg.determinant_floor,mean_penalty=cfg.mean_penalty,
    )
    meta={
        'method_id':'msdp', 'payload_bits':len(original),
        'pattern_selector':cfg.pattern_selector, 'integer_repair':cfg.integer_repair,
        'certified_integer_constraint_violations':int(failed_margin_blocks),
        'certified_integer_constraint_fraction':float(1.0-failed_margin_blocks/len(original)),
        'total_squared_distortion':total_squared_distortion,
        'pattern_counts':np.bincount(selected, minlength=cfg.pattern_count).tolist(),
        'pattern_map_bytes':len(codes),
        'pattern_map_bits_per_block':(cfg.pattern_count-1).bit_length(),
        'message_independent_pattern_map':True,
        'key_assisted_blind':True,
        'active_box_projection_blocks':box_active,
        'minimum_signed_affine_slack':minimum_margin,
        'minimum_attained_normalized_residual':minimum_normalized,
        'minimum_attained_signed_determinant':minimum_determinant,
        'changed_rgb_samples':int(np.count_nonzero(output!=original)),
        'integer_global_optimality_claimed':False,
        'constrained_pattern_selector_is_exact_minimax':False,
        'continuous_projection_global_optimality':True,
    }
    return _unblockize(output,shape),key,meta


def extract_msdp(image_rgb: np.ndarray,key: MSDPKey) -> np.ndarray:
    if key.method!='msdp_v1':
        raise ValueError('Unrecognized MSDP key')
    blocks,shape=_blockize(image_rgb)
    if shape!=key.image_shape:
        raise ValueError('Image geometry changed; no geometric registration included')
    indices=_unpack(key.pattern_codes,key.pattern_count,len(blocks))
    patterns=_codebook(key.pattern_count)[indices]
    # det(A_p)=u-v: the sign of normalized determinant equals sign of d.
    d=np.einsum('nmc,nm,c->n',blocks.astype(np.float64),patterns,W)*(2.0/N)
    signs=np.where(d>=0.0,1,-1)
    mask=np.random.default_rng(int(key.seed)).integers(0,2,size=len(blocks),dtype=np.int8)
    wm_sign=signs*(1-2*mask)
    return np.where(wm_sign>0,255,0).astype(np.uint8).reshape(key.watermark_shape)


def verify_msdp(image_rgb:np.ndarray,key:MSDPKey,expected_watermark:np.ndarray,*,threshold:float=0.15)->dict[str,float|bool|int]:
    if not 0.0<threshold<=1.0:
        raise ValueError('threshold must be in (0,1]')
    reference=np.asarray(expected_watermark)
    if reference.shape!=key.watermark_shape:
        raise ValueError('Wrong expected watermark shape')
    recovered=extract_msdp(image_rgb,key)
    agreement=float(np.mean((recovered>127)==(reference>0)))
    corr=2*agreement-1
    n=reference.size
    return {'accepted':bool(corr>=threshold),'correlation':corr,
            'bit_accuracy':agreement,'payload_bits':int(n),
            'conditional_null_upper_bound_at_threshold':float(math.exp(-0.5*n*threshold*threshold)),
            'conditional_null_upper_bound_at_observed_correlation':float(math.exp(-0.5*n*max(0.0,corr)**2))}


def key_to_jsonable(key:MSDPKey)->dict[str,Any]:
    return {'method':key.method,'image_shape':list(key.image_shape),
            'watermark_shape':list(key.watermark_shape),'pattern_count':key.pattern_count,
            'pattern_codes_base64':base64.b64encode(key.pattern_codes).decode('ascii'),
            'seed':key.seed,'normalized_margin':key.normalized_margin,
            'determinant_floor':key.determinant_floor,'mean_penalty':key.mean_penalty}


def key_from_jsonable(obj:dict[str,Any])->MSDPKey:
    if obj.get('method')!='msdp_v1':
        raise ValueError('Wrong key version')
    return MSDPKey(image_shape=tuple(int(x) for x in obj['image_shape']),
       watermark_shape=tuple(int(x) for x in obj['watermark_shape']),
       pattern_count=int(obj['pattern_count']),
       pattern_codes=base64.b64decode(obj['pattern_codes_base64'],validate=True),
       seed=int(obj['seed']),normalized_margin=float(obj['normalized_margin']),
       determinant_floor=float(obj['determinant_floor']),mean_penalty=float(obj['mean_penalty']))


__all__=['MSDPConfig','MSDPKey','MSDPInfeasibleError','embed_msdp','extract_msdp',
         'verify_msdp','key_to_jsonable','key_from_jsonable']
