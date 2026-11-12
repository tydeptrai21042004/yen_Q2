#!/usr/bin/env python3
"""Numerically validate affine residual, margin and continuous convex optimum.
Optional independent SciPy SLSQP reference; measured gap is a NUMERICAL check.
"""
import argparse
import math
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qr64_certified.msdp import (W,W2,N,_means,_affine_slack,_normalized_residual,
    _codebook,_continuous_project,_round_and_repair,_blockize)
from qr64_certified.common.io import load_host_rgb
from qr64_certified.research import hosts,csv_write,json_write

def objective(z,x,lam):
    d=z-x;m=d.mean(axis=0)
    return 0.5*float(np.sum(d*d))+lam*N/2.*float(m@m)

def independent_reference(x,p,s,kappa,floor,lam):
    a=(2./N)*(s*p[:,None]-kappa)*W[None,:]
    x0=x.ravel()
    def f(z):
        d=z.reshape(N,3)-x
        m=d.mean(axis=0)
        return .5*np.sum(d*d)+lam*N/2.*np.sum(m*m)
    def grad(z):
        d=z.reshape(N,3)-x
        return (d+lam*d.mean(axis=0)).ravel()
    cons={'type':'ineq','fun':lambda z: float(np.dot(a.ravel(),z)-floor),
          'jac':lambda z:a.ravel().copy()}
    result=minimize(f,x0,jac=grad,method='SLSQP',bounds=[(0,255)]*(N*3),
                    constraints=[cons],options={'maxiter':300,'ftol':1e-7})
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hosts',default='Girl')
    p.add_argument('--blocks-per-host',type=int,default=2)
    p.add_argument('--independent-solver',action='store_true')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--output',required=True)
    a=p.parse_args();rng=np.random.default_rng(a.seed);rows=[]
    mu=.015;kappa=mu/math.sqrt(2-mu**2); floor=.25;lam=2.
    patterns=_codebook(8)
    for h in hosts(a.hosts):
        original,_=_blockize(load_host_rgb(h))
        for idx in rng.choice(len(original),size=min(a.blocks_per_host,len(original)),replace=False):
            x=original[idx].astype(np.float64)
            for sign in (-1.,1.):
                ptn=patterns[int(rng.integers(len(patterns)))]
                y,active=_continuous_project(x,ptn,sign,kappa,floor,lam,W,W2)
                out,ok=_round_and_repair(y,x,ptn,sign,kappa,floor,lam,W)
                u,v=_means(out,ptn,W)
                row={'host':h.stem,'block':int(idx),'sign':sign,'box_active':bool(active),
                     'continuous_objective':objective(y,x,lam),
                     'continuous_slack':float(_affine_slack(y,ptn,sign,kappa,floor,W)),
                     'rounded_repair_success':bool(ok),
                     'integer_slack':float(_affine_slack(out,ptn,sign,kappa,floor,W)),
                     'integer_normalized':float(sign*_normalized_residual(out,ptn,W)),
                     'integer_objective':objective(out.astype(np.float64),x,lam),
                     'radius_to_sign_boundary_L2':float(abs(u-v)*math.sqrt(N)/(2*math.sqrt(W2)))}
                if a.independent_solver:
                    independent=independent_reference(x,ptn,sign,kappa,floor,lam)
                    row.update({'reference_solver_success':bool(independent.success),
                                'reference_objective':float(independent.fun),
                                'absolute_objective_gap':float(abs(row['continuous_objective']-independent.fun)),
                                'reference_message':independent.message})
                rows.append(row)
    csv_write(rows,Path(a.output)/'theory_validation.csv')
    json_write({'constraints_checked':len(rows),'all_integer_constraints_pass':all(r['integer_slack']>=-1e-8 for r in rows),
                'solver_reference_used':a.independent_solver},Path(a.output)/'theory_summary.json')

if __name__=='__main__':main()
