"""Numerical equivalence and per-day S2 OC microbenchmark (no full MC run).

Run: python validate_exact_speed.py --tests 4
Baseline is file-backed exact S2 with identical MILP. Test cases include
unreserved, 1MW, 2MW and 4MW reserve freezes at fixed intervals. Fail closed if
objective differs more than 1 cent per day or 1e-7 relative (whichever larger).
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tests',type=int,default=4)
    args=p.parse_args()
    if args.tests<1 or args.tests>8:p.error('--tests 1..8')
    orig=os.getcwd()
    cap_dir=ROOT/'MFRR&AFRR'
    sys.path.insert(0,str(cap_dir))
    os.chdir(cap_dir)
    try:
        import afrr_mfrr as cap
        cfg=cap.build_cfg()
        df,timestamps=cap.load_forecast_input_data(int(cfg['HORIZON']))
        base=cap.make_free_battery_reservation_profile(timestamps,cfg)
        pmax=float(cfg['BATTERY']['p_max_mw'])
        tests=[('baseline',base),('interval05_q1',cap.apply_symmetric_reserve_freeze(base,5,min(1.0,pmax),cfg)),
               ('interval30_q2',cap.apply_symmetric_reserve_freeze(base,30,min(2.0,pmax),cfg)),
               ('interval70_q4',cap.apply_symmetric_reserve_freeze(base,70,min(4.0,pmax),cfg)),
               ('interval10_q1',cap.apply_symmetric_reserve_freeze(base,10,min(1.0,pmax),cfg)),
               ('interval20_q3',cap.apply_symmetric_reserve_freeze(base,20,min(3.0,pmax),cfg)),
               ('interval50_q1',cap.apply_symmetric_reserve_freeze(base,50,min(1.0,pmax),cfg)),
               ('interval80_q2',cap.apply_symmetric_reserve_freeze(base,80,min(2.0,pmax),cfg))][:args.tests]
        timings=[]
        for name,rs in tests:
            os.environ['BALTIC_OC_METHOD']='exact'
            t=time.perf_counter();original=cap.run_s2_forecast_value(rs);old_sec=time.perf_counter()-t
            os.environ['BALTIC_OC_METHOD']='exact_sparse'
            t=time.perf_counter();fast=cap.run_s2_forecast_value(rs);new_sec=time.perf_counter()-t
            diff=abs(original-fast)
            tol=max(0.01,1e-7*abs(original))
            status='PASS' if diff<=tol else 'FAIL'
            print(f'[{status}] {name:20s} orig={original:,.7f} fast={fast:,.7f} '
                  f'diff={diff:.9f} orig_sec={old_sec:.3f} fast_sec={new_sec:.3f}',flush=True)
            if diff>tol:raise AssertionError(f'Objective mismatch {name}: {original} vs {fast}')
            timings.append((old_sec,new_sec))
        # Ignore first fast call's cache compilation when reporting steady state.
        old=sum(t[0] for t in timings[1:] or timings)
        new=sum(t[1] for t in timings[1:] or timings)
        print(f'PASS: {len(tests)} forecast MILP equivalence tests, '
              f'steady-state wall speedup = {old/new:.2f}x',flush=True)
    finally:
        os.chdir(orig)

if __name__=='__main__':main()
