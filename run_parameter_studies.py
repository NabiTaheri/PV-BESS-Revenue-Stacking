#!/usr/bin/env python3
"""Run additional *reoptimized* battery-size and degradation variants on May 2026.

This is a separate, substantial set of exact optimization runs; it does not
rescale the reference battery cashflows. Uses 20 independent first-level seeds
per variant by default and 1000 conditional activation draws per seed.
The 5 MW / 10 MWh, 25 EUR/MWh reference is the parent run_may_exact.py run.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parent
STUDIES=[('bess_2p5mw_5mwh',2.5,5.,25.),
         ('bess_5mw_5mwh',5.,5.,25.),
         ('bess_5mw_20mwh',5.,20.,25.),
         ('degradation_15_eur_mwh',5.,10.,15.),
         ('degradation_40_eur_mwh',5.,10.,40.)]


def summarize():
    rows=[]
    for name,p,e,c in STUDIES:
        path=ROOT/'studies'/name
        s=path/'run_summary.json'
        if not s.is_file():continue
        runs=json.loads(s.read_text(encoding='utf-8'))
        for case in ('reference','stress'):
            metrics=path/f'paper_tables_{case}'/'paper_metrics.json'
            if not metrics.is_file():raise RuntimeError(f'Missing expected results: {metrics}')
            m=json.loads(metrics.read_text(encoding='utf-8'))
            rows.append({'configuration':name,'BESS_P_MW':p,'BESS_E_MWh':e,'E_over_P_hours':e/p,
                         'degradation_eur_per_mwh':c,'activation_regime':case,
                         'completed_base_scenarios':runs['n_base_final'],
                         'samples_per_base':runs['n_activation_per_base'],
                         'paired_gap_mean_eur':m['mean_s3_minus_s2_eur'],
                         'prob_S3_outperforms_S2_percent':m['prob_s3_ge_s2_percent'],
                         'dispatch_reoptimized':True})
    if rows:
        out=ROOT/'studies';out.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out/'size_and_degradation_reoptimized_summary.csv',index=False)
    return len(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--base-scenarios',type=int,default=20)
    p.add_argument('--activation-samples',type=int,default=1000)
    p.add_argument('--only',choices=[x[0] for x in STUDIES])
    p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args()
    if a.summarize_only:
        print('Completed variant summary rows:',summarize());return
    if not 1<=a.base_scenarios<=113:raise SystemExit('base-scenarios must be 1..113')
    if a.activation_samples<20:raise SystemExit('activation-samples must be >=20')
    cases=[x for x in STUDIES if not a.only or a.only==x[0]]
    for name,power,energy,coef in cases:
        env=os.environ.copy()
        env.update(BALTIC_BESS_P_MW=str(power),BALTIC_BESS_E_MWH=str(energy),
                   BALTIC_CYCLE_COST_EUR_PER_MWH=str(coef),
                   BALTIC_RESULTS_SUBDIR=str(Path('studies')/name))
        cmd=[sys.executable,str(ROOT/'run_may_exact.py'),'--workers',str(a.workers),
             '--max-base',str(a.base_scenarios),'--min-base',str(a.base_scenarios),
             '--batch-size',str(a.base_scenarios),'--skip-convergence',
             '--interim-activation',str(a.activation_samples),
             '--final-activation',str(a.activation_samples)]
        print(f'\n[REOPTIMIZE] {name}: BESS {power} MW/{energy} MWh, coefficient {coef} EUR/MWh',flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
        print('[COMPLETE]',name,flush=True)
        summarize()
    print('[ALL STUDIES COMPLETE] summary:', ROOT/'studies/size_and_degradation_reoptimized_summary.csv',flush=True)

if __name__=='__main__':main()
