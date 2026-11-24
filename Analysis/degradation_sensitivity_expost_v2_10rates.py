#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only, fixed-schedule degradation sensitivity for the Estonia PV+BESS study.

Uses the SAME first-level market decisions, second-level activation samples,
recovery and activation-energy cashflows. Re-prices both Stage-1 scheduled
battery degradation (S2 and S3) and Stage-2 activation/recovery degradation.
No stochastic simulation or market optimisation is rerun.

S2(c) = S2(25) + (25-c)/25 * D_S2_stage1(25)
S3(c) = S3(25) + (25-c)/25 * [D_S3_stage1(25) + D_S3_activation(25)]

The result is conditional ex-post economic sensitivity under FIXED dispatch,
not re-optimised strategies at alternative degradation coefficients.

Example pilot:
 python degradation_sensitivity_expost_v1.py --root D:/BalticMay2026Final \
   --selected-ids-file D:/Baltic_SOC_Audit/may120_selected_base_runs.csv \
   --out D:/Baltic_SOC_Audit/degradation_pilot_10rates --base-count 2 --samples 20

Full:
 python degradation_sensitivity_expost_v1.py --root D:/BalticMay2026Final \
   --selected-ids-file D:/Baltic_SOC_Audit/may120_selected_base_runs.csv \
   --out D:/Baltic_SOC_Audit/degradation_full_10rates --samples 1000
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_RATE = 25.0
RATES = tuple(float(x) for x in np.linspace(12.5, 50.0, 10))
assert BASE_RATE in RATES, "Baseline degradation rate must be included"
COLS = [
    'base_run', 'sample_id', 's2_total_eur', 's3_gross_no_activation_eur',
    's3_activation_adjusted_net_eur', 'net_recovery_cost_eur',
    'degradation_eur', 'throughput_mwh', 'activation_energy_cashflow_eur',
]
STAGE_1_DIRS = {
    's2': 'results_S2_DA_MFRR_energy',
    's3': 'results_S3_reserve_competition_full_stack',
}


def selected_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f'Cannot find selected IDs file: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError('Selected IDs file is empty')
    header = [v.strip() for v in rows[0]]
    col = header.index('base_run') if 'base_run' in header else 0
    data_rows = rows[1:] if 'base_run' in header else rows
    vals = [row[col].strip() for row in data_rows if len(row) > col and row[col].strip()]
    if not vals or len(set(vals)) != len(vals):
        raise ValueError('Selected IDs list empty or contains duplicates')
    return vals


def stage1_totals(root: Path, name: str, strat: str) -> tuple[float, float]:
    f = root/'final_results'/name/'results'/STAGE_1_DIRS[strat]/'monthly_chain_summary.csv'
    if not f.is_file():
        raise FileNotFoundError(f'Missing first-stage summary: {f}')
    df = pd.read_csv(f, usecols=['battery_degradation_eur', 'total_revenue_market_actual_eur'])
    if len(df) < 1:
        raise ValueError(f'Empty first-stage summary: {f}')
    vals = df.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(vals.to_numpy()).all():
        raise ValueError(f'Invalid first-stage revenue/degradation: {f}')
    degradation = float(vals['battery_degradation_eur'].sum())
    revenue = float(vals['total_revenue_market_actual_eur'].sum())
    if degradation < -1e-8:
        raise ValueError(f'Negative Stage-1 degradation cost in: {f}')
    return revenue, degradation


def mean_ci95(x: np.ndarray) -> float:
    if len(x) < 2:
        return float('nan')
    try:
        from scipy.stats import t
        factor = float(t.ppf(0.975, df=len(x)-1))
    except ImportError:
        factor = 1.96
    return factor * float(np.std(x, ddof=1)) / math.sqrt(len(x))


def run(args):
    root = args.root
    stage = args.stage2 or (root/'Results/Stage2')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ids = selected_ids(args.selected_ids_file)
    if args.base_count is not None:
        ids = ids[:args.base_count]
    if args.samples < 1 or args.samples > 1000:
        raise ValueError('--samples must be between 1 and 1000')
    manifest = json.loads((stage/'stage2_manifest.json').read_text(encoding='utf-8-sig'))
    if not set(ids).issubset(set(manifest['base_runs'])):
        raise ValueError('Some selected IDs are absent from the Stage-2 manifest')

    # Confirm the saved second-level coefficient is the value being re-priced.
    for regime in ('reference', 'stress'):
        cfgfile = stage/regime/'random_activation_config.json'
        if cfgfile.is_file():
            cfg = json.loads(cfgfile.read_text(encoding='utf-8-sig'))
            stored = cfg.get('degradation_eur_per_mwh', None)
            if stored is not None and abs(float(stored)-BASE_RATE) > 1e-9:
                raise ValueError(f'{regime}: Stage-2 degradation coefficient is {stored}, expected {BASE_RATE}')

    stage1 = {}
    for i, base in enumerate(ids, 1):
        s2, d2 = stage1_totals(root, base, 's2')
        s3, d3 = stage1_totals(root, base, 's3')
        stage1[base] = {'s2': s2, 'd2': d2, 's3gross': s3, 'd3': d3}
        if i <= 2 or i == len(ids):
            print(f'Stage-1 {i}/{len(ids)}: {base}', flush=True)

    base_set = set(ids)
    collected = []
    diagnostics = {}
    for regime in ('reference', 'stress'):
        f = stage/regime/'all_random_activation_samples.csv'
        if not f.is_file():
            raise FileNotFoundError(f'Cannot find activation-sample file: {f}')
        actual = set(pd.read_csv(f, nrows=0).columns)
        missing = sorted(set(COLS)-actual)
        if missing:
            raise ValueError(f'{regime}: missing essential columns {missing}')

        # Save just the per-base statistics, keeping memory low for ~240k samples.
        totals = defaultdict(lambda: {c:{'n':0,'sum_s3':0.,'sum_s2':0.,'sum_gap':0.,
                                         'count_win':0,'sum_deg2':0.} for c in RATES})
        n_valid = 0
        max_errors = {'degradation_vs_throughput_eur':0.,'stage2_net_identity_eur':0.,
                      's2_vs_first_level_eur':0.,'s3gross_vs_first_level_eur':0.}
        for chunk in pd.read_csv(f, usecols=COLS, chunksize=25000, low_memory=False):
            chunk = chunk[chunk['base_run'].isin(base_set)]
            if args.samples < 1000:
                chunk = chunk[pd.to_numeric(chunk.sample_id, errors='raise') <= args.samples]
            if chunk.empty:
                continue
            numeric_cols = [c for c in COLS if c!='base_run']
            chunk[numeric_cols] = chunk[numeric_cols].apply(pd.to_numeric, errors='raise')
            if not np.isfinite(chunk[numeric_cols].to_numpy()).all():
                raise ValueError(f'{regime}: non-finite sample data')
            if (chunk['degradation_eur'] < -1e-8).any() or (chunk['throughput_mwh'] < -1e-8).any():
                raise ValueError(f'{regime}: negative throughput/degradation')
            chunk['s2stage1'] = chunk['base_run'].map({k:v['s2'] for k,v in stage1.items()})
            chunk['s3stage1'] = chunk['base_run'].map({k:v['s3gross'] for k,v in stage1.items()})
            d_err = (chunk['degradation_eur'] - BASE_RATE * chunk['throughput_mwh']).abs().max()
            net_formula = (chunk['s3_gross_no_activation_eur'] + chunk['activation_energy_cashflow_eur']
                           -chunk['net_recovery_cost_eur'] - chunk['degradation_eur'])
            net_err = (chunk['s3_activation_adjusted_net_eur'] - net_formula).abs().max()
            s2_err = (chunk['s2_total_eur'] - chunk['s2stage1']).abs().max()
            s3_err = (chunk['s3_gross_no_activation_eur'] - chunk['s3stage1']).abs().max()
            for k, val in zip(max_errors, (d_err,net_err,s2_err,s3_err)):
                max_errors[k] = max(max_errors[k], float(val))
            if (d_err > 1e-5 or net_err > 1e-5 or s2_err > 0.02 or s3_err > 0.02):
                raise ValueError(f'{regime}: stored revenue or degradation accounting does not match: {max_errors}')

            chunk['d2'] = chunk['base_run'].map({k:v['d2'] for k,v in stage1.items()})
            chunk['d3'] = chunk['base_run'].map({k:v['d3'] for k,v in stage1.items()})
            for rate in RATES:
                multiplier = (BASE_RATE-rate)/BASE_RATE
                s2x = chunk['s2_total_eur'] + multiplier * chunk['d2']
                s3x = chunk['s3_activation_adjusted_net_eur'] + multiplier * (
                    chunk['d3'] + chunk['degradation_eur'])
                # Repricing only; original asset allocation/physical dispatch remain unchanged.
                dx = s3x-s2x
                chunk[f'calc_s3_{rate}'] = s3x
                chunk[f'calc_s2_{rate}'] = s2x
                chunk[f'calc_gap_{rate}'] = dx
                chunk[f'win_{rate}'] = (dx >= 0).astype('int8')
            for base, g in chunk.groupby('base_run', sort=False):
                a = totals[base]
                n = len(g)
                for rate in RATES:
                    agg = a[rate]
                    agg['n'] += n
                    agg['sum_s3'] += float(g[f'calc_s3_{rate}'].sum())
                    agg['sum_s2'] += float(g[f'calc_s2_{rate}'].sum())
                    agg['sum_gap'] += float(g[f'calc_gap_{rate}'].sum())
                    agg['count_win'] += int(g[f'win_{rate}'].sum())
                    agg['sum_deg2'] += float(g['degradation_eur'].sum())
            n_valid += len(chunk)
        if set(totals) != base_set:
            raise ValueError(f'{regime}: sample data missing base scenarios: {base_set-set(totals)}')
        for base in ids:
            for rate in RATES:
                a=totals[base][rate]
                if a['n'] != args.samples:
                    raise ValueError(f'{regime} {base}: counted {a["n"]} samples, expected {args.samples}')
                collected.append({'regime':regime, 'base_run':base,
                                  'degradation_eur_per_mwh':rate, 'n_samples':a['n'],
                                  's2_fixed_dispatch_mean_eur':a['sum_s2']/a['n'],
                                  's3_fixed_dispatch_mean_eur':a['sum_s3']/a['n'],
                                  's3_minus_s2_fixed_dispatch_mean_eur':a['sum_gap']/a['n'],
                                  'prob_s3_ge_s2':a['count_win']/a['n'],
                                  'first_level_s2_deg_at25_eur':stage1[base]['d2'],
                                  'first_level_s3_deg_at25_eur':stage1[base]['d3'],
                                  'second_level_s3_deg_at25_mean_eur':a['sum_deg2']/a['n']})
        diagnostics[regime] = {'selected_bases':len(ids), 'evaluated_samples':n_valid,
                               'max_accounting_errors':max_errors}
        print(f'{regime}: verified {n_valid:,} saved activation samples; max accounting errors {max_errors}', flush=True)

    base_df = pd.DataFrame(collected)
    base_df.to_csv(out/'degradation_sensitivity_by_base.csv', index=False, float_format='%.12g')
    summary=[]
    for regime in ('reference','stress'):
        for rate in RATES:
            b=base_df[(base_df.regime==regime)&(base_df.degradation_eur_per_mwh==rate)]
            vals=b['s3_minus_s2_fixed_dispatch_mean_eur'].to_numpy()
            summary.append({'regime':regime,'degradation_eur_per_mwh':rate,
                            'n_base':len(b),'n_activation_samples':int(b.n_samples.sum()),
                            'mean_s2_eur':b.s2_fixed_dispatch_mean_eur.mean(),
                            'mean_s3_net_eur':b.s3_fixed_dispatch_mean_eur.mean(),
                            'mean_s3_minus_s2_eur':vals.mean(),
                            'mean_gap_95pct_ci_half_width_eur':mean_ci95(vals),
                            'prob_s3_ge_s2_pct':100*b.prob_s3_ge_s2.mean(),
                            'stage1_s2_degradation_at25_mean_eur':b.first_level_s2_deg_at25_eur.mean(),
                            'stage1_s3_degradation_at25_mean_eur':b.first_level_s3_deg_at25_eur.mean(),
                            'stage2_s3_degradation_at25_mean_eur':b.second_level_s3_deg_at25_mean_eur.mean()})
    sm = pd.DataFrame(summary)
    sm.to_csv(out/'degradation_sensitivity_summary.csv',index=False, float_format='%.12g')

    # If full May sample set, cross-check core paper figures (rounded EUR).
    if len(ids)==120 and args.samples==1000:
        ref=sm[(sm.regime=='reference')&(sm.degradation_eur_per_mwh==25.)].iloc[0]
        stress=sm[(sm.regime=='stress')&(sm.degradation_eur_per_mwh==25.)].iloc[0]
        compare={'mean_s2_eur':(float(ref.mean_s2_eur),69212.67),
                 'reference_s3net_eur':(float(ref.mean_s3_net_eur),124215.86),
                 'stress_s3net_eur':(float(stress.mean_s3_net_eur),40578.41)}
        for name,(actual,expected) in compare.items():
            if abs(actual-expected)>0.10:
                raise ValueError(f'Original May manuscript baseline does not match: {name} actual {actual:.2f}, expected {expected:.2f}')
        diagnostics['may_2026_manuscript_mean_reproduction'] = {
            name:{'actual_eur':round(actual,6),'paper_eur':expected}
            for name,(actual,expected) in compare.items()}
    diagnostics['scope'] = ('Ex-post fixed-first-level-dispatch sensitivity; reprices Stage-1 S2/S3 '
        'degradation and Stage-2 activation/recovery degradation, without market re-optimization. '
        'Any change in optimal dispatch, offered reserve MW, or activation outcomes with the '
        'degradation coefficient is beyond scope.')
    diagnostics['selected_base_ids'] = ids
    diagnostics['rates_eur_per_mwh'] = list(RATES)
    (out/'validation.json').write_text(json.dumps(diagnostics,indent=2,ensure_ascii=False), encoding='utf-8')
    print('\nRESULTS — FIXED dispatch and reserve allocation; no re-optimisation')
    print(sm[['regime','degradation_eur_per_mwh','n_base','n_activation_samples',
              'mean_s2_eur','mean_s3_net_eur','mean_s3_minus_s2_eur',
              'prob_s3_ge_s2_pct']].round(2).to_string(index=False))
    print('\nWROTE:',out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage2',type=Path,default=None)
    p.add_argument('--selected-ids-file',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--base-count',type=int,default=None)
    p.add_argument('--samples',type=int,default=20)
    args=p.parse_args()
    if args.base_count is not None and args.base_count < 1:
        p.error('--base-count must be >=1')
    run(args)

if __name__=='__main__':
    main()
