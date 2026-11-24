#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer 1 / Comment 3: exact empirical joint distribution of paired S2 and S3-net.

The S2 revenue is constant within each first-level scenario; 1000 S3-net
activation realisations are conditionally compared with that SAME S2 revenue.
Sample IDs are paired within a base scenario, not across Reference and Stress.

Read-only: no re-simulation, no battery/market reoptimisation.
Outputs: a complete paired table by regime, a joint 5x5 frequency table,
conditional S3 distributions by S2 quintile, scenario-level joint data,
methodology/validation JSON, and a compact summary.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ['base_run', 'sample_id', 's2_total_eur', 's3_activation_adjusted_net_eur']
MANUSCRIPT = {'s2': 69212.67, 'reference_s3': 124215.86, 'stress_s3': 40578.41}


def load_ids(filename: Path) -> list[str]:
    with filename.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f'Empty selected IDs list: {filename}')
    header = [str(s).strip() for s in rows[0]]
    field = header.index('base_run') if 'base_run' in header else 0
    data = rows[1:] if 'base_run' in header else rows
    ids = [row[field].strip() for row in data if len(row)>field and row[field].strip()]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Selected base_run IDs missing or duplicated')
    return ids


def ci95_scenario_mean(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 2:
        return float('nan')
    try:
        from scipy.stats import t
        factor = float(t.ppf(0.975, n-1))
    except ImportError:
        factor = 1.96
    return float(factor*np.std(arr, ddof=1)/math.sqrt(n))


def quantile_assign(series: pd.Series, n=5) -> tuple[np.ndarray,list[float]]:
    """Define bins with empirical quantile cutoffs; reject degenerate cutoffs."""
    cuts = np.quantile(series.to_numpy(dtype=float), np.linspace(0,1,n+1))
    if np.unique(cuts).size != n+1:
        raise ValueError('Repeated quantile cutoffs: cannot define a meaningful 5x5 binned joint distribution')
    groups = np.searchsorted(cuts[1:-1], series.to_numpy(dtype=float), side='right')+1
    if min(groups)<1 or max(groups)>n:
        raise AssertionError('Quantile group invalid')
    return groups, cuts.tolist()


def summarise_regime(filename:Path, ids:list[str], n_per_base:int, label:str, out:Path):
    missing = sorted(set(REQUIRED)-set(pd.read_csv(filename, nrows=0).columns))
    if missing:
        raise ValueError(f'{label}: required CSV columns absent: {missing}')
    df = pd.read_csv(filename, usecols=REQUIRED, low_memory=False)
    df = df[df.base_run.isin(set(ids))].copy()
    if df.empty:
        raise ValueError(f'{label}: no selected scenario observations')
    for col in ('sample_id','s2_total_eur','s3_activation_adjusted_net_eur'):
        df[col] = pd.to_numeric(df[col], errors='raise')
    if not np.isfinite(df[['sample_id','s2_total_eur','s3_activation_adjusted_net_eur']].to_numpy()).all():
        raise ValueError(f'{label}: infinite or missing numeric inputs')
    if not np.all(df.sample_id.to_numpy() == df.sample_id.astype('int64').to_numpy()):
        raise ValueError(f'{label}: noninteger sample IDs')
    df.sample_id = df.sample_id.astype('int64')
    df = df[df.sample_id <= n_per_base]
    if df.duplicated(['base_run','sample_id']).any():
        raise ValueError(f'{label}: duplicated (base_run,sample_id) pairs')
    counts = df.groupby('base_run',sort=False).sample_id.agg(['count','min','max','nunique'])
    if set(counts.index)!=set(ids) or (counts['count'] != n_per_base).any() or (counts['nunique'] != n_per_base).any() or (counts['min']!=1).any() or (counts['max']!=n_per_base).any():
        bad = counts[(counts['count']!=n_per_base)|(counts['min']!=1)|(counts['max']!=n_per_base)]
        raise ValueError(f'{label}: missing or nonconsecutive conditional sample IDs; first bad rows {bad.head().to_dict()}')
    s2_range = df.groupby('base_run').s2_total_eur.agg(['min','max'])
    max_s2_within_base = float((s2_range['max']-s2_range['min']).abs().max())
    if max_s2_within_base>1e-5:
        raise ValueError(f'{label}: S2 should be fixed for every base scenario; max span {max_s2_within_base}')
    df['gap_eur'] = df['s3_activation_adjusted_net_eur'] - df['s2_total_eur']
    df['s3_at_least_s2'] = (df['gap_eur'] >= 0).astype('int8')
    df['base_run'] = pd.Categorical(df['base_run'], categories=ids, ordered=True)
    df = df.sort_values(['base_run','sample_id']).reset_index(drop=True)
    df['base_run']=df.base_run.astype(str)

    df.rename(columns={'s2_total_eur':'s2_eur','s3_activation_adjusted_net_eur':'s3_net_eur'},inplace=True)
    df.to_csv(out/f'{label}_paired_120k_samples.csv',index=False,float_format='%.12g')

    per_base = df.groupby('base_run',sort=False).agg(s2_eur=('s2_eur','first'),s3_mean_eur=('s3_net_eur','mean'),
              s3_std_conditional_eur=('s3_net_eur','std'),s3_p05_conditional_eur=('s3_net_eur',lambda x: x.quantile(.05)),
              s3_p50_conditional_eur=('s3_net_eur','median'),s3_p95_conditional_eur=('s3_net_eur',lambda x: x.quantile(.95)),
              gap_mean_eur=('gap_eur','mean'),prob_s3_ge_s2=('s3_at_least_s2','mean'),n_activation=('sample_id','count')).reset_index()
    per_base.to_csv(out/f'{label}_scenario_level_paired.csv',index=False,float_format='%.12g')

    # The S2 quintile is based on the 120 independent first-level scenarios,
    # which avoids treating repeated S2 values as 120k independent observations.
    n_bins=min(5,len(per_base))
    bins_s2, cut_s2 = quantile_assign(per_base.s2_eur,n_bins)
    base_to_q=dict(zip(per_base.base_run,bins_s2))
    df['s2_quintile'] = df.base_run.map(base_to_q).astype(int)
    bins_s3, cut_s3 = quantile_assign(df.s3_net_eur,n_bins)
    df['s3_quintile'] = bins_s3
    crosstab = pd.crosstab(df.s2_quintile,df.s3_quintile).reindex(index=range(1,n_bins+1),columns=range(1,n_bins+1),fill_value=0)
    if int(crosstab.to_numpy().sum())!=len(df):
        raise AssertionError('Joint histogram does not sum to total sample count')
    # Long format: row=conditioned S2 quintile, col=S3 quintile.
    joint=[]
    for qi in range(1,n_bins+1):
        row_total=int(crosstab.loc[qi].sum())
        for qj in range(1,n_bins+1):
            n=int(crosstab.loc[qi,qj])
            joint.append({'regime':label,'s2_quintile':qi,'s3_quintile':qj,
                          'n_paired_samples':n,'joint_probability_pct':100*n/len(df),
                          'conditional_probability_s3_quintile_given_s2_pct':100*n/row_total})
    pd.DataFrame(joint).to_csv(out/f'{label}_joint_distribution_{n_bins}x{n_bins}.csv',index=False,float_format='%.10g')

    conditional=[]
    for i, sub in df.groupby('s2_quintile',sort=True):
        s3=sub.s3_net_eur
        conditional.append({'regime':label,'s2_quintile':i,
                            'n_independent_base_scenarios':int((per_base.base_run.map(base_to_q)==i).sum()),
                            'n_conditional_pairs':len(sub),'s2_min_eur':float(sub.s2_eur.min()),
                            's2_max_eur':float(sub.s2_eur.max()),'s2_mean_eur':float(sub.s2_eur.mean()),
                            's3_mean_eur':float(s3.mean()),'s3_p05_eur':float(s3.quantile(.05)),
                            's3_median_eur':float(s3.median()),'s3_p95_eur':float(s3.quantile(.95)),
                            'gap_mean_eur':float(sub.gap_eur.mean()),
                            'prob_s3_ge_s2_pct':float(100*sub.s3_at_least_s2.mean())})
    pd.DataFrame(conditional).to_csv(out/f'{label}_conditional_S3_by_S2_quintile.csv',index=False,float_format='%.10g')

    pearson_pooled = float(df.s2_eur.corr(df.s3_net_eur))
    pearson_scenario = float(per_base.s2_eur.corr(per_base.s3_mean_eur))
    spearman_scenario = float(per_base.s2_eur.corr(per_base.s3_mean_eur,method='spearman'))
    paired_gap_mean_by_base = per_base.gap_mean_eur.to_numpy()
    win_rate_by_base = per_base.prob_s3_ge_s2.to_numpy()
    result={'regime':label,'n_base':len(per_base),'conditional_samples_per_base':n_per_base,
            'n_conditional_pairs':len(df),
            'mean_s2_eur':float(per_base.s2_eur.mean()),
            'mean_s3net_eur':float(per_base.s3_mean_eur.mean()),
            'mean_paired_gap_eur':float(paired_gap_mean_by_base.mean()),
            'mean_gap_scenario_level_ci95_halfwidth_eur':ci95_scenario_mean(paired_gap_mean_by_base),
            'pooled_gap_p05_eur':float(df.gap_eur.quantile(.05)),
            'pooled_gap_median_eur':float(df.gap_eur.median()),
            'pooled_gap_p95_eur':float(df.gap_eur.quantile(.95)),
            'pooled_probability_s3_ge_s2_pct':float(100*win_rate_by_base.mean()),
            'probability_scenario_level_ci95_halfwidth_percentage_points':100*ci95_scenario_mean(win_rate_by_base),
            'pearson_corr_s2_s3_all_conditional_pairs':pearson_pooled,
            'pearson_corr_s2_s3_scenario_means_nbase':pearson_scenario,
            'spearman_corr_s2_s3_scenario_means_nbase':spearman_scenario,
            'max_within_base_s2_difference_eur':max_s2_within_base}
    print(f'{label}: {len(per_base)} bases, {len(df):,} conditional pairs | '
          f'mean S2={result["mean_s2_eur"]:,.2f}; mean S3={result["mean_s3net_eur"]:,.2f}; '
          f'gap={result["mean_paired_gap_eur"]:,.2f}; P(S3>=S2)={result["pooled_probability_s3_ge_s2_pct"]:.2f}%; '
          f'r_scenario={pearson_scenario:.4f}',flush=True)
    return result,{'s2_base_quantile_thresholds_eur':cut_s2,'s3_all_pair_quantile_thresholds_eur':cut_s3}


def main():
    a=argparse.ArgumentParser(description='Paired S2-S3-net empirical joint distribution, May 2026')
    a.add_argument('--root',required=True,type=Path)
    a.add_argument('--selected-ids-file',required=True,type=Path)
    a.add_argument('--out',required=True,type=Path)
    a.add_argument('--stage2',type=Path,default=None)
    a.add_argument('--samples',type=int,default=1000)
    a.add_argument('--base-count',type=int,default=None,help='Optional small validation pilot')
    args=a.parse_args()
    if not 1<=args.samples<=1000:
        raise ValueError('--samples must be 1..1000')
    ids=load_ids(args.selected_ids_file)
    if args.base_count is not None:
        if args.base_count<1: raise ValueError('--base-count must be positive')
        ids=ids[:args.base_count]
    stage=args.stage2 or args.root/'Results/Stage2'
    manifest=json.loads((stage/'stage2_manifest.json').read_text(encoding='utf-8-sig'))
    if not set(ids).issubset(set(manifest['base_runs'])):
        raise ValueError('Selected IDs not covered by Stage 2 manifest')
    args.out.mkdir(parents=True,exist_ok=True)
    summary=[]
    qc={'selected_ids':ids,'conditional_sample_count_per_base':args.samples,
        'note':('The 1000 conditional S3 outcomes share the single fixed S2 revenue within each '
                'base scenario. Each (base_run,sample_id) row is a valid economic pairing, '
                'but not an independent first-level observation; confidence intervals '
                'use the independent base scenarios. Both regimes are independent '
                'alternative activation conditions: sample_id does not pair Reference to Stress. '
                'The data are economic monthly outcomes, not matched chronological reserve-activation '
                'traces for S2 and S3.'), 'quintile_edges':{}}
    for label in ('reference','stress'):
        f=stage/label/'all_random_activation_samples.csv'
        if not f.is_file(): raise FileNotFoundError(str(f))
        row,edges=summarise_regime(f,ids,args.samples,label,args.out)
        summary.append(row)
        qc['quintile_edges'][label]=edges
    result=pd.DataFrame(summary)
    # Verify scenario-by-scenario that Reference/Stress use the identical S2 baseline.
    rbase=pd.read_csv(args.out/'reference_scenario_level_paired.csv')
    sbase=pd.read_csv(args.out/'stress_scenario_level_paired.csv')
    check=rbase[['base_run','s2_eur']].merge(sbase[['base_run','s2_eur']],on='base_run',how='outer',validate='one_to_one')
    err=float((check.s2_eur_x-check.s2_eur_y).abs().max())
    if err>1e-5 or len(check)!=len(ids):
        raise ValueError(f'Reference and Stress base-level S2 values inconsistent: max error EUR {err:.8g}')
    qc['max_s2_difference_between_regimes_eur']=err
    # Strict baseline data reproduction check only for all 120 bases, all 1000 samples.
    if len(ids)==120 and args.samples==1000:
        ref,stress=result.iloc[0],result.iloc[1]
        expect={'S2':(ref.mean_s2_eur,MANUSCRIPT['s2']),
                'S3 Reference':(ref.mean_s3net_eur,MANUSCRIPT['reference_s3']),
                'S3 Stress':(stress.mean_s3net_eur,MANUSCRIPT['stress_s3']),
                'S2 equal across regimes':(ref.mean_s2_eur,stress.mean_s2_eur)}
        for key,(actual,target) in expect.items():
            if abs(actual-target)>0.10:
                raise ValueError(f'Manuscript baseline mismatch: {key} -> {actual:.5f}, expected {target:.5f}')
        qc['full_may_baseline_reproduction']={k:{'observed':float(v),'expected':float(e)} for k,(v,e) in expect.items()}
    result.to_csv(args.out/'paired_revenue_joint_summary.csv',index=False,float_format='%.12g')
    (args.out/'pairing_validation.json').write_text(json.dumps(qc,ensure_ascii=False,indent=2),encoding='utf8')
    print('\nCOMPACT RESULTS')
    print(result[['regime','n_base','n_conditional_pairs','mean_paired_gap_eur','pooled_probability_s3_ge_s2_pct',
                  'pearson_corr_s2_s3_scenario_means_nbase']].round(4).to_string(index=False))
    print('\nOutput folder:',args.out)
    print('Primary reviewer tables: paired_revenue_joint_summary.csv, *_joint_distribution_5x5.csv, '
          '*_conditional_S3_by_S2_quintile.csv')

if __name__=='__main__':
    main()
