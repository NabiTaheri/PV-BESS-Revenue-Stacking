#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ex-post FCR direction x intensity sensitivity: Estonia PV+BESS May 2026.

READ ONLY. Uses saved Stage-2 per-product recovery and degradation components,
with original Stage-1 FCR awards. Does not rerun market allocation or MC.

For each (base, sample), replaces ONLY the FCR recovery/degradation contribution
in saved S3-net. FCR fractional activation intensity and upward share are varied
independently. aFRR/mFRR samples, settlement, S2, and reserve awards stay fixed.

The underlying Stage-2 implementation applies one FCR upward share to every
15-minute awarded FCR exposure in a sample. Therefore the FCR recovery cost can
be reconstructed with one FCR-exposure-weighted buy/sell settlement price per
base scenario. Those prices are algebraically recovered from the ORIGINAL
product-specific FCR cost records and are validated against every original
sample in both regimes. This is not independent historical price calibration.

Results are conditional *economic* revaluation, not chronological activation
or operational SOC/delivery feasibility. Replication checks fail closed.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ETA_C = .95
ETA_D = .95
DEGRADATION_EUR_MWH = 25.0
DT_HR = .25
DEFAULT_DIRECTIONS = (.50, .70, .90)
DEFAULT_INTENSITIES = (.05, .25, .50)
REQUIRED = [
    'base_run','sample_id', 'fcr_activation_fraction','fcr_up_share',
    'fcr_activated_up_mwh','fcr_activated_down_mwh',
    'fcr_net_recovery_cost_eur','fcr_degradation_eur',
    's2_total_eur','s3_gross_no_activation_eur',
    's3_activation_adjusted_net_eur','activation_energy_cashflow_eur',
    'net_recovery_cost_eur','degradation_eur',
]
MANUSCRIPT = {'s2':69212.67, 'reference_s3':124215.86, 'stress_s3':40578.41}


def read_ids(path: Path) -> list[str]:
    if not path.is_file(): raise FileNotFoundError(path)
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        rows = list(csv.reader(f))
    if not rows: raise ValueError('Empty selected ID file')
    header = [x.strip() for x in rows[0]]
    col = header.index('base_run') if 'base_run' in header else 0
    data = rows[1:] if 'base_run' in header else rows
    ids = [row[col].strip() for row in data if len(row)>col and row[col].strip()]
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('IDs empty or duplicated')
    return ids


def fcr_physical_energies(total_mwh, q):
    total = np.asarray(total_mwh,dtype=float)
    up = total*q
    down = total*(1.-q)
    drift = ETA_C*down-up/ETA_D
    buy = np.maximum(-drift,0.)/ETA_C
    sell = np.maximum(drift,0.)*ETA_D
    return up,down,buy,sell


def safe_price(quantity, value, label: str):
    quantity = np.asarray(quantity,dtype=float)
    value = np.asarray(value,dtype=float)
    mask = quantity>1e-5
    if mask.sum()==0:
        return None
    price = float(np.median(value[mask]/quantity[mask]))
    deviations = np.abs(value[mask]-price*quantity[mask])
    if float(deviations.max()) > 0.02:
        raise ValueError(f'{label}: product FCR recovery prices inconsistent across activation samples; max absolute cost mismatch EUR {deviations.max():.6g}')
    return price


def sample_mean_ci(vals):
    vals=np.asarray(vals,dtype=float)
    if len(vals)<2:return float('nan')
    try:
        from scipy.stats import t
        q=float(t.ppf(.975,len(vals)-1))
    except ImportError:q=1.96
    return float(q*vals.std(ddof=1)/math.sqrt(len(vals)))


def read_regime(stage: Path, regime: str, ids: list[str], ns: int):
    path=stage/regime/'all_random_activation_samples.csv'
    if not path.is_file():raise FileNotFoundError(path)
    header=pd.read_csv(path,nrows=0).columns
    miss=sorted(set(REQUIRED)-set(header))
    if miss:raise ValueError(f'{regime}: missing saved Stage-2 component columns {miss}')
    data=pd.read_csv(path,usecols=REQUIRED,low_memory=False)
    data=data[data.base_run.isin(ids)].copy()
    for c in REQUIRED:
        if c!='base_run': data[c]=pd.to_numeric(data[c],errors='raise')
    if not np.isfinite(data.drop(columns='base_run').to_numpy(dtype=float)).all():
        raise ValueError(f'{regime}: nan/inf in Stage-2 inputs')
    if not np.array_equal(data.sample_id.to_numpy(),data.sample_id.astype('int64').to_numpy()):
        raise ValueError(f'{regime}: non-integer sample IDs')
    data['sample_id']=data.sample_id.astype(int)
    data=data[data.sample_id.between(1,ns)]
    if data.duplicated(['base_run','sample_id']).any():
        raise ValueError(f'{regime}: duplicate base/sample rows')
    counts=data.groupby('base_run').sample_id.agg(['size','min','max'])
    if set(counts.index)!=set(ids) or not ((counts['size']==ns)&(counts['min']==1)&(counts['max']==ns)).all():
        raise ValueError(f'{regime}: missing activation pairs in selected base scenarios')
    if ((data.fcr_activation_fraction < -1e-10)|(data.fcr_activation_fraction>1+1e-10)|
        (data.fcr_up_share<-1e-10)|(data.fcr_up_share>1+1e-10)).any():
        raise ValueError(f'{regime}: invalid activation factors or direction shares')
    net=(data.s3_gross_no_activation_eur+data.activation_energy_cashflow_eur-
         data.net_recovery_cost_eur-data.degradation_eur)
    err=float((net-data.s3_activation_adjusted_net_eur).abs().max())
    if err>.02:raise ValueError(f'{regime}: saved Stage-2 net identity failed {err}')
    for col in ('s2_total_eur','s3_gross_no_activation_eur'):
        span=data.groupby('base_run')[col].agg(lambda x:x.max()-x.min()).max()
        if span>1e-5:raise ValueError(f'{regime}: Stage-1 {col} changes between conditional samples: {span}')
    return data


def awarded_fcr_exposure(root: Path, base: str):
    path=(root/'final_results'/base/'results'/
         'results_S3_reserve_competition_full_stack'/
         'monthly_afrr_mfrr_capacity_results.csv')
    if not path.is_file():raise FileNotFoundError(f'Stage-1 FCR award file missing: {path}')
    row=pd.read_csv(path,usecols=['Accepted_FCR_MW'])
    cap=pd.to_numeric(row['Accepted_FCR_MW'],errors='raise').to_numpy(dtype=float)
    if not np.isfinite(cap).all() or (cap < -1e-9).any():
        raise ValueError(f'{base}: bad FCR capacity series')
    return float(np.maximum(cap,0.).sum()*DT_HR)


def analyse_base(base, ref, stress, fcr_exp, shares, intensities, n_samples):
    original=pd.concat([ref,stress],ignore_index=True)
    f=original.fcr_activation_fraction.to_numpy(dtype=float)
    q=original.fcr_up_share.to_numpy(dtype=float)
    e_up=original.fcr_activated_up_mwh.to_numpy(dtype=float)
    e_down=original.fcr_activated_down_mwh.to_numpy(dtype=float)
    total=e_up+e_down
    recon_total=f*fcr_exp
    eerr=max(float(np.max(np.abs(total-recon_total))),
             float(np.max(np.abs(e_up-recon_total*q))),
             float(np.max(np.abs(e_down-recon_total*(1-q)))))
    if eerr>1e-5:raise ValueError(f'{base}: mismatch between awards, Stage-2 FCR intensity and activated energy: {eerr:.7g} MWh')
    u,d,buy,sell=fcr_physical_energies(total,q)
    old_recovery=original.fcr_net_recovery_cost_eur.to_numpy(dtype=float)
    old_degradation=original.fcr_degradation_eur.to_numpy(dtype=float)
    expected_degradation=DEGRADATION_EUR_MWH*(u+d+buy+sell)
    degerr=float(np.max(np.abs(expected_degradation-old_degradation)))
    if degerr>.02:raise ValueError(f'{base}: FCR degradation throughput identity mismatch EUR {degerr}')
    if (old_degradation<-1e-8).any(): raise ValueError(f'{base}: negative FCR degradation')
    pbuy=safe_price(buy,old_recovery,f'{base} buy price')
    psell=safe_price(sell,-old_recovery,f'{base} sell price')
    if pbuy is None:
        if (buy>1e-8).any():raise ValueError(f'{base}: unidentifiable buy settlement price')
        if fcr_exp>1e-7:raise ValueError(f'{base}: FCR awards present, but no original upward drift samples to identify purchase settlement price')
        pbuy=0.
    if psell is None:psell=0.
    baseline_pred=pbuy*buy-psell*sell
    recerr=float(np.max(np.abs(baseline_pred-old_recovery)))
    if recerr>.02:raise ValueError(f'{base}: cannot reconstruct original FCR recovery cost exactly; max EUR {recerr:.7g}')
    # None of the intended upward shares (>=0.5) creates a recovery sale
    # with 95% symmetric efficiency. If one is changed to a lower value,
    # the original data must identify a weighted sell price.
    all_shares=sorted(set(shares))
    if any(x < (ETA_C*ETA_D)/(1.+ETA_C*ETA_D)-1e-12 for x in all_shares) and not (sell>1e-5).any():
        raise ValueError(f'{base}: requested low FCR upward share needs sell-price evidence missing from saved samples')

    summaries=[]
    split={'reference':ref,'stress':stress}
    for regime,df in split.items():
        n=len(df)
        fvec=df.fcr_activation_fraction.to_numpy(dtype=float)
        s2=df.s2_total_eur.to_numpy(dtype=float)
        gross=df.s3_activation_adjusted_net_eur.to_numpy(dtype=float)
        fc_old=df.fcr_net_recovery_cost_eur.to_numpy(dtype=float)
        fd_old=df.fcr_degradation_eur.to_numpy(dtype=float)
        without_fcr=gross+fc_old+fd_old
        if len(df)!=n_samples:raise ValueError('unexpected sample count')
        for mode in ('direction_only','fixed_fcr_intensity'):
            fs=[None] if mode=='direction_only' else list(intensities)
            for f_fixed in fs:
                fcr_total=(fvec if f_fixed is None else np.full(n,f_fixed))*fcr_exp
                for share in shares:
                    u,d,rec_b,rec_s=fcr_physical_energies(fcr_total,share)
                    c_rec=rec_b*pbuy-rec_s*psell
                    c_deg=DEGRADATION_EUR_MWH*(u+d+rec_b+rec_s)
                    s3=without_fcr-c_rec-c_deg
                    gap=s3-s2
                    summaries.append({
                      'regime':regime,'base_run':base,'evaluation':mode,
                      'fcr_activation_fraction_fixed':f_fixed if f_fixed is not None else np.nan,
                      'fcr_up_share':share,'n_samples':len(gap),
                      'mean_s2_eur':float(s2.mean()),
                      'mean_s3_net_eur':float(s3.mean()),
                      'mean_gap_eur':float(gap.mean()),
                      'prob_s3_ge_s2':float((gap>=0).mean()),
                      'mean_fcr_recovery_cost_eur':float(c_rec.mean()),
                      'mean_fcr_degradation_eur':float(c_deg.mean()),
                    })
    meta={'base_run':base,'fcr_awarded_capacity_hours_mwh_per_unit_fraction':fcr_exp,
          'inferred_fcr_weighted_buy_price_eur_mwh':pbuy,
          'inferred_fcr_weighted_sell_price_eur_mwh':psell if (sell>1e-5).any() else None,
          'original_cost_reconstruction_max_abs_eur':recerr,
          'original_degradation_reconstruction_max_abs_eur':degerr,
          'original_energy_reconstruction_max_abs_mwh':eerr}
    return summaries,meta


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--selected-ids-file',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stage2',type=Path,default=None)
    ap.add_argument('--samples',type=int,default=1000)
    ap.add_argument('--base-count',type=int,default=None)
    ap.add_argument('--directions',type=float,nargs='+',default=list(DEFAULT_DIRECTIONS))
    ap.add_argument('--intensities',type=float,nargs='+',default=list(DEFAULT_INTENSITIES))
    args=ap.parse_args()
    if not 1<=args.samples<=1000:raise ValueError('--samples must be within 1...1000')
    if args.base_count is not None and args.base_count<1:raise ValueError('--base-count must be >=1')
    if len(set(args.directions))!=len(args.directions) or any(not(.0<=x<=1.) for x in args.directions):
        raise ValueError('Direction shares must be distinct and between 0 and 1')
    if len(set(args.intensities))!=len(args.intensities) or any(not(.0<=x<=1.) for x in args.intensities):
        raise ValueError('FCR activation fractions must be distinct and between 0 and 1')
    if not args.root.is_dir():raise FileNotFoundError(args.root)
    stage=args.stage2 or args.root/'Results/Stage2'
    ids=read_ids(args.selected_ids_file)
    if args.base_count is not None:ids=ids[:args.base_count]
    manifest=json.loads((stage/'stage2_manifest.json').read_text(encoding='utf-8-sig'))
    if not set(ids).issubset(set(manifest['base_runs'])):
        raise ValueError('Some selected base IDs absent from Stage-2 manifest')
    args.out.mkdir(parents=True,exist_ok=True)
    samples={r:read_regime(stage,r,ids,args.samples) for r in ['reference','stress']}
    print(f'INPUT: {len(ids)} base scenarios x {args.samples} samples per regime; FCR shares {args.directions}; FCR intensity grid {args.intensities}',flush=True)
    d={r:{base:g for base,g in samples[r].groupby('base_run',sort=False)} for r in samples}
    allrows=[]; validation=[]
    for idx,base in enumerate(ids,1):
        b1=d['reference'][base];b2=d['stress'][base]
        if abs(b1.s2_total_eur.iloc[0]-b2.s2_total_eur.iloc[0])>1e-5:
            raise ValueError(f'{base}: matched S2 differs across activation regimes')
        awarded=awarded_fcr_exposure(args.root,base)
        rows,v=analyse_base(base,b1,b2,awarded,args.directions,args.intensities,args.samples)
        allrows.extend(rows); validation.append(v)
        if idx<=2 or idx%20==0 or idx==len(ids): print(f'VALIDATED {idx}/{len(ids)}: {base}',flush=True)
    # Confirm the ORIGINAL published Monte Carlo sample-set means before reporting counterfactuals.
    originals={r:{'s2':float(samples[r].s2_total_eur.mean()),
                  's3':float(samples[r].s3_activation_adjusted_net_eur.mean())}
               for r in samples}
    if len(ids)==120 and args.samples==1000:
        for name,actual,target in [('S2',originals['reference']['s2'],MANUSCRIPT['s2']),
                                   ('Reference S3',originals['reference']['s3'],MANUSCRIPT['reference_s3']),
                                   ('Stress S3',originals['stress']['s3'],MANUSCRIPT['stress_s3'])]:
            if abs(actual-target)>.10:raise ValueError(f'May manuscript baseline reproduction failed {name}: {actual} vs {target}')
    bdf=pd.DataFrame(allrows)
    if bdf.empty:raise ValueError('No samples analysed')
    bdf.to_csv(args.out/'fcr_direction_intensity_by_base.csv',index=False,float_format='%.12g')
    checks=pd.DataFrame(validation)
    checks.to_csv(args.out/'fcr_original_accounting_validation_by_base.csv',index=False,float_format='%.12g')
    grp_cols=['regime','evaluation','fcr_activation_fraction_fixed','fcr_up_share']
    records=[]
    for key,x in bdf.groupby(grp_cols,sort=False,dropna=False):
        reg,kind,frac,share=key
        values=x.mean_gap_eur.to_numpy(dtype=float)
        records.append({
          'regime':reg,'evaluation':kind,
          'fcr_activation_fraction_fixed':frac,'fcr_up_share':share,
          'n_base':len(x),'n_conditional_pairs':int(x.n_samples.sum()),
          'mean_s2_eur':float(x.mean_s2_eur.mean()),
          'mean_s3_net_eur':float(x.mean_s3_net_eur.mean()),
          'mean_s3_minus_s2_eur':float(values.mean()),
          'gap_scenario_level_ci95_halfwidth_eur':sample_mean_ci(values),
          'prob_s3_ge_s2_pct':100.*float(x.prob_s3_ge_s2.mean()),
          'mean_fcr_recovery_cost_eur':float(x.mean_fcr_recovery_cost_eur.mean()),
          'mean_fcr_degradation_eur':float(x.mean_fcr_degradation_eur.mean()),
        })
    sm=pd.DataFrame(records).sort_values(['evaluation','regime','fcr_activation_fraction_fixed','fcr_up_share'],kind='stable')
    direction=sm[sm.evaluation=='direction_only'].copy()
    grid=sm[sm.evaluation=='fixed_fcr_intensity'].copy()
    direction.to_csv(args.out/'fcr_direction_only_summary.csv',index=False,float_format='%.12g')
    grid.to_csv(args.out/'fcr_direction_intensity_grid.csv',index=False,float_format='%.12g')
    # This is the critical limitation to preserve in published interpretations.
    note={
      'scope':'Counterfactual repricing of saved activation samples. Original Stage-1 battery dispatch and reserve awards are fixed; original aFRR/mFRR sample factors, recovery and activation settlement are fixed. The FCR activation intensity and upward share are systematically varied using the recorded product-level FCR recovery/degradation accounting.',
      'no_reoptimization_or_operational_replay':True,
      'not_a_chronological_SOC_feasibility_test':True,
      'price_inference':'FCR award-weighted recovery buy/sell prices are inferred algebraically from stored original FCR product-level recovery costs; all original samples are exactly checked against those inferred prices. No independent measured activation trace or price calibration is claimed.',
      'reference_and_stress_first_level_baseline':originals,
      'selected_base_count':len(ids),'samples_per_base_regime':args.samples,
      'selected_base_ids':ids,'directions':args.directions,'intensities':args.intensities,
      'max_original_recovery_cost_reconstruction_error_eur':float(checks.original_cost_reconstruction_max_abs_eur.max()),
      'max_original_degradation_reconstruction_error_eur':float(checks.original_degradation_reconstruction_max_abs_eur.max()),
      'max_original_FCR_energy_reconstruction_error_mwh':float(checks.original_energy_reconstruction_max_abs_mwh.max()),
    }
    (args.out/'validation.json').write_text(json.dumps(note,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nDIRECTION ONLY (original reserve-activation intensity samples unchanged):')
    print(direction[['regime','fcr_up_share','mean_s3_minus_s2_eur','prob_s3_ge_s2_pct']].round(2).to_string(index=False))
    print('\nFCR INTENSITY x DIRECTION (aFRR/mFRR activation distributions unchanged):')
    print(grid[['regime','fcr_activation_fraction_fixed','fcr_up_share','mean_s3_minus_s2_eur','prob_s3_ge_s2_pct']].round(2).to_string(index=False))
    print('\nVALIDATION max original FCR recovery cost error EUR',note['max_original_recovery_cost_reconstruction_error_eur'])
    print('WROTE:',args.out,flush=True)

if __name__=='__main__':main()
