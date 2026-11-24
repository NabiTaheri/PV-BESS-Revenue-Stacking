#!/usr/bin/env python3
"""Read-only, ex-post interval SOC recovery screening for Estonia PV+BESS (v3).

NOT a replay of observed reserve activation. Reconstructs interval-level
reserve-energy exposure using Stage-2 sample fractions and accepted capacity.
The resulting violation fractions are conditional on the stated allocation
and greedy-recovery policy, NOT observed true infeasibility probabilities.

Uses ONLY Stage-1 outputs and Stage-2 samples. Runs NO optimizers.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ETA_C = 0.95
ETA_D = 0.95
EPS = 1e-8
ENERGY_MWH = 10.0
DT = 0.25

S3 = 'results_S3_reserve_competition_full_stack'
CAP = 'monthly_afrr_mfrr_capacity_results.csv'
DISP = 'monthly_da_after_capacity_report.csv'
POWER = 'monthly_remaining_capacity_after_fcr_afrr_mfrr.csv'

REQUIRED_SAMPLES = [
    'base_run','sample_id','fcr_activation_fraction',
    'afrr_activation_fraction','mfrr_activation_fraction','fcr_up_share',
    'fcr_activated_up_mwh','fcr_activated_down_mwh',
    'afrr_activated_up_mwh','afrr_activated_down_mwh',
    'mfrr_activated_up_mwh','mfrr_activated_down_mwh',
    'recovery_buy_mwh','recovery_sell_mwh',
]


def num(df, field):
    if field not in df.columns:
        raise ValueError(f'Missing mandatory column: {field}')
    s=pd.to_numeric(df[field],errors='coerce')
    if s.isna().any(): raise ValueError(f'Null/non-numeric values in {field}')
    return s.to_numpy(dtype=float)


def ts(df, name):
    if 'timestamp' not in df: raise ValueError(f'{name}: missing timestamp')
    x=pd.to_datetime(df.timestamp,errors='coerce')
    if x.isna().any(): raise ValueError(f'{name}: invalid timestamps')
    if x.duplicated().any(): raise ValueError(f'{name}: duplicated timestamps')
    return x


def load_interval_inputs(root, base):
    loc=root/'final_results'/base/'results'/S3
    cap=pd.read_csv(loc/CAP,encoding='utf-8-sig')
    da=pd.read_csv(loc/DISP,encoding='utf-8-sig')
    rem=pd.read_csv(loc/POWER,encoding='utf-8-sig')
    ac=pd.read_csv(root/'Data'/'actuals.csv',encoding='utf-8-sig')
    arrays=[]
    for df,name in [(cap,'capacity'),(da,'S3 dispatch'),(rem,'remaining power'),(ac,'actuals')]:
        time=ts(df,name); df=df.copy();df['_ts']=time
        # Some source tables may contain extra timestamps (e.g. other months).
        df=df.sort_values('_ts')
        arrays.append(df)
    cap,da,rem,ac=arrays
    # Strict one-to-one matching. Never create duplicate rows on merge.
    # Real S3 capacity output repeats the same 4 field names found in the
    # post-reservation remaining-power table. Keep the POST-RESERVATION values
    # explicitly; otherwise pandas renames duplicates as *_x / *_y and
    # subsequent lookups for p_charge_max_mw fail.
    headroom_cols=['p_charge_max_mw','p_discharge_max_mw','soc_min_pct','soc_max_pct']
    missing=[k for k in headroom_cols if k not in rem.columns]
    if missing:
        raise ValueError(f'{base}: missing post-reservation envelope fields {missing}')
    if any(k in cap.columns for k in headroom_cols):
        print(f'INFO {base}: capacity summary and remaining-power CSV share '
              'headroom/SOC column names; using post-reservation values '
              'from monthly_remaining_capacity_after_fcr_afrr_mfrr.csv.',flush=True)
    base_tab=cap.drop(columns=headroom_cols,errors='ignore').copy()
    joins=[
        (da,['soc_after_all_mwh','battery_grid_charge_mw',
             'battery_pv_charge_mw','battery_discharge_mw'],'S3 dispatch'),
        (rem,headroom_cols,'remaining-power output'),
        (ac,['balancing_flag'],'actuals'),
    ]
    for df,cols,source in joins:
        missing=[k for k in cols if k not in df.columns]
        if missing: raise ValueError(f'{base}: {source} columns absent: {missing}')
        collisions=[k for k in cols if k in base_tab.columns]
        if collisions: raise ValueError(f'{base}: unexpected overlapping columns from {source}: {collisions}')
        part=df[['_ts']+cols]
        base_tab=base_tab.merge(part,on='_ts',how='left',validate='one_to_one')
    required=['soc_after_all_mwh']+headroom_cols+['balancing_flag']
    for col in required:
        if col not in base_tab.columns:
            raise ValueError(f'{base}: missing merged column {col}; fields: {list(base_tab.columns)}')
        if base_tab[col].isna().any():
            raise ValueError(f'{base}: missing values after joining timestamps for {col}')
    base_tab=base_tab.sort_values('_ts').reset_index(drop=True)
    if len(base_tab)<2: raise ValueError(f'{base}: too few intervals')
    timestep=np.diff(base_tab['_ts'].values.astype('datetime64[s]').astype('int64'))/3600
    # Use the same 15-minute energy-accounting rule; reject irregular lengths.
    if not np.all(np.isclose(timestep,DT,atol=1e-4)):
        raise ValueError(f'{base}: expected consecutive 15-minute intervals; got {np.unique(timestep)[:10]}')
    required_cap=['Accepted_FCR_MW','Accepted_aFRR_UP_MW','Accepted_aFRR_DOWN_MW',
                  'Accepted_mFRR_UP_MW','Accepted_mFRR_DOWN_MW']
    for k in required_cap:
        if k not in base_tab: raise ValueError(f'{base}: missing reserve column {k}')
    flag=num(base_tab,'balancing_flag')
    # Monthly Monte Carlo generator gates upward/downward aFRR and mFRR by the balancing flag.
    exposure={
       'fcr':np.maximum(num(base_tab,'Accepted_FCR_MW'),0)*DT,
       'afrr_up':np.maximum(num(base_tab,'Accepted_aFRR_UP_MW'),0)*DT*(flag>0),
       'afrr_down':np.maximum(num(base_tab,'Accepted_aFRR_DOWN_MW'),0)*DT*(flag<0),
       'mfrr_up':np.maximum(num(base_tab,'Accepted_mFRR_UP_MW'),0)*DT*(flag>0),
       'mfrr_down':np.maximum(num(base_tab,'Accepted_mFRR_DOWN_MW'),0)*DT*(flag<0),
    }
    # The post-reservation bounds describe the SOC interval remaining for
    # market dispatch AFTER accepted reserve commitments have tightened it.
    # They are not the physical battery SOC limits; testing reserve activation
    # against them would double-count the energy headroom reserved for delivery.
    soc=num(base_tab,'soc_after_all_mwh')
    reserved_lower=num(base_tab,'soc_min_pct')/100*ENERGY_MWH
    reserved_upper=num(base_tab,'soc_max_pct')/100*ENERGY_MWH
    # Verify the physical battery SOC limits against the manuscript and actual
    # market_config used in the study before reporting manuscript numbers.
    lower=np.full(len(base_tab),2.0)  # 20% of the 10 MWh battery
    upper=np.full(len(base_tab),8.0)  # 80% of the 10 MWh battery
    charge=num(base_tab,'battery_grid_charge_mw')+num(base_tab,'battery_pv_charge_mw')
    discharge=num(base_tab,'battery_discharge_mw')
    # Conservative additional recovery headroom, AFTER reserve locking AND scheduled DA dispatch.
    hc=np.maximum(0,num(base_tab,'p_charge_max_mw')-charge)
    hd=np.maximum(0,num(base_tab,'p_discharge_max_mw')-discharge)
    if (reserved_lower>reserved_upper+EPS).any():
        raise ValueError(f'{base}: invalid post-reservation SOC envelope')
    # Report departures from the stricter POST-RESERVATION band for separate
    # assessment. Do not relabel them as physical SOC violations.
    reservation_excursion=np.maximum(reserved_lower-soc,0)+np.maximum(soc-reserved_upper,0)
    physical_excursion=np.maximum(lower-soc,0)+np.maximum(soc-upper,0)
    base_tab['_baseline_outside_reserved_mwh']=reservation_excursion
    base_tab['_baseline_outside_physical_mwh']=physical_excursion
    base_tab['_baseline_power_overschedule_charge_mw']=np.maximum(charge-num(base_tab,'p_charge_max_mw'),0)
    base_tab['_baseline_power_overschedule_discharge_mw']=np.maximum(discharge-num(base_tab,'p_discharge_max_mw'),0)
    n_res=int((reservation_excursion>1e-4).sum())
    n_phys=int((physical_excursion>1e-4).sum())
    n_power=int(((base_tab['_baseline_power_overschedule_charge_mw']>1e-4) | (base_tab['_baseline_power_overschedule_discharge_mw']>1e-4)).sum())
    print(f'DIAGNOSTIC {base}: post-reservation SOC band exceedances '
          f'{n_res}/{len(soc)} (max {reservation_excursion.max():.6f} MWh); '
          f'physical 20-80% SOC exceedances {n_phys}/{len(soc)}; '
          f'baseline dispatch exceeding reported remaining power: {n_power}/{len(soc)}',flush=True)
    if n_phys:
        raise ValueError(f'{base}: baseline S3 SOC OUTSIDE PHYSICAL 20-80% band '
            f'at {n_phys} slots (max {physical_excursion.max():.6f} MWh). '
            'Stop: inspect source dispatch and battery configuration; cannot publish recovery rates.')
    return base_tab,exposure,soc,lower,upper,hc,hd


def activations(samples,ex):
    f=num(samples,'fcr_activation_fraction')[:,None]
    a=num(samples,'afrr_activation_fraction')[:,None]
    m=num(samples,'mfrr_activation_fraction')[:,None]
    sh=num(samples,'fcr_up_share')[:,None]
    fu=f*sh*ex['fcr'][None,:]
    fd=f*(1-sh)*ex['fcr'][None,:]
    au=a*ex['afrr_up'][None,:]
    ad=a*ex['afrr_down'][None,:]
    mu=m*ex['mfrr_up'][None,:]
    md=m*ex['mfrr_down'][None,:]
    return fu,fd,au,ad,mu,md


def validate_monthly(samples,updown):
    cols=['fcr_activated_up_mwh','fcr_activated_down_mwh',
          'afrr_activated_up_mwh','afrr_activated_down_mwh',
          'mfrr_activated_up_mwh','mfrr_activated_down_mwh']
    err=[]
    for col,arr in zip(cols,updown):
        stored=num(samples,col)
        estimated=arr.sum(axis=1)
        err.append(np.abs(stored-estimated).max(initial=0))
    # Recovery totals must reproduce the original product-by-product direction netting.
    buys=np.zeros(len(samples)); sells=np.zeros(len(samples))
    for i in (0,2,4):
        drift=ETA_C*updown[i+1]-updown[i]/ETA_D
        buys += (np.maximum(0,-drift)/ETA_C).sum(axis=1)
        sells += (np.maximum(0,drift)*ETA_D).sum(axis=1)
    err.append(np.abs(buys-num(samples,'recovery_buy_mwh')).max(initial=0))
    err.append(np.abs(sells-num(samples,'recovery_sell_mwh')).max(initial=0))
    worst=max(err)
    if worst > 1e-3:
        raise ValueError(f'Energy reconstruction mismatch: max MWh error {worst:.6f}. '
            'Stage-2 generator may use a different flag/interval algorithm. No physical rates calculated.')
    return worst


def screen(samples,tab,ex,soc,low,high,hc,hd):
    p=activations(samples,ex)
    worst_err=validate_monthly(samples,p)
    up=p[0]+p[2]+p[4]; down=p[1]+p[3]+p[5]
    drift=ETA_C*down-up/ETA_D
    n,ntime=drift.shape
    # A no-recovery-at-this-interval headroom screen (strictly NOT a full feasibility test).
    immediate=soc[None,:]+drift
    instant_bad=(immediate<low[None,:]-EPS)|(immediate>high[None,:]+EPS)
    instant_n=instant_bad.sum(axis=1)
    # A conservative chronological recovery policy, starting the next interval.
    delta=np.zeros(n)
    breach_count=np.zeros(n,dtype=np.int32)
    max_excursion=np.zeros(n)
    recovered_buy=np.zeros(n)
    recovered_sell=np.zeros(n)
    day_end_resid=[]
    times=tab['_ts'].to_numpy()
    for t in range(ntime):
        # Previous activation energy can be restored using currently unused power.
        state=soc[t]+delta
        charging=np.minimum.reduce((np.maximum(-delta,0),
                                    np.full(n,hc[t]*DT*ETA_C),
                                    np.maximum(high[t]-state,0)))
        discharging=np.minimum.reduce((np.maximum(delta,0),
                                       np.full(n,hd[t]*DT/ETA_D),
                                       np.maximum(state-low[t],0)))
        delta+=charging-discharging
        recovered_buy+=charging/ETA_C
        recovered_sell+=discharging*ETA_D
        # Activate reserved energy for this 15-minute interval.
        delta+=drift[:,t]
        state=soc[t]+delta
        gap=np.maximum(low[t]-state,0)+np.maximum(state-high[t],0)
        breach_count+=(gap>EPS)
        max_excursion=np.maximum(max_excursion,gap)
        if t==ntime-1 or times[t].astype('datetime64[D]')!=times[t+1].astype('datetime64[D]'):
            day_end_resid.append(np.abs(delta.copy()))
    daily=np.array(day_end_resid).T
    return pd.DataFrame({
        'base_run':samples['base_run'].values,
        'sample_id':samples['sample_id'].values,
        'instant_headroom_screen_any':instant_n>0,
        'instant_headroom_screen_intervals':instant_n,
        'greedy_policy_soc_breach_any':breach_count>0,
        'greedy_policy_soc_breach_intervals':breach_count,
        'max_greedy_soc_excursion_mwh':max_excursion,
        'end_month_unrestored_soc_mwh':np.abs(delta),
        'maximum_day_end_unrestored_soc_mwh':daily.max(axis=1),
        'greedy_recovery_buy_mwh':recovered_buy,
        'greedy_recovery_sell_mwh':recovered_sell,
        'reconstruction_max_abs_error_mwh':np.full(n,worst_err),
    })


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--n-base',type=int,default=120)
    a.add_argument('--samples',type=int,default=1000,help='samples per base and regime; default all 1000')
    a.add_argument('--chunk',type=int,default=100)
    a.add_argument('--selected-ids-file',type=Path,help='optional text file: one exact base_run ID per line (preferred for paper)')
    args=a.parse_args()
    root=args.root;stage=root/'Results/Stage2'
    manifest=json.loads((stage/'stage2_manifest.json').read_text())
    if args.selected_ids_file:
        names=[x.strip() for x in args.selected_ids_file.read_text().splitlines() if x.strip() and not x.startswith('#')]
        selection='explicit_user_selected_ids'
    else:
        names=list(manifest['base_runs'])[:args.n_base]
        selection='FIRST_N_IN_STAGE2_MANIFEST__NOT_INDEPENDENTLY_VERIFIED_AS_CONVERGENCE_SELECTION'
        print(f'WARNING: Using first {len(names)} manifest entries for this run; these are NOT yet independently verified as the manuscript convergence-selected base IDs.')
    if len(names)!=len(set(names)): raise ValueError('Selected run IDs duplicated')
    if not set(names).issubset(set(manifest['base_runs'])):raise ValueError('Selected IDs not in Stage-2 manifest')
    if not names: raise ValueError('Empty selected run list')
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'base_run':names}).to_csv(args.out/'selected_base_runs_USED_FOR_AUDIT.csv',index=False)
    per=[]
    baseline_diag=[]
    for regime in ('reference','stress'):
        path=stage/regime/'all_random_activation_samples.csv'
        samples=pd.read_csv(path,usecols=REQUIRED_SAMPLES)
        if samples.duplicated(['base_run','sample_id']).any():raise ValueError('Duplicate Stage-2 sample IDs')
        actual_counts=samples.groupby('base_run').size()
        missing=[name for name in names if actual_counts.get(name,0)<args.samples]
        if missing: raise ValueError(f'Stage-2 missing {len(missing)} selected runs; e.g. {missing[:3]}')
        samples=samples[samples['base_run'].isin(names)]
        samples=samples.sort_values(['base_run','sample_id'])
        if args.samples<1000:samples=samples.groupby('base_run',group_keys=False).head(args.samples)
        outfile=args.out/f'{regime}_soc_screening_samples.csv'
        if outfile.exists():outfile.unlink()
        header=True
        for idx,name in enumerate(names,1):
            df=samples.loc[samples.base_run==name]
            if not len(df):raise ValueError(f'Missing: {name}')
            tab,ex,soc,low,high,hc,hd=load_interval_inputs(root,name)
            if regime=='reference':
                baseline_diag.append({
                    'base_run':name,'n_intervals':len(tab),
                    'n_baseline_outside_reserved_soc':int((tab['_baseline_outside_reserved_mwh']>1e-4).sum()),
                    'max_baseline_reserved_soc_excursion_mwh':float(tab['_baseline_outside_reserved_mwh'].max()),
                    'n_baseline_outside_physical_soc':int((tab['_baseline_outside_physical_mwh']>1e-4).sum()),
                    'n_baseline_power_overschedule':int(((tab['_baseline_power_overschedule_charge_mw']>1e-4) | (tab['_baseline_power_overschedule_discharge_mw']>1e-4)).sum()),
                    'max_baseline_power_overschedule_mw':float(max(tab['_baseline_power_overschedule_charge_mw'].max(),tab['_baseline_power_overschedule_discharge_mw'].max()))
                })
            for st in range(0,len(df),args.chunk):
                out=screen(df.iloc[st:st+args.chunk],tab,ex,soc,low,high,hc,hd)
                out.to_csv(outfile,mode='a',header=header,index=False)
                header=False
            print(f'{regime}: {idx}/{len(names)} {name}',flush=True)
        analyzed=pd.read_csv(outfile)
        per.append({'regime':regime,'n_base':len(names),'n_samples':len(analyzed),
                    'instant_headroom_screen_pct':100*analyzed.instant_headroom_screen_any.mean(),
                    'greedy_policy_any_soc_breach_pct':100*analyzed.greedy_policy_soc_breach_any.mean(),
                    'greedy_policy_mean_max_excursion_mwh':analyzed.max_greedy_soc_excursion_mwh.mean(),
                    'mean_end_month_unrestored_soc_mwh':analyzed.end_month_unrestored_soc_mwh.mean(),
                    'max_reconstruction_error_mwh':analyzed.reconstruction_max_abs_error_mwh.max()})
    pd.DataFrame(baseline_diag).to_csv(args.out/'baseline_S3_reservation_consistency.csv',index=False)
    summary=pd.DataFrame(per)
    summary.to_csv(args.out/'soc_screening_summary.csv',index=False)
    (args.out/'audit_method_and_selection.json').write_text(json.dumps({
        'method':'ex-post approximate 15-minute reconstruction with greedy next-slot recovery; SOC tested against physical 20-80% battery limits, not already-tightened post-reservation dispatch envelopes',
        'selection':selection,'n_base':len(names),'samples_per_base_and_regime':args.samples,
        'parameters':{'eta_charge':ETA_C,'eta_discharge':ETA_D,'battery_energy_mwh':ENERGY_MWH,'dt_h':DT},
        'warning':'Not observed physical infeasibility. Not an optimized recourse or actual activation replay. These are conditional screens under synthetic temporal allocation and conservative greedy recovery. Monthly totals are validated against original Stage-2 sample energy. IMPORTANT: baseline post-reservation SOC and dispatch-power inconsistencies are reported separately and require evaluation prior to publication. Physical limits 20-80% must be confirmed against actual config.',
    },indent=2),encoding='utf-8')
    print('\n'+summary.to_string(index=False))
    print('\nBASELINE DISPATCH / RESERVATION CONSISTENCY:')
    print(pd.DataFrame(baseline_diag).to_string(index=False))
    print('\nWROTE:',args.out)

if __name__=='__main__': main()
