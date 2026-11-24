#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ex-post reserve-activation SOC reachability (optimistic relaxation, read-only).

This checks whether ANY schedule of incremental recovery charge/discharge,
within post-reservation power headroom, could keep the reconstructed 15-minute
SOC trajectory within the physical 20-80% battery limits.

If the reachable SOC interval becomes empty, infeasibility has been demonstrated
UNDER the specified reconstructed activation trajectory and residual-power model.
If the interval stays nonempty, feasibility is ONLY under an optimistic relaxed
recourse model: this is not a guarantee of physical, market or day-end feasibility.

Same-slot recovery is permitted, simultaneous charging/discharging are relaxed,
and terminal SOC is not imposed except as a separate reporting diagnostic.
This deliberately enlarges the set of admissible controls, making failed checks
harder to obtain. The monthly Stage-2 totals are validated first.

Requires soc_recovery_expost_audit_v3.py in the same directory; that file
provides original output joins and monthly energy validation.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

TOL = 1e-8


def load_v3(script: Path):
    if not script.exists():
        raise FileNotFoundError(f'Missing {script}. Put this v4 script next to soc_recovery_expost_audit_v3.py')
    spec=importlib.util.spec_from_file_location('soc_v3_input',script)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reachable(drift, soc, low, high, charge_internal, discharge_internal, times):
    """Vectorized across Stage-2 samples; chronological recurrence over intervals.

    At time t, deviation delta is allowed to vary within [L, U]. The *relaxed*
    control can be any net internal-energy change from -discharge_internal[t]
    to +charge_internal[t]. After activation the deviation must lie in
    [low[t]-soc[t], high[t]-soc[t]]. Feasible sets are closed intervals.
    """
    n, nt = drift.shape
    l = np.zeros(n, dtype=float)
    u = np.zeros(n, dtype=float)
    failed = np.zeros(n, dtype=bool)
    first_fail = np.full(n,-1,dtype=int)
    first_gap = np.zeros(n,dtype=float)
    first_cause=np.full(n,'',dtype=object)
    for t in range(nt):
        active=~failed
        if not active.any():
            break
        lo_new=l + drift[:,t]-discharge_internal[t]
        hi_new=u + drift[:,t]+charge_internal[t]
        permitted_low=low[t]-soc[t]
        permitted_high=high[t]-soc[t]
        new_l=np.maximum(lo_new,permitted_low)
        new_u=np.minimum(hi_new,permitted_high)
        bad=active & (new_l>new_u+TOL)
        first_fail[bad]=t
        below=hi_new[bad]<permitted_low
        first_gap[bad]=np.maximum(permitted_low-hi_new[bad],lo_new[bad]-permitted_high)
        first_cause[bad]=np.where(below,'below_minimum_SOC','above_maximum_SOC')
        failed |= bad
        live=~failed
        l[live]=new_l[live]
        u[live]=new_u[live]
    restore=np.where(~failed,(l<=TOL)&(u>=-TOL),False)
    return pd.DataFrame({
        'reachability_infeasible_under_relaxation':failed,
        'first_infeasible_interval':first_fail,
        'first_infeasible_timestamp':[str(times[i]) if i>=0 else '' for i in first_fail],
        'first_reachable_gap_mwh':first_gap,
        'first_failure_direction':first_cause,
        'relaxed_end_month_zero_deviation_reachable':restore,
        'reachable_delta_min_at_end_mwh':np.where(failed,np.nan,l),
        'reachable_delta_max_at_end_mwh':np.where(failed,np.nan,u),
    })


def read_selected(path: Path):
    if path.suffix.lower()=='.csv':
        df=pd.read_csv(path)
        if 'base_run' not in df.columns:
            raise ValueError('Selected IDs CSV must have column base_run')
        return df['base_run'].astype(str).str.strip().tolist()
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def load_samples(stage:Path, regime: str, names: list[str], n_samples:int, required:list[str]):
    path=stage/regime/'all_random_activation_samples.csv'
    wanted=set(names)
    parts=[]
    for block in pd.read_csv(path,usecols=required,chunksize=20000):
        subset=block[block['base_run'].isin(wanted)]
        if len(subset):parts.append(subset)
    if not parts:raise ValueError(f'{regime}: none of the selected base IDs found')
    allrows=pd.concat(parts,ignore_index=True)
    if allrows.duplicated(['base_run','sample_id']).any():
        raise ValueError(f'{regime}: duplicate (base_run,sample_id) pairs')
    allrows=allrows.sort_values(['base_run','sample_id'])
    out=allrows.groupby('base_run',sort=False).head(n_samples)
    counts=out.groupby('base_run').size()
    bad=[k for k in names if int(counts.get(k,0))!=n_samples]
    if bad:raise ValueError(f'{regime}: wrong sample counts in {bad[:5]}')
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,help='D:/BalticMay2026Final')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--n-base',type=int,default=2)
    ap.add_argument('--samples',type=int,default=20)
    ap.add_argument('--selected-ids-file',type=Path,help='Exact convergence-selected base IDs as .txt or base_run CSV')
    ap.add_argument('--v3',type=Path,default=Path(__file__).with_name('soc_recovery_expost_audit_v3.py'))
    ap.add_argument('--trace',type=Path,help='Optional: test a single V3 diagnostic trace without root')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.trace:
        d=pd.read_csv(args.trace)
        missing={'activation_soc_change_mwh','baseline_soc_mwh','additional_charge_headroom_mw',
                 'additional_discharge_headroom_mw','timestamp'}-set(d.columns)
        if missing:raise ValueError(f'Missing trace fields {missing}')
        e=d.activation_soc_change_mwh.to_numpy(dtype=float)[None,:]
        c=d.additional_charge_headroom_mw.to_numpy(dtype=float)*.25*.95
        z=d.additional_discharge_headroom_mw.to_numpy(dtype=float)*.25/.95
        soc=d.baseline_soc_mwh.to_numpy(dtype=float)
        r=reachable(e,soc,np.full(len(d),2.),np.full(len(d),8.),c,z,d.timestamp.to_numpy())
        print(r.to_string(index=False))
        r.to_csv(args.out/'trace_reachability.csv',index=False)
        return
    if not args.root:ap.error('Specify --root or --trace')
    if args.samples<1 or args.samples>1000:ap.error('--samples must be 1..1000')
    root=args.root;stage=root/'Results/Stage2'
    manifest=json.loads((stage/'stage2_manifest.json').read_text(encoding='utf-8'))
    selected=args.selected_ids_file is not None
    if selected:
        names=read_selected(args.selected_ids_file)
    else:
        names=list(manifest['base_runs'])[:args.n_base]
        if args.n_base>5:
            raise ValueError('For a production-size run, provide --selected-ids-file with the EXACT convergence-selected base_run IDs.')
        print('PILOT ONLY: base IDs taken from manifest. Not the verified convergence-selected set.')
    if not names or len(names)!=len(set(names)):raise ValueError('Missing or repeated base_run identifiers')
    if not set(names).issubset(set(manifest['base_runs'])):
        raise ValueError('Some selected base IDs do not occur in the Stage-2 manifest')
    mod=load_v3(args.v3)
    data={reg:load_samples(stage,reg,names,args.samples,mod.REQUIRED_SAMPLES)
          for reg in ('reference','stress')}
    out_by_reg={'reference':[],'stress':[]}
    for j,base in enumerate(names,1):
        tab,ex,soc,low,high,hc,hd=mod.load_interval_inputs(root,base)
        times=tab['_ts'].astype(str).to_numpy()
        # Recovery power translated to stored/internal battery energy each step.
        cr=hc*mod.DT*mod.ETA_C
        dr=hd*mod.DT/mod.ETA_D
        for regime in ('reference','stress'):
            df=data[regime].loc[data[regime]['base_run']==base].copy()
            components=mod.activations(df,ex)
            err=mod.validate_monthly(df,components)
            up=components[0]+components[2]+components[4]
            down=components[1]+components[3]+components[5]
            drift=mod.ETA_C*down-up/mod.ETA_D
            rr=reachable(drift,soc,low,high,cr,dr,times)
            rr.insert(0,'sample_id',df.sample_id.to_numpy())
            rr.insert(0,'base_run',base)
            rr.insert(0,'regime',regime)
            rr['monthly_validation_error_mwh']=err
            out_by_reg[regime].append(rr)
        print(f'{j}/{len(names)} {base}: complete',flush=True)
    summaries=[]
    for regime,parts in out_by_reg.items():
        all_results=pd.concat(parts,ignore_index=True)
        all_results.to_csv(args.out/f'{regime}_relaxed_reachability_samples.csv',index=False)
        all_fail=all_results.reachability_infeasible_under_relaxation
        summaries.append({
            'regime':regime,
            'base_count':len(names),
            'sample_count':len(all_results),
            'relaxed_reachability_infeasible_pct':round(100*all_fail.mean(),6),
            'infeasible_samples_count':int(all_fail.sum()),
            'no_empty_reachability_pct':round(100*(~all_fail).mean(),6),
            'terminal_zero_deviation_reachable_pct':round(100*all_results.relaxed_end_month_zero_deviation_reachable.mean(),6),
            'max_monthly_energy_reconstruction_error_mwh':all_results.monthly_validation_error_mwh.max(),
        })
    pd.DataFrame(summaries).to_csv(args.out/'relaxed_reachability_summary.csv',index=False)
    pd.DataFrame({'base_run':names}).to_csv(args.out/'base_runs_audited.csv',index=False)
    (args.out/'interpretation.json').write_text(json.dumps({
        'reconstruction':'Stage-2 monthly activation energy exactly validated against 15-minute accepted reserve exposures',
        'methods':'Optimistic 15-minute interval reachability using fixed S3 baseline dispatch and post-reservation recovery headroom; allows same-slot recovery and continuous signed recourse, and does not impose daily terminal SOC.',
        'failed_case':'No recovery schedule exists within this RELAXED reconstructed model; physical feasibility under real service delivery requires additional time-resolved validation.',
        'passed_case':'Only feasible in the relaxed reachability model; NOT verified physically feasible or cost-neutral.',
        'selection':'explicit convergence-selected IDs' if selected else 'pilot unverified manifest IDs',
        'warning':'S2/S3 conditional activation factors are distributed uniformly over intervals with accepted capacity. The source Stage-2 data do not store realised activation timestamps. Report conditional model-based rates, never measured or real-world service infeasibility rates.',
    },indent=2),encoding='utf-8')
    print('\n'+pd.DataFrame(summaries).to_string(index=False))
    print('\nWROTE:',args.out)

if __name__=='__main__':
    main()
