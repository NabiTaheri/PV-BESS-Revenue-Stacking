#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only sensitivity analysis: how recovery-SOC reachability depends on the
interpretation of reserve-locked power. Uses original S3 and Stage-2 results.

This is an exploratory validation of the V4 conditional rate, NOT a certified
physical feasibility analysis. Every trajectory assumes constant monthly
activation factors applied at every reserved 15-minute interval.

Pilot:
 python soc_recovery_headroom_sensitivity_v5.py --root D:/BalticMay2026Final --out D:/Baltic_SOC_Audit/pilot_v5 --n-base 2 --samples 20
Full (only after pilot checks):
 python soc_recovery_headroom_sensitivity_v5.py --root D:/BalticMay2026Final --out D:/Baltic_SOC_Audit/full_v5 --selected-ids-file D:/Baltic_SOC_Audit/may120_selected_base_runs.csv --samples 1000
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd

ROOT_SCRIPT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT_SCRIPT))
import soc_recovery_expost_audit_v3 as base3
from soc_recovery_reachability_v4 import reachable,load_samples,read_selected

RESERVE_FIELDS=(
 'fcr_recovery_charge_locked_mw',
 'afrr_mfrr_recovery_charge_locked_mw',
 'fcr_recovery_discharge_locked_mw',
 'afrr_mfrr_recovery_discharge_locked_mw',
)


def compute(root: Path, out: Path, ids: list[str], sample_count: int):
 out.mkdir(parents=True,exist_ok=True)
 stage=root/'Results/Stage2'
 data={reg:load_samples(stage,reg,ids,sample_count,base3.REQUIRED_SAMPLES) for reg in ('reference','stress')}
 sums={}
 for reg in ('reference','stress'):
  sums[reg]={key:{'infeasible':0,'zero_deviation':0} for key in ('remaining_only_V4','remaining_plus_recovery_locks','full_rated_power_minus_DA','full_rated_power_ignoring_DA')}
  sums[reg]['reconstruction_max_abs_mwh']=0.
 detail=[]
 for j,id in enumerate(ids,1):
  tbl,expo,soc,lower,upper,hc,hd=base3.load_interval_inputs(root,id)
  # Stage-1 accepted-capacity report can provide separately locked RECOVERY MW.
  # Releasing them is hypothetical, not an assertion that those contracts allow it.
  for k in RESERVE_FIELDS:
   if k not in tbl.columns:
    raise ValueError(f'{id}: {k} absent; recovery lock sensitivity cannot be assessed')
  times=tbl['_ts'].astype(str).to_numpy()
  scheduled_charge=(base3.num(tbl,'battery_grid_charge_mw')+base3.num(tbl,'battery_pv_charge_mw'))
  scheduled_discharge=base3.num(tbl,'battery_discharge_mw')
  free_charge=base3.num(tbl,'p_charge_max_mw')
  free_discharge=base3.num(tbl,'p_discharge_max_mw')
  recovery_charge_lock=base3.num(tbl,'fcr_recovery_charge_locked_mw')+base3.num(tbl,'afrr_mfrr_recovery_charge_locked_mw')
  recovery_discharge_lock=base3.num(tbl,'fcr_recovery_discharge_locked_mw')+base3.num(tbl,'afrr_mfrr_recovery_discharge_locked_mw')
  if min(recovery_charge_lock.min(),recovery_discharge_lock.min())<-1e-7:
   raise ValueError(f'{id}: negative reserved-recovery values')
  p=5.
  capacity={
   'remaining_only_V4':(hc,hd),
   'remaining_plus_recovery_locks':(
      np.minimum(np.maximum(0,free_charge+recovery_charge_lock-scheduled_charge),np.maximum(0,p-scheduled_charge)),
      np.minimum(np.maximum(0,free_discharge+recovery_discharge_lock-scheduled_discharge),np.maximum(0,p-scheduled_discharge))),
   'full_rated_power_minus_DA':(
      np.maximum(0,p-scheduled_charge),np.maximum(0,p-scheduled_discharge)),
   'full_rated_power_ignoring_DA':(np.full(len(hc),p),np.full(len(hc),p)),
  }
  if not (np.all(np.isfinite(recovery_charge_lock)) and np.all(np.isfinite(recovery_discharge_lock))):
   raise ValueError(f'{id}: recovery lock values not finite')
  for regime in ('reference','stress'):
   df=data[regime].loc[data[regime].base_run==id]
   activ=base3.activations(df,expo)
   err=base3.validate_monthly(df,activ)
   sums[regime]['reconstruction_max_abs_mwh']=max(sums[regime]['reconstruction_max_abs_mwh'],err)
   up=activ[0]+activ[2]+activ[4];down=activ[1]+activ[3]+activ[5]
   drift=base3.ETA_C*down-up/base3.ETA_D
   for case,(pc,pdis) in capacity.items():
    charge_internal=pc*base3.DT*base3.ETA_C
    discharge_internal=pdis*base3.DT/base3.ETA_D
    result=reachable(drift,soc,lower,upper,charge_internal,discharge_internal,times)
    inf=int(result['reachability_infeasible_under_relaxation'].sum());good=int(result['relaxed_end_month_zero_deviation_reachable'].sum())
    sums[regime][case]['infeasible']+=inf
    sums[regime][case]['zero_deviation']+=good
    detail.append({'base_run':id,'regime':regime,'scenario':case,'n_samples':len(df),'infeasible':inf,
                   'terminal_zero_deviation_reachable':good,
                   'mean_available_charge_mw':float(pc.mean()),'total_available_charge_internal_mwh':float(charge_internal.sum())})
  print(f'{j}/{len(ids)} {id}',flush=True)
 n=len(ids)*sample_count
 results=[]
 for regime in ('reference','stress'):
  for case,vals in sums[regime].items():
   if case=='reconstruction_max_abs_mwh':continue
   results.append({'regime':regime,'headroom_interpretation':case,'base_count':len(ids),'sample_count':n,
                   'infeasible_samples':vals['infeasible'],'infeasible_pct':100*vals['infeasible']/n,
                   'terminal_zero_reachable_pct':100*vals['zero_deviation']/n,
                   'max_monthly_reconstruction_error_mwh':sums[regime]['reconstruction_max_abs_mwh']})
 result=pd.DataFrame(results)
 result.to_csv(out/'headroom_sensitivity_summary.csv',index=False)
 pd.DataFrame(detail).to_csv(out/'headroom_sensitivity_by_base.csv',index=False)
 (out/'interpretation.json').write_text(json.dumps({
   'scope':'Counterfactual headroom sensitivity of reconstructed constant-fraction 15-minute activation profiles; read-only.',
   'remaining_only_V4':'Uses only residual reservation power minus scheduled DA charge/discharge, as in V4.',
   'remaining_plus_recovery_locks':'Hypothetically releases MW marked for reserve recovery as part of accessible SOC-restoration capability. This may NOT reflect actual rules or scheduling.',
   'full_rated_power_minus_DA':'Ignores reserve locks while keeping baseline DA power scheduled; optimistic physical-power upper bound, not a market-feasible schedule.',
   'full_rated_power_ignoring_DA':'Extreme upper bound; ignores concurrent scheduled DA consumption and reserved capacity, not physically simultaneous.',
   'caution':'Differences among cases mean V4 rates cannot be asserted as actual physical infeasibility. No new S3-net or penalties computed.'
 },indent=2),encoding='utf-8')
 print(result.to_string(index=False));print('WROTE',out)


def main():
 pa=argparse.ArgumentParser(description=__doc__)
 pa.add_argument('--root',type=Path,required=True)
 pa.add_argument('--out',type=Path,required=True)
 pa.add_argument('--samples',type=int,default=20)
 pa.add_argument('--n-base',type=int,default=2)
 pa.add_argument('--selected-ids-file',type=Path)
 args=pa.parse_args()
 if args.samples<1 or args.samples>1000:pa.error('--samples must be 1..1000')
 stage=args.root/'Results/Stage2'
 man=json.loads((stage/'stage2_manifest.json').read_text())
 if args.selected_ids_file:ids=read_selected(args.selected_ids_file)
 else:
  if args.n_base>5:pa.error('For full-size audit provide --selected-ids-file')
  ids=man['base_runs'][:args.n_base]
  print('PILOT ONLY: first manifest base IDs; verify the exact selection before paper reporting')
 if len(ids)!=len(set(ids)) or not ids or not set(ids).issubset(man['base_runs']):
  raise ValueError('Invalid/missing base_run IDs')
 compute(args.root,args.out,ids,args.samples)

if __name__=='__main__':main()
