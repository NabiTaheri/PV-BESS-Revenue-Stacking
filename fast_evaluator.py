"""Vectorized, algebraically equivalent second-level activation postprocessor.

Uses five evaluations of the ORIGINAL settlement routine per base scenario.
The remaining samples are evaluated by piecewise-linear interpolation, exact
for the original per-product model (uniform FCR-up share, fixed efficiencies).
An embedded equivalence test checks against the original method.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def original_module():
    spec=importlib.util.spec_from_file_location('baltic_activation_original', ROOT/'1000maker.py')
    mod=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def vectorized_base(cap, rng, n, case, original=None, spec=None):
    m=original or original_module()
    betas=spec if spec is not None else {'reference':((1,19),(2,38),(3,17),(.45,.55)),
           'stress':((10,10),(10,10),(10,10),(.85,.95))}[case]
    f=rng.beta(*betas[0],size=n)
    a=rng.beta(*betas[1],size=n)
    b=rng.beta(*betas[2],size=n)
    s=rng.uniform(*betas[3],size=n)
    m.CHARGE_EFFICIENCY=float(m.CHARGE_EFFICIENCY)
    m.DISCHARGE_EFFICIENCY=float(m.DISCHARGE_EFFICIENCY)
    # A drift has the same sign for every FCR interval because s is shared
    # across intervals. The signed drift changes at one analytic breakpoint.
    eta_c=m.CHARGE_EFFICIENCY; eta_d=m.DISCHARGE_EFFICIENCY
    cut=(eta_c*eta_d)/(1+eta_c*eta_d)
    def E(phi,share,aa=0.,bb=0.):
        return m.evaluate_activation_sample(cap,m.ActivationSample(
            1,float(phi),float(aa),float(bb),float(share)))
    zero=E(0.,0.)
    c0=E(1.,0.)
    cm=E(1.,cut)
    c1=E(1.,1.)
    ca=E(0.,0.,aa=1.)
    cb=E(0.,0.,bb=1.)
    out={}
    ignored={'sample_id','fcr_activation_fraction','afrr_activation_fraction',
             'mfrr_activation_fraction','fcr_up_share'}
    w=np.where(s<=cut,s/max(cut,1e-12),(s-cut)/max(1-cut,1e-12))
    for key in zero:
        if key in ignored: continue
        z=float(zero[key]); y0=float(c0[key]); ym=float(cm[key]); y1=float(c1[key])
        fcr=np.where(s<=cut,y0+(ym-y0)*w,ym+(y1-ym)*w)-z
        out[key]=z + f*fcr + a*(float(ca[key])-z)+b*(float(cb[key])-z)
    out['sample_id']=np.arange(1,n+1,dtype=int)
    out['fcr_activation_fraction']=f
    out['afrr_activation_fraction']=a
    out['mfrr_activation_fraction']=b
    out['fcr_up_share']=s
    df=pd.DataFrame(out)
    return df


def eval_one_run(run_dir,case,n_activation=128,activation_seed=20260622):
    m=original_module()
    iteration,seed=m.parse_iteration_seed(run_dir)
    s3=run_dir/'results'/m.S3_DIR_NAME
    s2=run_dir/'results'/m.S2_DIR_NAME
    cap=m.load_capacity_results(s3)
    cap=m.attach_settlement_prices(cap,m.find_actuals_csv(run_dir))
    s3gross=m.sum_column_from_csv(s3/m.S3_DAILY_SUMMARY_NAME,['total_revenue_market_actual_eur'])
    s3energy=m.sum_column_from_csv(s3/m.S3_DAILY_SUMMARY_NAME,['da_net_cashflow_eur_actual'])
    s3reserve=m.sum_column_from_csv(s3/m.S3_DAILY_SUMMARY_NAME,['reserve_capacity_revenue_gross_eur'])
    s2total=m.sum_column_from_csv(s2/m.S2_DAILY_SUMMARY_NAME,['total_revenue_market_actual_eur'])
    if s3reserve is None and s3gross is not None and s3energy is not None:
        s3reserve = s3gross - s3energy
    if s3gross is None or s2total is None or s3energy is None or s3reserve is None:
        raise RuntimeError('Missing S2/S3 monthly totals: '+str(run_dir))
    rng=np.random.default_rng(activation_seed + seed*1009)
    x=vectorized_base(cap,rng,n_activation,case,m)
    x['base_run']=run_dir.name
    x['iteration']=iteration
    x['seed']=seed
    x['s3_gross_no_activation_eur']=s3gross
    x['s3_energy_net_eur']=s3energy
    x['s3_reserve_capacity_gross_eur']=s3reserve
    x['s2_total_eur']=s2total
    x['s3_activation_adjusted_net_eur']=s3gross+x['net_activation_adjustment_eur']
    x['s3_cost_only_net_eur']=s3gross+x['cost_only_adjustment_eur']
    x['s3_minus_s2_eur']=x['s3_activation_adjusted_net_eur']-s2total
    x['s3_cost_only_minus_s2_eur']=x['s3_cost_only_net_eur']-s2total
    base={'base_run':run_dir.name,'iteration':iteration,'seed':seed,
          's1_total_eur':float(pd.read_csv(run_dir/'comparison.csv')['S1_total_eur'].iloc[0]),
          's2_total_eur':s2total,'s3_gross_no_activation_eur':s3gross,
          's3_energy_net_eur':s3energy,'s3_reserve_capacity_gross_eur':s3reserve}
    for name,column in [('s3_net','s3_activation_adjusted_net_eur'),
                        ('s3_cost_only','s3_cost_only_net_eur'),
                        ('activation_energy_cashflow','activation_energy_cashflow_eur'),
                        ('net_recovery_cost','net_recovery_cost_eur'),
                        ('degradation','degradation_eur')]:
        vals=x[column]
        for k,v in m.summarize_series(vals).items():base[name+'_'+k]=v
    base['prob_s3_net_ge_s2']=float((x['s3_minus_s2_eur']>=0).mean())
    base['prob_s3_cost_only_ge_s2']=float((x['s3_cost_only_minus_s2_eur']>=0).mean())
    base['paired_delta_mean_eur']=float(x['s3_minus_s2_eur'].mean())
    return x,base


def evaluate_all(run_dirs,case,n_activation,output_dir=None,quiet=False):
    sample_frames=[]; base=[]
    for path in sorted(run_dirs):
        x,b=eval_one_run(path,case,n_activation)
        sample_frames.append(x);base.append(b)
        if not quiet: print(f'[ACTIVATION {case}] {path.name} net={b["s3_net_mean"]:,.0f}',flush=True)
    data=pd.concat(sample_frames,ignore_index=True)
    bases=pd.DataFrame(base).sort_values('iteration').reset_index(drop=True)
    if output_dir is not None:
        output_dir.mkdir(parents=True,exist_ok=True)
        data.to_csv(output_dir/'all_random_activation_samples.csv',index=False)
        bases.to_csv(output_dir/'summary_by_base_scenario.csv',index=False)
        m=original_module()
        overall=[]
        for key in ['s3_activation_adjusted_net_eur','s3_cost_only_net_eur','s3_cost_only_minus_s2_eur',
                    'afrr_activation_energy_cashflow_eur','mfrr_activation_energy_cashflow_eur','activation_energy_cashflow_eur',
                    'net_activation_adjustment_eur','net_recovery_cost_eur','recovery_buy_cost_eur',
                    'recovery_sell_revenue_eur','degradation_eur','throughput_mwh','s3_minus_s2_eur']:
            overall.append({'metric':key,**m.summarize_series(data[key])})
        overall.append({'metric':'probability_s3_activation_adjusted_net_ge_s2',
                       'mean':float((data['s3_minus_s2_eur']>=0).mean())})
        pd.DataFrame(overall).to_csv(output_dir/'summary_overall.csv',index=False)
        cfg={'source':'vectorized algebraically equivalent activation calculator',
             'case':case,'n_base':len(bases),'n_activation_per_base':n_activation,
             'seed_formula':'20260622 + base_seed * 1009','prices':'Data/settlement_prices.csv',
             'fcr_energy_cashflow':False,'afrr_mfrr_energy_cashflow':True,
             'opportunity_cost_method': 'set separately in run manifest'}
        (output_dir/'random_activation_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        specs={'reference':[(1,19),(2,38),(3,17),(.45,.55)],'stress':[(10,10),(10,10),(10,10),(.85,.95)]}[case]
        ass=[]
        for product,pair,col in [('FCR',specs[0],'fcr_activation_fraction'),
                                 ('aFRR',specs[1],'afrr_activation_fraction'),
                                 ('mFRR',specs[2],'mfrr_activation_fraction')]:
            ass.append({'market':product,'distribution':'beta','beta_alpha':pair[0],
                        'beta_beta':pair[1],'theoretical_mean':pair[0]/sum(pair),
                        'sample_mean':float(data[col].mean())})
        pd.DataFrame(ass).to_csv(output_dir/'activation_assumptions_summary.csv',index=False)
    return bases,data


def self_test():
    m=original_module()
    rng=np.random.default_rng(7)
    n=14
    cap=pd.DataFrame({'timestamp':pd.date_range('2026-05-01',periods=n,freq='15min'),
      'Accepted_FCR_MW':rng.integers(0,4,n),'Accepted_aFRR_UP_MW':rng.integers(0,4,n),
      'Accepted_aFRR_DOWN_MW':rng.integers(0,4,n),'Accepted_mFRR_UP_MW':rng.integers(0,4,n),
      'Accepted_mFRR_DOWN_MW':rng.integers(0,4,n),'balancing_flag_for_activation':rng.choice([-1,1],n),
      'imbalance_buy_price':rng.uniform(-50,200,n),'imbalance_sell_price':rng.uniform(-80,120,n)})
    for k in ('afrr_up','afrr_down','mfrr_up','mfrr_down'):
        val=rng.uniform(-70,150,n)
        val[rng.integers(0,n,3)]=np.nan
        cap[k+'_price_eur_mwh']=val
    for case in ('reference','stress'):
        f=np.random.default_rng(80)
        x=vectorized_base(cap,f,15,case,m)
        for _,row in x.iterrows():
            sample=m.ActivationSample(int(row['sample_id']),float(row['fcr_activation_fraction']),
                    float(row['afrr_activation_fraction']),float(row['mfrr_activation_fraction']),float(row['fcr_up_share']))
            truth=m.evaluate_activation_sample(cap,sample)
            for key,val in truth.items():
                assert np.isclose(float(row[key]),float(val),atol=1e-7,rtol=1e-9), (case,key,row[key],val)
    print('[PASS] Vectorization matches the original settlement calculator for reference/stress test cases.')

if __name__=='__main__':self_test()
