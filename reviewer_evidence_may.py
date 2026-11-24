#!/usr/bin/env python3
"""Supplementary, reproducible May-2026 reviewer evidence from completed exact base runs.

Outputs empirical input/forecast correlation diagnostics, activation intensity x
FCR direction, activation-degradation cost sensitivity, time-ordered SOC recovery
feasibility audits (diagnostic only), and convergence visualizations. All reported
S1/S2/S3 schedules are produced by the exact model in run_may_exact.py.
"""
from __future__ import annotations
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from fast_evaluator import vectorized_base, original_module

ROOT = Path(__file__).resolve().parent


def _pair_corr(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    good=np.isfinite(x)&np.isfinite(y)
    if good.sum()<3 or np.std(x[good])<1e-10 or np.std(y[good])<1e-10:return float('nan')
    return float(np.corrcoef(x[good],y[good])[0,1])


def correlation_evidence(out,dirs):
    actual=pd.read_csv(ROOT/'Data/actuals.csv')
    pairs=[('imbalance_price_actual','balancing_price','imbalance_price_forecast','balancing_price_forecast'),
           ('da_clearing_price','imbalance_price_actual','da_price_forecast','imbalance_price_forecast'),
           ('afrr_capacity_price_up','mfrr_capacity_price_up','afrr_capacity_price_up_forecast','mfrr_capacity_price_up_forecast'),
           ('pv_actual_mw','da_clearing_price','pv_forecast_mw','da_price_forecast')]
    lag=[('da_clearing_price','da_price_forecast'),('balancing_price','balancing_price_forecast'),
         ('imbalance_price_actual','imbalance_price_forecast'),('pv_actual_mw','pv_forecast_mw')]
    rows=[]
    for dr in dirs:
        fc=pd.read_csv(dr/'data/forecasts_rolling.csv')
        if len(fc)!=len(actual):raise ValueError('Forecast correlation evidence has wrong number of rows')
        for a,b,c,d in pairs:
            rows.append({'base_run':dr.name,'metric':f'{a} vs {b}','kind':'cross_market',
                         'observed_corr':_pair_corr(actual[a],actual[b]),'forecast_corr':_pair_corr(fc[c],fc[d])})
        for a,c in lag:
            rows.append({'base_run':dr.name,'metric':f'{a} lag1','kind':'temporal',
                         'observed_corr':_pair_corr(actual[a].iloc[:-1],actual[a].iloc[1:]),
                         'forecast_corr':_pair_corr(fc[c].iloc[:-1],fc[c].iloc[1:])})
    df=pd.DataFrame(rows);df.to_csv(out/'forecast_correlations_per_base.csv',index=False)
    summary=df.groupby(['kind','metric'],as_index=False).agg(observed_corr=('observed_corr','mean'),
                      mean_simulated_forecast_corr=('forecast_corr','mean'),
                      p05_simulated_corr=('forecast_corr',lambda z:z.quantile(.05)),
                      p95_simulated_corr=('forecast_corr',lambda z:z.quantile(.95)))
    summary.to_csv(out/'forecast_correlation_summary.csv',index=False)
    print('[EVIDENCE] Cross-market and temporal forecast diagnostics written',flush=True)


def postprocess_convergence(out,results):
    history=results/'convergence_history.csv'
    if history.is_file() and history.stat().st_size:
        hist=pd.read_csv(history)
        if len(hist):
            fig,ax=plt.subplots(figsize=(9,4.5))
            for case in ('reference','stress'):
                x=hist['n_base'].to_numpy(int);y=hist[f'{case}_gap_mean'].to_numpy(float)
                hw=hist[f'{case}_gap_ci_halfwidth'].to_numpy(float)
                ax.plot(x,y,marker='o',label=case)
                ax.fill_between(x,y-hw,y+hw,alpha=.17)
            ax.axhline(0,color='gray',linewidth=.8)
            ax.set_xlabel('Completed first-level scenarios');ax.set_ylabel('Paired mean S3-net minus S2 (EUR)')
            ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'convergence_paired_gap.png',dpi=220);plt.close(fig)
            fig,ax=plt.subplots(figsize=(9,4.5))
            for case in ('reference','stress'):
                x=hist['n_base'].to_numpy(int); y=hist[f'{case}_prob_mean'].to_numpy(float)
                lo=hist[f'{case}_prob_ci_lower'].to_numpy(float);hi=hist[f'{case}_prob_ci_upper'].to_numpy(float)
                ax.plot(x,y,marker='o',label=case);ax.fill_between(x,lo,hi,alpha=.17)
            ax.set_ylim(0,1);ax.set_xlabel('Completed first-level scenarios')
            ax.set_ylabel('Pr(S3-net >= S2), base-cluster confidence interval')
            ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'convergence_outperformance.png',dpi=220);plt.close(fig)
    print('[EVIDENCE] Convergence charts written when checkpoints exist',flush=True)


def baseline_degradation_sensitivity(out,results):
    """Post-settlement activation throughput sensitivity, holding market dispatch fixed.

    This is NOT a fully re-optimised comparison across battery aging parameters.
    """
    records=[]
    for regime in ('reference','stress'):
        f=results/f'random_activation_{regime}'/'all_random_activation_samples.csv'
        d=pd.read_csv(f,usecols=['base_run','s3_gross_no_activation_eur','s2_total_eur',
                                 'activation_energy_cashflow_eur','net_recovery_cost_eur','throughput_mwh',
                                 's3_activation_adjusted_net_eur'])
        reference_coef=float(os.environ.get('BALTIC_CYCLE_COST_EUR_PER_MWH','25'))
        check=d.s3_gross_no_activation_eur+d.activation_energy_cashflow_eur-d.net_recovery_cost_eur-reference_coef*d.throughput_mwh
        if np.max(np.abs(check-d.s3_activation_adjusted_net_eur))>1e-5:
            raise RuntimeError('Degradation identity check failed; refuse to publish sensitivity')
        for c in (10.,15.,25.,35.,50.):
            d['net']=d.s3_gross_no_activation_eur+d.activation_energy_cashflow_eur-d.net_recovery_cost_eur-c*d.throughput_mwh
            d['gap']=d.net-d.s2_total_eur
            groups=d.groupby('base_run',sort=False)
            basegap=groups['gap'].mean();basewin=groups['gap'].agg(lambda x:float((x>=0).mean()))
            records.append({'regime':regime,'activation_degradation_eur_per_mwh':c,
                            'mean_s3_net_eur':float(d.net.mean()),'mean_paired_gap_eur':float(basegap.mean()),
                            'prob_s3_ge_s2':float(basewin.mean()),'n_base':len(basegap),
                            'treatment':'ex_post_activation_and_recovery_throughput_only_fixed_bids'})
    pd.DataFrame(records).to_csv(out/'degradation_activation_only_sensitivity.csv',index=False)
    print('[EVIDENCE] Degradation coefficient sensitivity written (fixed dispatch)',flush=True)


def beta_blend(v,low,high):
    return (float((1-v)*low[0]+v*high[0]),float((1-v)*low[1]+v*high[1]))


def intensity_direction_grid(out,dirs,n):
    m=original_module();rows=[];basis={}
    weights=(0.,1/3,2/3,1.)
    up_means=(.50,.70,.90)
    for i,dr in enumerate(dirs):
        s3=dr/'results'/m.S3_DIR_NAME
        cap=m.attach_settlement_prices(m.load_capacity_results(s3),m.find_actuals_csv(dr))
        gross=m.sum_column_from_csv(s3/m.S3_DAILY_SUMMARY_NAME,['total_revenue_market_actual_eur'])
        s2=m.sum_column_from_csv(dr/'results'/m.S2_DIR_NAME/m.S2_DAILY_SUMMARY_NAME,['total_revenue_market_actual_eur'])
        if gross is None or s2 is None:raise RuntimeError('Missing S2/S3 totals for intensity grid')
        for iw,w in enumerate(weights):
            shape=tuple(beta_blend(w,ref,stress) for ref,stress in
                        zip(((1,19),(2,38),(3,17)),((10,10),(10,10),(10,10))))
            for jd,u in enumerate(up_means):
                spec=(*shape,(u-.05,u+.05))
                rng=np.random.default_rng(20260622+int(dr.name.split('_seed_')[-1])*1009+41000*iw+900000*jd)
                d=vectorized_base(cap,rng,n,'reference',m,spec=spec)
                gap=gross+d['net_activation_adjustment_eur']-s2
                rows.append({'base_run':dr.name,'intensity_level':iw,'intensity_weight':w,
                             'fcr_up_mean':u,'fcr_beta_alpha':shape[0][0],'fcr_beta_beta':shape[0][1],
                             'afrr_beta_alpha':shape[1][0],'afrr_beta_beta':shape[1][1],
                             'mfrr_beta_alpha':shape[2][0],'mfrr_beta_beta':shape[2][1],
                             'theoretical_fcr_mean':shape[0][0]/sum(shape[0]),
                             'theoretical_afrr_mean':shape[1][0]/sum(shape[1]),
                             'theoretical_mfrr_mean':shape[2][0]/sum(shape[2]),
                             'mean_gap_eur':float(gap.mean()),'win_probability':float((gap>=0).mean()),
                             'n_activation':n})
        if (i+1)%10==0:print(f'[EVIDENCE] intensity-direction grid: {i+1}/{len(dirs)} bases',flush=True)
    raw=pd.DataFrame(rows)
    raw.to_csv(out/'activation_grid_by_base.csv',index=False)
    summary=raw.groupby(['intensity_level','intensity_weight','fcr_up_mean'],as_index=False).agg(
        mean_paired_gap_eur=('mean_gap_eur','mean'),prob_s3_ge_s2=('win_probability','mean'),
        sd_between_bases_eur=('mean_gap_eur','std'),n_base=('base_run','size'),
        fcr_activation_mean=('theoretical_fcr_mean','mean'),afrr_activation_mean=('theoretical_afrr_mean','mean'),
        mfrr_activation_mean=('theoretical_mfrr_mean','mean'))
    summary.to_csv(out/'activation_intensity_direction_grid.csv',index=False)
    for field,figure,title in [('mean_paired_gap_eur','activation_grid_mean_gap.png','Mean paired revenue gap, EUR'),
                               ('prob_s3_ge_s2','activation_grid_win_probability.png','Pr(S3-net >= S2)')]:
        table=summary.pivot(index='intensity_level',columns='fcr_up_mean',values=field)
        fig,ax=plt.subplots(figsize=(7,4.7));img=ax.imshow(table.to_numpy(),aspect='auto')
        ax.set_xticks(range(len(table.columns)),[f'{float(v):.0%}' for v in table.columns])
        ax.set_yticks(range(len(table.index)),['reference','intermediate-1','intermediate-2','stress'])
        ax.set_xlabel('Mean FCR upward share');ax.set_ylabel('Activation intensity distribution')
        ax.set_title(title);fig.colorbar(img,ax=ax)
        fig.tight_layout();fig.savefig(out/figure,dpi=220);plt.close(fig)
    print('[EVIDENCE] Activation intensity/direction grid written',flush=True)


def time_ordered_recovery_audit(out,dirs,max_samples=12):
    """Feasibility diagnostic: greedy recovery of incremental activation SoC drift.

    Uses the fixed accepted S3 schedule and its residual physical battery power;
    pre-existing reserve commitments remain protected. This is a sequential SOC
    stress *audit*, not an optimization of intraday recovery transactions, and
    its results are not added to the headline settlement cashflow.
    """
    m=original_module();eta=.95
    emax=float(os.environ.get('BALTIC_BESS_E_MWH','10'))
    pmax=float(os.environ.get('BALTIC_BESS_P_MW','5'));dt=.25
    soc_lo=.2*emax;soc_hi=.8*emax
    rows=[]
    for dr in dirs:
        s3dir=dr/'results'/m.S3_DIR_NAME
        da=pd.read_csv(s3dir/'monthly_da_after_capacity_report.csv')
        cap=m.attach_settlement_prices(m.load_capacity_results(s3dir),m.find_actuals_csv(dr))
        da['timestamp']=pd.to_datetime(da['timestamp']);cap['timestamp']=pd.to_datetime(cap['timestamp'])
        da=da.sort_values('timestamp').reset_index(drop=True);cap=cap.sort_values('timestamp').reset_index(drop=True)
        if len(da)!=2976 or len(cap)!=2976 or not da.timestamp.equals(cap.timestamp):
            raise RuntimeError(f'DA and reserve trajectories do not align for {dr.name}')
        soc0=da['soc_after_all_mwh'].to_numpy(float)
        base_ch=(da['battery_grid_charge_mw']+da['battery_pv_charge_mw']+da['mfrr_down_activated_mw']).to_numpy(float)
        base_dis=(da['battery_discharge_mw']+da['mfrr_up_activated_mw']).to_numpy(float)
        fcr=m.series_or_zero(cap,['Accepted_FCR_MW','accepted_fcr_mw']).to_numpy(float)
        fu=m.series_or_zero(cap,['Accepted_aFRR_UP_MW','accepted_afrr_up_mw']).to_numpy(float)
        fd=m.series_or_zero(cap,['Accepted_aFRR_DOWN_MW','accepted_afrr_down_mw']).to_numpy(float)
        mu=m.series_or_zero(cap,['Accepted_mFRR_UP_MW','accepted_mfrr_up_mw']).to_numpy(float)
        md=m.series_or_zero(cap,['Accepted_mFRR_DOWN_MW','accepted_mfrr_down_mw']).to_numpy(float)
        flag=pd.to_numeric(cap['balancing_flag_for_activation'],errors='coerce').fillna(0).to_numpy(float)
        fu*=((flag>0)&cap['afrr_up_price_eur_mwh'].notna().to_numpy());fd*=((flag<0)&cap['afrr_down_price_eur_mwh'].notna().to_numpy())
        mu*=((flag>0)&cap['mfrr_up_price_eur_mwh'].notna().to_numpy());md*=((flag<0)&cap['mfrr_down_price_eur_mwh'].notna().to_numpy())
        # Protect contemporaneous accepted reserve deliverability; no additional
        # charging/discharging assumed beyond existing unused power headroom.
        available_ch=np.clip(pmax-base_ch-fcr-fd-md,0,pmax)*dt*eta
        available_dis=np.clip(pmax-base_dis-fcr-fu-mu,0,pmax)*dt/eta
        for case,spec in [('reference',((1,19),(2,38),(3,17),(.45,.55))),
                          ('stress',((10,10),(10,10),(10,10),(.85,.95)))]:
            seed=int(dr.name.rsplit('_seed_',1)[-1]);rng=np.random.default_rng(20260622+seed*1009)
            for j in range(max_samples):
                f=rng.beta(*spec[0]);a=rng.beta(*spec[1]);b=rng.beta(*spec[2]);share=rng.uniform(*spec[3])
                up=dt*(f*fcr*share+a*fu+b*mu)
                down=dt*(f*fcr*(1-share)+a*fd+b*md)
                act_soc=eta*down-up/eta
                dev=0.; maxabs=0.;breaches=0; deficit=0.; rec_buy=0.;rec_sell=0.
                for k in range(len(soc0)):
                    # Recover only pre-existing imbalance, not future activation.
                    if dev<0:
                        q=min(-dev,float(available_ch[k]),max(0,soc_hi-(soc0[k]+dev)))
                        dev+=q;rec_buy+=q/eta
                    elif dev>0:
                        q=min(dev,float(available_dis[k]),max(0,soc0[k]+dev-soc_lo))
                        dev-=q;rec_sell+=q*eta
                    dev+=float(act_soc[k]);post=float(soc0[k]+dev)
                    if post <soc_lo-1e-8 or post>soc_hi+1e-8:
                        breaches+=1;deficit+=max(soc_lo-post,post-soc_hi,0.)
                    maxabs=max(maxabs,abs(dev))
                rows.append({'base_run':dr.name,'regime':case,'sample_id':j+1,
                             'fcr_fraction':f,'afrr_fraction':a,'mfrr_fraction':b,'fcr_up_share':share,
                             'violated_intervals':breaches,'max_abs_soc_deviation_mwh':maxabs,
                             'remaining_soc_deviation_end_mwh':dev,
                             'sum_limit_violation_mwh':deficit,
                             'recovery_grid_buy_mwh':rec_buy,'recovery_grid_sell_mwh':rec_sell,
                             'recovery_policy':'greedy_forward_SOC_audit_only_not_joint_redispatch'})
    df=pd.DataFrame(rows);df.to_csv(out/'time_ordered_soc_recovery_audit_samples.csv',index=False)
    summary=df.groupby('regime',as_index=False).agg(n_samples=('sample_id','size'),
        share_any_soc_violation=('violated_intervals',lambda x:float((x>0).mean())),
        mean_violating_intervals=('violated_intervals','mean'),
        mean_abs_final_soc_deviation_mwh=('remaining_soc_deviation_end_mwh',lambda s:float(s.abs().mean())),
        mean_max_abs_soc_deviation_mwh=('max_abs_soc_deviation_mwh','mean'))
    summary.to_csv(out/'time_ordered_soc_recovery_audit_summary.csv',index=False)
    print('[EVIDENCE] Time-ordered SOC recovery audit written (separate from monetary settlement)',flush=True)


def settlement_comparison(out,results):
    rows=[]
    for case in ('reference','stress'):
        samples=pd.read_csv(results/f'random_activation_{case}'/'all_random_activation_samples.csv')
        for key in ['s3_gross_no_activation_eur','s3_energy_net_eur','s3_reserve_capacity_gross_eur',
                    's2_total_eur','afrr_activation_energy_cashflow_eur','mfrr_activation_energy_cashflow_eur',
                    'net_recovery_cost_eur','degradation_eur','s3_activation_adjusted_net_eur',
                    's3_minus_s2_eur']:
            if key in samples:
                rows.append({'regime':case,'metric':key,'mean_eur':float(samples[key].mean()),
                             'p05_eur':float(samples[key].quantile(.05)),'p95_eur':float(samples[key].quantile(.95))})
    pd.DataFrame(rows).to_csv(out/'settlement_boundary_comparison.csv',index=False)


def generate_evidence(results,dirs,n_samples):
    results=Path(results);out=results/'reviewer_evidence';out.mkdir(parents=True,exist_ok=True)
    correlation_evidence(out,dirs)
    postprocess_convergence(out,results)
    settlement_comparison(out,results)
    baseline_degradation_sensitivity(out,results)
    intensity_direction_grid(out,dirs,n_samples)
    time_ordered_recovery_audit(out,dirs,max_samples=min(12,max(3,n_samples)))
    limits={
       'analysis_period':'May 2026, Estonia only',
       'statistical_unit':'Independent forecast/market-condition base scenario; activation samples nested within scenario',
       'activation_distributions':'Scenario assumptions, not statistically fitted to Baltic activation records',
       'soc_time_order_audit':'Forward-greedy operational audit holding original DA schedule; NOT joint optimal redispatch nor revised monetary settlement',
       'degradation_sensitivity':'Ex-post incremental activation/recovery throughput at fixed dispatch; not a reoptimized battery aging model',
       'no_synthetic_claims':'Results generated here do not establish other-season or other-country generalization, actual BSP bid acceptance, or thermal runaway safety'}
    (out/'INTERPRETATION_LIMITS.json').write_text(json.dumps(limits,indent=2),encoding='utf-8')
    print('[EVIDENCE DONE]',out,flush=True)

if __name__=='__main__':
    from run_may_exact import RESULT_DIR,is_complete,scenario_path
    paths=[scenario_path(i) for i in range(1,114) if is_complete(scenario_path(i))]
    if not paths:raise SystemExit('No completed base scenarios available')
    n=1000
    generate_evidence(RESULT_DIR,paths,n)
