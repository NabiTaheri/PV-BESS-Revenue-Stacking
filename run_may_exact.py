#!/usr/bin/env python3
"""Exact May-2026 Estonian market Monte Carlo with parallel workers and sequential confidence checks.

All reserve opportunity-cost calculations solve the original S2 MILP.
A reusable sparse representation and safe clearing operations accelerate the same objective
and constraints; intermediate and final activation assessments use the original settlement
algebra with identical numerical results. No other market model is substituted.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import norm
from fast_evaluator import evaluate_all, self_test

ROOT=Path(__file__).resolve().parent
BASE_SEED=1010
SCRIPTS=('run_monte_carlo.py','run_S1_DA_only.py','run_S2_DA_MFRR_energy.py',
         'run_S3_reserve_competition_full_stack.py')
RESULT_DIR=ROOT/os.environ.get('BALTIC_RESULTS_SUBDIR','final_results')
WORK_DIR=ROOT/('_workspaces_'+os.environ.get('BALTIC_RESULTS_SUBDIR','final_results'))
ACTIVATION_REGIMES=('reference','stress')
STAGE_NAMES=('results_S1_DA_only','results_S2_DA_MFRR_energy',
             'results_S3_reserve_competition_full_stack')


def scenario_path(index):
    return RESULT_DIR/f'run_{index:04d}_seed_{BASE_SEED+index-1}'


def is_complete(p):
    if not ((p/'comparison.csv').is_file() and (p/'scenario_totals.csv').is_file()):return False
    try:
        for item in STAGE_NAMES:
            f=p/'results'/item/'monthly_chain_summary.csv'
            if not f.exists() or len(pd.read_csv(f))!=31:return False
        return len(pd.read_csv(p/'comparison.csv'))==1 and len(pd.read_csv(p/'scenario_totals.csv'))==3
    except Exception:return False


def files_for_worker(w):
    w.mkdir(parents=True,exist_ok=True)
    for d in ('DA','MFRR&AFRR','Data'):
        shutil.copytree(ROOT/d,w/d,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
    for script in SCRIPTS:
        shutil.copy2(ROOT/script,w/script)
    # All worker files are isolated. Legacy child scripts mutate their own DA/Data files.


def one_base(index,keep_failed=False):
    seed=BASE_SEED+index-1
    dest=scenario_path(index)
    if is_complete(dest):return (index,True,'already complete',0.)
    workspace=WORK_DIR/f'run_{index:04d}_seed_{seed}'
    begin=time.perf_counter()
    success=False
    try:
        if workspace.exists():shutil.rmtree(workspace)
        files_for_worker(workspace)
        # BALTIC_WORKER_HIGHS_TIMEOUT_1800_V1
        # Only the temporary worker copy changes; the root model and manifest remain unchanged.
        solver_copy = workspace / 'MFRR&AFRR' / 'fast_exact_oc.py'
        solver_text = solver_copy.read_text(encoding='utf-8')
        prior_limit = "'time_limit':300.0"
        if solver_text.count(prior_limit) != 1:
            raise RuntimeError('Unexpected HiGHS timeout definition in worker copy; refusing unsafe change')
        solver_copy.write_text(solver_text.replace(prior_limit, "'time_limit':1800.0"), encoding='utf-8')
        env=os.environ.copy()
        env.update(BALTIC_OC_METHOD='exact_sparse',BALTIC_N_RUNS='1',BALTIC_BASE_SEED=str(seed),
                   BALTIC_MC_ROOT=str(workspace/'_local'),
                   OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1',PYTHONUTF8='1')
        log=workspace/'worker_output.log'
        with log.open('w',encoding='utf-8',errors='replace') as fout:
            p=subprocess.run([sys.executable,str(workspace/'run_monte_carlo.py')],
                              cwd=workspace,env=env,stdout=fout,stderr=subprocess.STDOUT)
        output=workspace/'_local'/f'run_0001_seed_{seed}'
        if p.returncode!=0 or not (output/'comparison.csv').is_file():
            tail='\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-35:])
            raise RuntimeError(f'Worker seed {seed} failed (exit {p.returncode}):\n{tail}')
        if dest.exists():shutil.rmtree(dest)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(output),str(dest))
        # Worker always uses local iteration=1; restore global ordinal for
        # publication provenance and downstream seed/iteration reconciliation.
        for name in ('comparison.csv','scenario_totals.csv'):
            csvpath=dest/name
            frame=pd.read_csv(csvpath)
            frame['iteration']=index
            if 'iteration_dir' in frame.columns:
                frame['iteration_dir']=str(dest)
            if 'result_dir' in frame.columns:
                frame['result_dir']=frame['result_dir'].map(
                    lambda value: str(dest/'results'/Path(str(value)).name))
            if 'summary_file' in frame.columns:
                frame['summary_file']=frame['summary_file'].map(
                    lambda value: str(dest/'results'/Path(str(value)).parent.name/Path(str(value)).name))
            frame.to_csv(csvpath,index=False)
        if not is_complete(dest):
            raise RuntimeError(f'Incomplete monthly output in {dest}')
        success=True
        return (index,False,'completed',time.perf_counter()-begin)
    finally:
        if success and workspace.exists() and not keep_failed:
            shutil.rmtree(workspace,ignore_errors=True)


def look_points(maximum,minimum,step):
    x=list(range(step,maximum+1,step))
    if maximum not in x:x.append(maximum)
    return [v for v in x if v>=minimum]


def wilson(p,n,z):
    n=max(int(n),1)
    d=1+z*z/n
    center=(p+z*z/(2*n))/d
    r=z*np.sqrt(p*(1-p)/n +z*z/(4*n*n))/d
    return center-r,center+r


def summarize_look(n,bases,alpha_look,args,previous=None):
    """Base-scenario clustering and preallocated-look multiplicity adjustment."""
    tcrit=float(student_t.ppf(1-alpha_look/2,n-1))
    z=float(norm.ppf(1-alpha_look/2))
    record={'n_base':n,'n_activation_interim':args.interim_activation,
            'alpha_per_predeclared_look':alpha_look,'method':'exact_reusable_milp'}
    okay=True
    for case,df in bases.items():
        for name,column in [('s2','s2_total_eur'),('s3net','s3_net_mean'),
                            ('gap','paired_delta_mean_eur'),('prob','prob_s3_net_ge_s2')]:
            x=df[column].to_numpy(dtype=float)
            mean=float(np.mean(x));sd=float(np.std(x,ddof=1));se=sd/np.sqrt(n)
            record[f'{case}_{name}_mean']=mean
            record[f'{case}_{name}_cluster_se']=se
            if name=='prob':
                # Each x[i] is a Monte Carlo conditional probability for one
                # independent first-level base scenario. Estimate its population
                # mean with scenario-clustered SE rather than pooled Bernoulli n*M.
                hw=se*tcrit
                # Guard against an unrealistically zero-width interval at
                # observed all-success/all-failure checkpoints. The binomial
                # Wilson score envelope is a conservative boundary check.
                score_low,score_high=wilson(mean,n,z)
                lo=max(0.,min(mean-hw,score_low))
                hi=min(1.,max(mean+hw,score_high))
                hw=max(mean-lo,hi-mean)
                record[f'{case}_prob_ci_lower']=lo
                record[f'{case}_prob_ci_upper']=hi
                record[f'{case}_prob_ci_worst_halfwidth']=hw
                if hw>args.prob_ci_width:okay=False
            else:
                hw=se*tcrit
                record[f'{case}_{name}_ci_halfwidth']=hw
                if name=='gap' and hw>args.gap_ci_eur:okay=False
                if name=='s3net' and hw>max(args.mean_ci_eur,args.mean_ci_relative*abs(mean)):
                    okay=False
    # A one-off lucky checkpoint must never trigger stopping.
    stable=False
    if previous is not None:
        stable=True
        for case in ACTIVATION_REGIMES:
            for metric in ('s2','s3net','gap','prob'):
                key=f'{case}_{metric}_mean'
                new=float(record[key]);old=float(previous[key])
                tol= args.stable_prob if metric=='prob' else max(
                    args.stable_eur,args.stable_relative*max(abs(new),abs(old)))
                if abs(new-old)>tol:stable=False
    record['ci_gates_passed']=bool(okay)
    record['change_from_prior_look_stable']=bool(stable)
    record['stop_eligible']=bool(okay and stable and n>=args.min_base)
    return record


def interim_check(n,args,alpha_look,previous=None):
    from fast_evaluator import eval_one_run
    bases={}
    for case in ACTIVATION_REGIMES:
        rows=[]
        for i in range(1,n+1):
            _,r=eval_one_run(scenario_path(i),case,args.interim_activation)
            rows.append(r)
        bases[case]=pd.DataFrame(rows).sort_values('iteration').reset_index(drop=True)
    return summarize_look(n,bases,alpha_look,args,previous)


def publication_outputs(n,args):
    dirs=[scenario_path(i) for i in range(1,n+1)]
    # Export old-compatible base structure and aggregated CSVs.
    cmps=[]; totals=[]
    for d in dirs:
        cmps.append(pd.read_csv(d/'comparison.csv'))
        totals.append(pd.read_csv(d/'scenario_totals.csv'))
    pd.concat(cmps,ignore_index=True).to_csv(RESULT_DIR/'all_iterations_comparison.csv',index=False)
    pd.concat(totals,ignore_index=True).to_csv(RESULT_DIR/'all_iterations_long.csv',index=False)
    for case in ACTIVATION_REGIMES:
        out=RESULT_DIR/f'random_activation_{case}'
        bases,data=evaluate_all(dirs,case,args.final_activation,out,quiet=False)
        try:
            import importlib.util
            def load(name,filename):
                spec=importlib.util.spec_from_file_location(name,ROOT/filename)
                mod=importlib.util.module_from_spec(spec)
                sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
            tables=load('tables_'+case,'make_case_study_tables.py')
            tables.MC_ROOT=RESULT_DIR;tables.ACT_DIR=out
            tables.OUT_DIR=RESULT_DIR/f'paper_tables_{case}'
            tables.main()
            plots=load('plots_'+case,'plot_case_study_figures.py')
            plots.MC_ROOT=RESULT_DIR;plots.ACT_DIR=out
            plots.TABLE_DIR=tables.OUT_DIR
            plots.SCENARIO_FILE=tables.OUT_DIR/'scenario_level_inputs_with_s1_s2_s3.csv'
            plots.SAMPLES_FILE=out/'all_random_activation_samples.csv'
            plots.OUT_DIR=RESULT_DIR/f'paper_figures_{case}'
            plots.OUT_DIR.mkdir(exist_ok=True)
            plots.FIG1_PATH=plots.OUT_DIR/'revenue_distributions.png'
            plots.FIG2_PATH=plots.OUT_DIR/'s3_pooled_hist.png'
            plots.main()
        except Exception as exc:
            (RESULT_DIR/f'paper_generator_error_{case}.txt').write_text(traceback.format_exc(),encoding='utf-8')
            raise RuntimeError(f'Required table/figure output failed for {case}: {exc}') from exc
    print(f'[FINAL] Core tables and figures written to {RESULT_DIR}',flush=True)


def check_sources():
    needed=[*SCRIPTS,'fast_evaluator.py','1000maker.py','make_case_study_tables.py',
            'plot_case_study_figures.py','Data/make_forecast.py','Data/actuals.csv',
            'Data/settlement_prices.csv','DA/market_config.py',
            'MFRR&AFRR/afrr_mfrr.py', 'MFRR&AFRR/fast_exact_oc.py']
    for f in needed:
        if not (ROOT/f).exists():raise RuntimeError('Required source missing: '+f)
    data=pd.read_csv(ROOT/'Data/actuals.csv')
    prices=pd.read_csv(ROOT/'Data/settlement_prices.csv')
    if len(data)!=2976 or len(prices)!=2976:
        raise RuntimeError(f'Input rows must be 2976; actuals={len(data)}, prices={len(prices)}')
    if 'timestamp' not in data or 'timestamp' not in prices:
        raise RuntimeError('Both input files require timestamp')
    ts=pd.to_datetime(data['timestamp'],errors='raise')
    ps=pd.to_datetime(prices['timestamp'],errors='raise')
    if ts.duplicated().any() or ps.duplicated().any() or not ts.equals(ps):
        raise RuntimeError('Duplicated/misaligned timestamps in actuals and settlement prices')
    if not ts.equals(pd.date_range('2026-05-01',periods=2976,freq='15min').to_series().reset_index(drop=True)):
        raise RuntimeError('Input timestamps are not all 15-minute intervals of May 2026')
    settlement_cols=['afrr_up_price_eur_mwh','afrr_down_price_eur_mwh',
                     'mfrr_up_price_eur_mwh','mfrr_down_price_eur_mwh']
    if any(c not in prices or pd.to_numeric(prices[c],errors='coerce').notna().sum()==0 for c in settlement_cols):
        raise RuntimeError('One or more real aFRR/mFRR settlement price columns are absent or entirely empty')
    power=float(os.environ.get('BALTIC_BESS_P_MW','5'))
    energy=float(os.environ.get('BALTIC_BESS_E_MWH','10'))
    coefficient=float(os.environ.get('BALTIC_CYCLE_COST_EUR_PER_MWH','25'))
    if not (power>=1 and energy>0 and coefficient>=0):
        raise RuntimeError('Invalid storage nameplate or degradation coefficient')
    self_test()
    print(f'[CHECK] 2976 May intervals, settlement prices, exact model sources and activation algebra OK. BESS={power:g} MW/{energy:g} MWh.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-base',type=int,default=113)
    p.add_argument('--min-base',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=20)
    p.add_argument('--interim-activation',type=int,default=1000)
    p.add_argument('--final-activation',type=int,default=1000)
    p.add_argument('--gap-ci-eur',type=float,default=5000.)
    p.add_argument('--mean-ci-eur',type=float,default=5000.)
    p.add_argument('--mean-ci-relative',type=float,default=.05)
    p.add_argument('--prob-ci-width',type=float,default=.08)
    p.add_argument('--stable-relative',type=float,default=.025)
    p.add_argument('--stable-eur',type=float,default=2000.)
    p.add_argument('--stable-prob',type=float,default=.03)
    p.add_argument('--keep-workspaces',action='store_true')
    p.add_argument('--check',action='store_true')
    p.add_argument('--skip-convergence',action='store_true')
    args=p.parse_args()
    if not (1<=args.workers<=16):p.error('workers must be 1..16')
    if not (1<=args.min_base<=args.max_base<=113):p.error('Require 1 <= min <= max <= 113')
    if args.batch_size<1 or args.interim_activation<20 or args.final_activation<20:
        p.error('Invalid batch/sample size')
    check_sources()
    if args.check:return
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    existing_manifest=RESULT_DIR/'fast_run_manifest.json'
    if existing_manifest.exists():
        old=json.loads(existing_manifest.read_text(encoding='utf-8'))
        if old['engine']!='original_S2_MILP_reusable_sparse' or old.get('bess_power_mw') != float(os.environ.get('BALTIC_BESS_P_MW','5')) or old.get('bess_energy_mwh') != float(os.environ.get('BALTIC_BESS_E_MWH','10')) or old.get('degradation_eur_per_mwh')!=float(os.environ.get('BALTIC_CYCLE_COST_EUR_PER_MWH','25')):
            raise RuntimeError('Existing manifest incompatible with required exact model. Use a clean results folder.')
        for f,digest in old.get('input_sha256',{}).items():
            if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Input/model source {f} changed after runs began; cannot merge results with different code or data')
    else:
        manifest={'engine':'original_S2_MILP_reusable_sparse','base_seed_first':BASE_SEED,
                  'base_seed_last':BASE_SEED+112,'max_base_requested':args.max_base,
                  'workers':args.workers,'bess_power_mw':float(os.environ.get('BALTIC_BESS_P_MW','5')),'bess_energy_mwh':float(os.environ.get('BALTIC_BESS_E_MWH','10')),
                  'degradation_eur_per_mwh':float(os.environ.get('BALTIC_CYCLE_COST_EUR_PER_MWH','25')),
                  'activation_engine':'algebraically identical vectorized settlement calculations',
                  'input_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                    ['Data/actuals.csv','Data/settlement_prices.csv','MFRR&AFRR/afrr_mfrr.py', 'MFRR&AFRR/fast_exact_oc.py']},
                  'stopping':'prespecified looks; Bonferroni-corrected cluster t intervals',
                  'time_period':'2026-05-01 to 2026-05-31','location':'Estonia bidding zone'}
        existing_manifest.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    looks=look_points(args.max_base,args.min_base,args.batch_size)
    alpha=.05/max(len(looks),1)
    target=0;records=[];previous=None;stop=False
    history_file=RESULT_DIR/'convergence_history.csv'
    # Recompute checkpoints for existing completed runs to preserve diagnostics on resume.
    for goal in looks:
        todo=[i for i in range(target+1,goal+1) if not is_complete(scenario_path(i))]
        if todo:
            print(f'\n[BATCH] {target+1}..{goal} | {len(todo)} new base scenarios | {args.workers} workers',flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs={pool.submit(one_base,i,args.keep_workspaces):i for i in todo}
                failures=[]
                for future in as_completed(futs):
                    index=futs[future]
                    try:
                        _,_,msg,sec=future.result()
                        print(f'[BASE {index:03d}/{args.max_base}] {msg} ({sec:.1f}s)',flush=True)
                    except Exception as exc:
                        print(f'[ERROR base {index}] {exc}',flush=True)
                        failures.append(index)
                if failures:raise RuntimeError(f'Failed base scenarios {failures}; inspect _exact_workspaces logs or rerun.')
        target=goal
        if not all(is_complete(scenario_path(i)) for i in range(1,goal+1)):
            raise RuntimeError('Missing or incomplete base scenario before convergence')
        if args.skip_convergence:continue
        rec=interim_check(goal,args,alpha,previous)
        records.append(rec)
        pd.DataFrame(records).to_csv(history_file,index=False)
        print(f'[CONVERGENCE n={goal}] REF gap={rec["reference_gap_mean"]:,.0f} ±{rec["reference_gap_ci_halfwidth"]:,.0f}, '
              f'STRESS gap={rec["stress_gap_mean"]:,.0f} ±{rec["stress_gap_ci_halfwidth"]:,.0f}, '
              f'prob REF={rec["reference_prob_mean"]:.3f}, STRESS={rec["stress_prob_mean"]:.3f}; '
              f'stop={rec["stop_eligible"]}',flush=True)
        previous=rec
        if rec['stop_eligible']:
            stop=True
            print('[STOP] Predeclared cluster-aware convergence criteria satisfied.',flush=True)
            break
    if target==0:raise RuntimeError('No bases finished')
    publication_outputs(target,args)
    from reviewer_evidence_may import generate_evidence
    generate_evidence(RESULT_DIR, [scenario_path(i) for i in range(1,target+1)], args.final_activation)
    outcome={'n_base_final':target,'n_activation_per_base':args.final_activation,
             'stopped_by_convergence':stop,'engine':'original_S2_MILP_reusable_sparse',
             'same_reserve_objective_and_constraints':True,
             'revenue_outputs':'Results from THIS engine; do not copy old 113-scenario figures',
             'history_file':'convergence_history.csv',
             'reviewer_evidence_folder':'reviewer_evidence'}
    (RESULT_DIR/'run_summary.json').write_text(json.dumps(outcome,indent=2),encoding='utf-8')
    print('[DONE]',outcome,flush=True)

if __name__=='__main__':main()
