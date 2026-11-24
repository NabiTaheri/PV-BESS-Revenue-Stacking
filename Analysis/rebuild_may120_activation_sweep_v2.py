#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recalculate a May-2026 2D activation-factor sweep using exact selected IDs.

Reads ONLY the exact 120 convergence-selected May bases and the CURRENT,
settlement-aware 1000maker.py in D:/BalticMay2026Final. No MILP rerun.
No edits to inputs. Outputs a new PDF, PNG, grid CSV and provenance JSON.

The sweep fixes FCR upward share at 0.90 and independently varies
phi_FCR and phi_R=phi_aFRR=phi_mFRR between 0 and 1. For each grid point:
   delta = mean over selected bases of [ S3_gross + activation adjustment - S2 ]
Using the SAME evaluate_activation_sample() implementation as Stage 2.

An affine shortcut is used ONLY after numerical checks against direct engine
calls at off-grid points for EACH base. Otherwise the exact direct evaluator
is used at every grid point (slower).

Run from PowerShell:
   python rebuild_may120_activation_sweep.py --check
   python rebuild_may120_activation_sweep.py --run

When the original run directories are outside D:/BalticMay2026Final, pass one or
more --source-root arguments pointing to their parent project folders.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = re.compile(r'^run_(\d+)_seed_(\d+)$')
CAP_REL = Path('results/results_S3_reserve_competition_full_stack/monthly_afrr_mfrr_capacity_results.csv')
S2_REL = Path('results/results_S2_DA_MFRR_energy/monthly_chain_summary.csv')
REGIMES = ('reference', 'stress')
KEYS = ('base_run', 'iteration', 'seed')


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Check source IDs, raw runs and engine; do not calculate')
    mode.add_argument('--run', action='store_true', help='Compute the sweep and save figures')
    parser.add_argument('--selected', type=Path,
                        default=PROJECT_ROOT / 'Results/Convergence/selected_base_scenarios.csv')
    parser.add_argument('--stage2', type=Path,
                        default=PROJECT_ROOT / 'Results/Stage2')
    parser.add_argument('--engine', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--source-root', type=Path, action='append', default=[],
                        help='Can be repeated; search a project directory for original first-level run folders')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'generated/activation_sensitivity')
    parser.add_argument('--points', type=int, default=41, help='Number of points on each activation axis')
    parser.add_argument('--fcr-up-share', type=float, default=0.9, help='Fixed upward FCR fraction')
    parser.add_argument('--expected-bases', type=int, default=120)
    parser.add_argument('--direct', action='store_true', help='Evaluate every grid point directly (slower)')
    return parser.parse_args()


def must_file(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def to_int(series, column):
    values = pd.to_numeric(series, errors='raise').astype('int64')
    if (values < 0).any():
        raise ValueError(f'Invalid {column}')
    return values


def load_selection(args):
    tab = pd.read_csv(must_file(args.selected), encoding='utf-8-sig')
    needed = {'month', *KEYS}
    if not needed.issubset(tab.columns):
        raise ValueError(f'Selection columns missing: {sorted(needed-set(tab.columns))}')
    selected = tab[tab.month.astype(str).eq('May_2026')].copy()
    selected['iteration'] = to_int(selected['iteration'], 'iteration')
    selected['seed'] = to_int(selected['seed'], 'seed')
    selected = selected.sort_values(['iteration', 'seed']).reset_index(drop=True)
    if len(selected) != args.expected_bases:
        raise ValueError(f'Expected {args.expected_bases} selected May bases; found {len(selected)}')
    for c in KEYS:
        if selected[c].duplicated().any():
            raise ValueError(f'Non-unique selected {c}')
    selected['base_run'] = selected['base_run'].astype(str)
    expected = [f'run_{int(r.iteration):04d}_seed_{int(r.seed)}' for r in selected.itertuples()]
    if selected['base_run'].tolist() != expected:
        raise ValueError('Selected canonical run IDs do not match iteration + seed')
    # Valid convergence selections can have gaps in original run iterations:
    # the selected list is the first N VALID BASES, not necessarily run_0001..run_0NNN.
    # Keep exact canonical IDs; never renumber or replace skipped/invalid scenarios.
    iters = selected['iteration'].tolist()
    skipped = iters[-1] - iters[0] + 1 - len(iters)
    print(f'[PASS] Loaded {len(selected)} unique May-selected canonical IDs '
          f'(original iteration range {iters[0]}..{iters[-1]}, gaps={skipped}).', flush=True)
    print(f'[INFO] Selected first: {selected.base_run.iloc[0]}; '
          f'last: {selected.base_run.iloc[-1]}', flush=True)
    return selected


def load_regime_summaries(args, selected):
    data = {}
    for r in REGIMES:
        file = must_file(args.stage2 / r / 'summary_by_base_scenario.csv')
        tab = pd.read_csv(file, encoding='utf-8-sig')
        needed = set(KEYS) | {'s2_total_eur', 's3_gross_no_activation_eur', 's3_net_mean',
                              'paired_delta_mean_eur', 'activation_energy_cashflow_mean',
                              'net_recovery_cost_mean', 'degradation_mean'}
        if not needed.issubset(tab.columns):
            raise ValueError(f'{file} missing columns: {sorted(needed-set(tab.columns))}')
        if tab.base_run.duplicated().any():
            raise ValueError(f'Duplicate canonical run name in {file}')
        tab['iteration'] = to_int(tab.iteration, 'iteration')
        tab['seed'] = to_int(tab.seed, 'seed')
        if tab['iteration'].duplicated().any() or tab['seed'].duplicated().any():
            raise ValueError(f'{file}: duplicated original May iteration or seed')
        # Verify the fixed convergence-selected list matches the FIRST 120 VALID
        # bases in Stage2 chronological/iteration order. Do not silently take an
        # arbitrary set of 120 rows or assume iterations are consecutive.
        actual_first_n = (tab.sort_values(['iteration', 'seed'])
                          .head(args.expected_bases)['base_run'].astype(str).tolist())
        selected_ids = selected['base_run'].tolist()
        if actual_first_n != selected_ids:
            mismatch = next(((a,b) for a,b in zip(actual_first_n,selected_ids) if a != b),None)
            raise ValueError(f'{r}: selected May IDs differ from first '
                             f'{args.expected_bases} valid Stage2 bases in iteration order. '
                             f'First mismatch (Stage2, selected): {mismatch}. '
                             'Check selection-file provenance; refusing to substitute IDs.')
        tab = selected.merge(tab, on='base_run', how='left', validate='one_to_one',
                             indicator=True, suffixes=('_chosen', ''))
        if not tab['_merge'].eq('both').all():
            missing = tab.loc[tab['_merge'] != 'both', 'base_run'].tolist()
            raise ValueError(f'{r} missing selected May IDs: {missing[:8]}')
        tab = tab.drop(columns='_merge')
        for field in ('iteration','seed'):
            if not (tab[field].to_numpy() == tab[f'{field}_chosen'].to_numpy()).all():
                raise ValueError(f'{r} inconsistent {field} for selection')
            tab = tab.drop(columns=f'{field}_chosen')
        for field in ('s2_total_eur','s3_gross_no_activation_eur','s3_net_mean',
                      'paired_delta_mean_eur','activation_energy_cashflow_mean',
                      'net_recovery_cost_mean','degradation_mean'):
            tab[field] = pd.to_numeric(tab[field], errors='raise')
            if not np.isfinite(tab[field]).all():
                raise ValueError(f'{r} has invalid {field}')
        expected = (tab.s3_gross_no_activation_eur + tab.activation_energy_cashflow_mean
                    - tab.net_recovery_cost_mean - tab.degradation_mean)
        if np.max(np.abs(expected - tab.s3_net_mean)) > 1.0:
            raise ValueError(f'{r} Stage 2 settlement accounting fails for selected May bases')
        if np.max(np.abs(tab.s3_net_mean - tab.s2_total_eur - tab.paired_delta_mean_eur)) > .1:
            raise ValueError(f'{r} Stage 2 paired difference accounting fails')
        data[r] = tab.sort_values('iteration').reset_index(drop=True)
    a,b = data['reference'], data['stress']
    for field in KEYS:
        if not a[field].equals(b[field]):
            raise ValueError(f'Reference/Stress selected IDs differ: {field}')
    for field in ('s2_total_eur', 's3_gross_no_activation_eur'):
        if not np.allclose(a[field], b[field], atol=0.05, rtol=0):
            raise ValueError(f'Reference/Stress unpaired first-level output {field}')
    print(f'[PASS] Exactly {len(a)} selected May bases, Reference/Stress paired, new accounting validated.', flush=True)
    return data


def lineage_sources(stage2):
    file = stage2 / 'base_lineage.csv'
    if not file.is_file():
        return {}
    src = pd.read_csv(file, encoding='utf-8-sig')
    mapping = {}
    for row in src.to_dict('records'):
        key = row.get('canonical_base_run') or row.get('base_run')
        if not isinstance(key, str):
            continue
        candidates = []
        for col in ('source_path', 'original_path', 'source_run_path'):
            if isinstance(row.get(col), str) and row[col].strip():
                candidates.append(Path(row[col]))
        source_project = row.get('source_project')
        source_base = row.get('source_base_run')
        if isinstance(source_project, str) and isinstance(source_base, str):
            project = Path(source_project)
            if project.is_absolute():
                candidates.append(project / 'final_results' / source_base)
                candidates.append(project / 'monte_carlo_results2' / source_base)
        mapping[key] = candidates
    return mapping


def enumerate_run_dirs(sources):
    """Prune recursion as soon as a run folder is found: avoid its thousands of daily files."""
    index = defaultdict(list)
    scanned = []
    for root in sources:
        root = root.expanduser()
        if not root.is_dir():
            continue
        scanned.append(str(root))
        for current, dirs, _ in os.walk(root):
            if RUN_ID.fullmatch(Path(current).name):
                # A root may itself be a specific run folder.
                m = RUN_ID.fullmatch(Path(current).name)
                path = Path(current)
                if (path / CAP_REL).is_file() and (path / S2_REL).is_file():
                    index[int(m.group(2))].append(path)
                dirs[:] = []
                continue
            for d in list(dirs):
                m = RUN_ID.fullmatch(d)
                if m:
                    path = Path(current) / d
                    if (path / CAP_REL).is_file() and (path / S2_REL).is_file():
                        index[int(m.group(2))].append(path)
                    dirs.remove(d)
            # Do not traverse known huge unrelated archive/version trees.
            dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', '.venv', 'venv',
                                                      'paper_figures', 'paper_table_outputs_current_runs',
                                                      'random_activation_sensitivity'}]
    return index, scanned


def resolve_source_runs(args, reference):
    # Source roots can be specified explicitly if May first-level runs live elsewhere.
    sources = args.source_root or [args.engine]
    index, scanned = enumerate_run_dirs(sources)
    lineage = lineage_sources(args.stage2)
    result = []
    problems = []
    for row in reference.itertuples():
        candidate = None
        direct_candidates = list(lineage.get(row.base_run, []))
        if hasattr(row, 'source_path') and isinstance(row.source_path, str):
            direct_candidates.insert(0, Path(row.source_path))
        for path in direct_candidates:
            if path.is_dir() and (path / CAP_REL).is_file() and (path / S2_REL).is_file():
                candidate = path
                break
        if candidate is None:
            candidates = list(index.get(int(row.seed), []))
            if len(candidates) > 1:
                # If some paths are mirrored copies, choose only if the science-bearing CSV is byte-identical.
                fingerprints = {}
                for path in candidates:
                    h = hashlib.sha256()
                    for rel in (CAP_REL, S2_REL):
                        with (path / rel).open('rb') as f:
                            for block in iter(lambda: f.read(1024*1024), b''):
                                h.update(block)
                    fingerprints[str(path)] = h.hexdigest()
                if len(set(fingerprints.values())) == 1:
                    candidates.sort(key=lambda p: (len(str(p)), str(p)))
                    candidate = candidates[0]
                else:
                    problems.append(f'Seed {row.seed}: conflicting distinct runs: {candidates[:4]}')
            elif len(candidates) == 1:
                candidate = candidates[0]
        if candidate is None:
            if not any(str(row.seed) in e for e in problems):
                problems.append(f'{row.base_run} seed={row.seed} raw run not found')
            continue
        actual_id = RUN_ID.fullmatch(candidate.name)
        if not actual_id or int(actual_id.group(2)) != int(row.seed):
            problems.append(f'{row.base_run}: source seed mismatch {candidate}')
            continue
        result.append((row, candidate))
    if problems:
        print(f'[INFO] Searched run roots: {scanned}', flush=True)
        print('[INFO] First source-resolution problems:', flush=True)
        for item in problems[:12]:
            print('   ', item, flush=True)
        raise RuntimeError('Cannot identify all 120 original May bases. Re-run with '
                           '--source-root "PATH_TO_ORIGINAL_MAY_PROJECT" (repeat for each batch).')
    print(f'[PASS] Located {len(result)} exact first-level May runs. Source roots: {scanned}', flush=True)
    return result


def load_current_engine(engine):
    file = must_file(engine / '1000maker.py')
    # This script is intentionally NOT compatible with the old cost-only 1000maker.
    text = file.read_text(encoding='utf-8-sig')
    required = ('afrr_activation_energy_cashflow_eur', 'mfrr_activation_energy_cashflow_eur',
                'net_recovery_cost_eur', 'degradation_eur')
    for marker in required:
        if marker not in text:
            raise RuntimeError(f'CURRENT 1000maker.py missing {marker}; refuse old cost-only formula')
    modname = 'may_sweep_settlement_engine'
    spec = importlib.util.spec_from_file_location(modname, file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load {file}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    required_api = ('load_capacity_results', 'attach_settlement_prices',
                    'find_actuals_csv', 'evaluate_activation_sample', 'ActivationSample')
    for name in required_api:
        if not hasattr(mod, name):
            raise RuntimeError(f'Current settlement module lacks {name}')
    required_sample = {'sample_id', 'fcr_activation_fraction',
                       'afrr_activation_fraction','mfrr_activation_fraction', 'fcr_up_share'}
    fields = {f.name for f in dataclasses.fields(mod.ActivationSample)}
    if not required_sample.issubset(fields):
        raise RuntimeError(f'ActivationSample mismatch: {fields}')
    mod.ROOT_DIR = engine.resolve()
    if not (engine / 'Data' / 'settlement_prices.csv').is_file():
        raise FileNotFoundError(f'Current settlement prices missing: {engine / "Data" / "settlement_prices.csv"}')
    return mod


def sample_obj(mod, phi_f, phi_r, up):
    return mod.ActivationSample(sample_id=1, fcr_activation_fraction=float(phi_f),
                                afrr_activation_fraction=float(phi_r),
                                mfrr_activation_fraction=float(phi_r), fcr_up_share=float(up))


def calc_result(mod, cap, phi_f, phi_r, up):
    row = mod.evaluate_activation_sample(cap, sample_obj(mod, phi_f, phi_r, up))
    required = ('net_activation_adjustment_eur', 'net_recovery_cost_eur', 'degradation_eur',
                'afrr_activation_energy_cashflow_eur', 'mfrr_activation_energy_cashflow_eur')
    if any(field not in row for field in required):
        raise RuntimeError('Settlement engine returned no activation-energy accounting. Refuse drawing.')
    cash = float(row['afrr_activation_energy_cashflow_eur']) + float(row['mfrr_activation_energy_cashflow_eur'])
    calc = cash - float(row['net_recovery_cost_eur']) - float(row['degradation_eur'])
    net = float(row['net_activation_adjustment_eur'])
    if abs(calc - net) > 0.01:
        raise RuntimeError(f'Settlement net adjustment mismatch: {calc} != {net}')
    if not np.isfinite(net):
        raise RuntimeError('Nonfinite settlement adjustment')
    return net


def compare_saved_samples(mod, cap, source_run, expected_row, regime, stage2):
    """Audit a saved real activation sample so price/date/algebra mismatches fail loudly."""
    file = stage2 / regime / 'all_random_activation_samples.csv'
    if not file.is_file():
        raise FileNotFoundError(f'Cannot audit sweep engine: Stage2 samples not found: {file}')
    cols = ['base_run', 'sample_id', 'fcr_activation_fraction', 'afrr_activation_fraction',
            'mfrr_activation_fraction', 'fcr_up_share', 's3_activation_adjusted_net_eur']
    header = pd.read_csv(file, nrows=0, encoding='utf-8-sig').columns
    if not set(cols).issubset(header):
        raise RuntimeError(f'Cannot audit Stage2 sample: missing {sorted(set(cols)-set(header))}')
    # Read only the first matching row, in chunks, to keep memory low.
    sample_row = None
    for chunk in pd.read_csv(file, usecols=cols, chunksize=25000, encoding='utf-8-sig'):
        found = chunk.loc[chunk.base_run.astype(str).eq(str(expected_row.base_run))]
        if len(found):
            sample_row = found.iloc[0]
            break
    if sample_row is None:
        raise ValueError(f'Stage2 sample for {expected_row.base_run} not found ({regime})')
    s = mod.ActivationSample(sample_id=int(sample_row.sample_id),
                             fcr_activation_fraction=float(sample_row.fcr_activation_fraction),
                             afrr_activation_fraction=float(sample_row.afrr_activation_fraction),
                             mfrr_activation_fraction=float(sample_row.mfrr_activation_fraction),
                             fcr_up_share=float(sample_row.fcr_up_share))
    values = mod.evaluate_activation_sample(cap, s)
    candidate = float(expected_row.s3_gross_no_activation_eur) + float(values['net_activation_adjustment_eur'])
    actual = float(sample_row.s3_activation_adjusted_net_eur)
    if abs(candidate - actual) > 0.25:
        raise RuntimeError(f'Engine/price mismatch for {expected_row.base_run}, {regime}: '
                           f'new calculation {candidate:.3f}, saved Stage2 {actual:.3f}. '
                           'Do not plot until source/price versions are reconciled.')
    return candidate, actual


def make_sweep(args, selected_runs, mod, reference):
    n = len(selected_runs)
    xgrid = np.linspace(0, 1, args.points)
    ygrid = np.linspace(0, 1, args.points)
    Z = np.zeros((args.points, args.points), dtype=float)
    coeff = []
    linear_everywhere = not args.direct
    # Audit one saved Reference and Stress draw for the first selected base.
    first_row, first_run = selected_runs[0]
    first_cap = mod.attach_settlement_prices(mod.load_capacity_results(first_run / CAP_REL.parent),
                                             mod.find_actuals_csv(first_run))
    for regime in REGIMES:
        calc, saved = compare_saved_samples(mod, first_cap, first_run, first_row, regime, args.stage2)
        print(f'[PASS] May {regime} saved-sample audit: calculated={calc:.3f}; recorded={saved:.3f}', flush=True)
    for k, (row, run) in enumerate(selected_runs, 1):
        cap = first_cap if k == 1 else mod.attach_settlement_prices(
            mod.load_capacity_results(run / CAP_REL.parent), mod.find_actuals_csv(run))
        baseline = float(row.s3_gross_no_activation_eur) - float(row.s2_total_eur)
        a0 = calc_result(mod, cap, 0.0, 0.0, args.fcr_up_share)
        af = calc_result(mod, cap, 1.0, 0.0, args.fcr_up_share) - a0
        ar = calc_result(mod, cap, 0.0, 1.0, args.fcr_up_share) - a0
        for tf, tr in ((.37,.61), (.82,.25), (1.,1.)):
            zcheck = calc_result(mod, cap, tf, tr, args.fcr_up_share)
            zapprox = a0 + tf * af + tr * ar
            if not np.isclose(zcheck, zapprox, atol=.01, rtol=1e-10):
                linear_everywhere = False
        # As an independent accounting check, the zero-activation adjustment should be zero.
        if abs(a0) > .01:
            raise RuntimeError(f'At phi=(0,0) activation adjustment should be zero: {row.base_run} = {a0}')
        coeff.append((baseline+a0,af,ar))
        if k % 20 == 0 or k == n:
            print(f'[CHECK] Evaluated {k}/{n} May bases; affine shortcut valid so far: {linear_everywhere}', flush=True)
    if linear_everywhere:
        coeff = np.array(coeff, dtype=float)
        b, f, r = np.mean(coeff, axis=0)
        xx,yy = np.meshgrid(xgrid,ygrid)
        Z = b + f*xx + r*yy
        method = 'Verified engine-affine sweep (off-grid direct checks for all 120 bases)'
    else:
        print('[NOTICE] Engine is non-affine for some bases; running direct grid evaluations.', flush=True)
        Z = np.zeros((args.points,args.points), dtype=float)
        for k, (row, run) in enumerate(selected_runs, 1):
            cap = mod.attach_settlement_prices(mod.load_capacity_results(run / CAP_REL.parent),
                                               mod.find_actuals_csv(run))
            baseline = float(row.s3_gross_no_activation_eur) - float(row.s2_total_eur)
            for iy,phi_r in enumerate(ygrid):
                for ix,phi_f in enumerate(xgrid):
                    Z[iy,ix] += (baseline+calc_result(mod,cap,phi_f,phi_r,args.fcr_up_share))/n
            if k%10==0 or k==n:
                print(f'[GRID] Completed {k}/{n} bases', flush=True)
        method = 'Direct engine evaluation for every base and grid point'
    if not np.isfinite(Z).all():
        raise RuntimeError('Nonfinite activation sweep surface')
    return xgrid, ygrid, Z, method


def save_outputs(args, selected, reference, xgrid, ygrid, Z, method, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import FuncFormatter
    output = args.output
    if output.exists():
        raise FileExistsError(f'Output exists; refusing to overwrite: {output}')
    output.mkdir(parents=True,exist_ok=False)
    xx,yy=np.meshgrid(xgrid,ygrid)
    grid = pd.DataFrame({'phi_FCR':xx.ravel(), 'phi_R':yy.ravel(),
                         'mean_paired_delta_eur':Z.ravel()})
    grid.to_csv(output/'may120_activation_sweep_grid.csv',index=False)
    # Detailed provenance ensures later journal revisions can reproduce this figure.
    pd.DataFrame([{'base_run':row.base_run,'iteration':int(row.iteration),
                   'seed':int(row.seed),'source_path':str(path)} for row,path in runs]).to_csv(
                       output/'selected_120_run_lineage.csv',index=False)
    meta = {
        'month':'May_2026','n_selected':len(selected),'selected_file':str(args.selected),
        'stage2_root':str(args.stage2),'settlement_engine_file':str(args.engine/'1000maker.py'),
        'settlement_engine_sha256':hashlib.sha256((args.engine/'1000maker.py').read_bytes()).hexdigest(),
        'settlement_prices_sha256':hashlib.sha256((args.engine/'Data'/'settlement_prices.csv').read_bytes()).hexdigest(),
        'phi_FCR_range':[0,1],'phi_R_range':[0,1],'phi_R_definition':'phi_aFRR = phi_mFRR',
        'fcr_up_share':args.fcr_up_share, 'grid_points_per_axis':args.points,
        'method':method,'no_first_level_simulation':True,
        'current_accounting':'S3_net = S3_gross + aFRR/mFRR activation cashflow - net recovery - degradation',
        'zero_boundary_present':bool(float(Z.min()) <= 0 <= float(Z.max())),
        'mean_S2_eur':float(reference.s2_total_eur.mean()),
        'mean_S3_gross_eur':float(reference.s3_gross_no_activation_eur.mean()),
        'mean_gap_at_zero_activation_eur':float(reference.s3_gross_no_activation_eur.mean()-reference.s2_total_eur.mean()),
    }
    (output/'sweep_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.labelsize':11, 'axes.titlesize':12,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(9.5,6.6))
    lo,hi=float(Z.min()),float(Z.max())
    if lo < 0 < hi:
        norm = TwoSlopeNorm(vmin=lo,vcenter=0,vmax=hi)
        levels=np.linspace(lo,0,12).tolist()[:-1]+np.linspace(0,hi,12).tolist()
        cf=ax.contourf(xx,yy,Z,levels=levels,cmap='RdBu_r',norm=norm,extend='neither')
    else:
        cf=ax.contourf(xx,yy,Z,levels=22,cmap='RdBu_r')
    if lo <= 0 <= hi:
        ct=ax.contour(xx,yy,Z,levels=[0],colors='#161616',linewidths=2.2)
        ax.clabel(ct,fmt={0:'Break-even'},fontsize=9,inline=True)
    cb=fig.colorbar(cf,ax=ax,pad=.025,shrink=.95)
    cb.set_label('Mean paired S3-net âˆ’ S2 revenue (EUR/month)')
    cb.formatter=FuncFormatter(lambda v,pos:f'{v/1000:,.0f}k')
    cb.update_ticks()
    ax.set_xlabel(r'FCR activation factor, $\phi_{\mathrm{FCR}}$')
    ax.set_ylabel(r'Common aFRR/mFRR activation factor, $\phi_R$')
    ax.set_title('May 2026 Â· 120 convergence-selected scenarios')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xticks(np.linspace(0,1,6)); ax.set_yticks(np.linspace(0,1,6))
    ax.grid(alpha=.10,color='k',linestyle=':',linewidth=.6)
    fig.text(.5,.02,'FCR upward share fixed at 90%  |  Black contour: mean paired break-even',
             ha='center',va='center',fontsize=9)
    fig.subplots_adjust(left=.11,right=.9,bottom=.14,top=.91)
    png=output/'fig_activation_sweep_May2026_n120.png'
    pdf=output/'fig_activation_sweep_May2026_n120.pdf'
    fig.savefig(png,dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(pdf,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(f'\n[SUCCESS] {len(selected)}-base activation-factor sweep generated',flush=True)
    print('  PDF:',pdf,flush=True)
    print('  PNG:',png,flush=True)
    print('  GRID:',output/'may120_activation_sweep_grid.csv',flush=True)
    print('  MIN GAP:', f'{lo:,.2f} EUR', ' MAX GAP:',f'{hi:,.2f} EUR',flush=True)
    print('  ZERO CONTOUR:', 'YES' if lo<=0<=hi else 'NO',flush=True)


def main():
    args=args_parser()
    if args.points<3 or args.points>201:
        raise ValueError('Grid points must be 3..201')
    if not (0<=args.fcr_up_share<=1):
        raise ValueError('FCR upward share must be between 0 and 1')
    selected=load_selection(args)
    summary=load_regime_summaries(args,selected)
    runs=resolve_source_runs(args,summary['reference'])
    mod=load_current_engine(args.engine)
    print('[PASS] New activation cashflow settlement engine loaded:',
          args.engine/'1000maker.py',flush=True)
    if args.check:
        print('[CHECK PASSED] Ready for May n=120 sweep. No original files modified.',flush=True)
        return
    if args.output.exists():
        raise FileExistsError(f'Output exists: {args.output}; choose --output NEW_FOLDER')
    xx, yy, Z, method=make_sweep(args,runs,mod,summary['reference'])
    save_outputs(args,selected,summary['reference'],xx,yy,Z,method,runs)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('\n[ERROR]',str(exc),file=sys.stderr,flush=True)
        raise SystemExit(1)
