"""Verify archived May results or explicitly rerun the selected base scenarios."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'Results'
METRICS = ('s1_total_eur', 's2_total_eur', 's3_gross_no_activation_eur',
           's3_net_mean', 'paired_delta_mean_eur', 'prob_s3_net_ge_s2')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def selection():
    selected = rows(RESULTS / 'selected_scenarios.csv')
    require(len(selected) == 120, 'Expected exactly 120 selected scenarios')
    require(len({x['base_run'] for x in selected}) == 120 and
            len({x['seed'] for x in selected}) == 120, 'Duplicate scenario or seed')
    for r in selected:
        i, seed = int(r['iteration']), int(r['seed'])
        require(seed == 1010 + i - 1 and r['base_run'] == f'run_{i:04d}_seed_{seed}',
                f'Unexpected ID/seed mapping: {r}')
    return selected


def compare(actual, expected, tol=1e-5):
    a = {r['base_run']: r for r in actual}
    b = {r['base_run']: r for r in expected}
    require(len(a) == len(actual) and len(b) == len(expected), 'Duplicate base rows')
    require(a.keys() == b.keys(), 'Base ID sets differ')
    for k, r in b.items():
        require(int(a[k]['seed']) == int(r['seed']) and int(a[k]['iteration']) == int(r['iteration']),
                f'Seed/iteration mismatch: {k}')
        for m in METRICS:
            require(math.isclose(float(a[k][m]), float(r[m]), rel_tol=0, abs_tol=tol),
                    f'{k}: {m} differs: {a[k][m]} vs {r[m]}')


def verify(selected):
    ids = {r['base_run'] for r in selected}
    manifest = json.loads((RESULTS/'Stage2/stage2_manifest.json').read_text(encoding='utf-8-sig'))
    require(ids.issubset(set(manifest['base_runs'])), 'Selected IDs missing from Stage2 manifest')
    for regime in ('reference', 'stress'):
        expected = rows(RESULTS / f'May_2026_{regime}_selected_base_data.csv')
        require(len(expected) == 120 and {r['base_run'] for r in expected} == ids,
                f'{regime}: selected IDs differ')
        archived = rows(RESULTS / 'Stage2' / regime / 'summary_by_base_scenario.csv')
        compare([r for r in archived if r['base_run'] in ids], expected)
        means = {m: sum(float(r[m]) for r in expected)/120 for m in METRICS}
        print(f'{regime}: 120 matched base summaries; means: {means}')
    print('PASS: archived selected summaries match Stage2 rows. No simulation was run.')


def rerun(selected, workers):
    # The original runner's one_base() supports individual global IDs.
    # Its legacy main() imposes an unrelated 113-scenario limit.
    os.environ['BALTIC_RESULTS_SUBDIR'] = 'reproduced_may_selected'
    os.environ.update(BALTIC_BESS_P_MW='5', BALTIC_BESS_E_MWH='10',
                      BALTIC_CYCLE_COST_EUR_PER_MWH='25')
    import run_may_exact as original
    from fast_evaluator import evaluate_all
    require(not original.RESULT_DIR.exists(),
            'Output directory already exists. Rename it before a fresh reproduction.')
    require(not original.WORK_DIR.exists(), 'Worker directory exists. Inspect/rename it first.')
    original.check_sources()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(original.one_base, [int(r['iteration']) for r in selected]):
            print(result, flush=True)
    dirs = [original.scenario_path(int(r['iteration'])) for r in selected]
    require(all(original.is_complete(d) for d in dirs), 'Incomplete base runs')
    for regime in ('reference', 'stress'):
        out = original.RESULT_DIR / f'random_activation_{regime}'
        evaluate_all(dirs, regime, 1000, out)
        compare(rows(out/'summary_by_base_scenario.csv'),
                rows(RESULTS/f'May_2026_{regime}_selected_base_data.csv'))
    print('PASS: all selected base-level metrics match archived results within 1e-5.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true', help='Explicitly launch the costly simulations')
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if not 1 <= a.workers <= 16:
        p.error('workers must be 1..16')
    selected = selection()
    verify(selected)
    if a.run:
        rerun(selected, a.workers)


if __name__ == '__main__':
    main()
