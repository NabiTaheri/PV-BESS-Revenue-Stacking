#!/usr/bin/env python3
"""Retrospective Reference convergence: start at 50 base scenarios, add 10.

Only the Reference activation regime is used to identify convergence.
At the verified Reference count, Stress is summarized on exactly the same
matched base scenarios; no separate Stress convergence test is carried out.

The look-by-look thresholds are provisional research choices; review and freeze
before describing results as definitive in an academic manuscript. Nominal
95% CIs at repeated looks do not constitute a sequential 95% stopping guarantee.

Windows example:
  python reference_convergence_step10.py --input May_2026=D:\\BalticMay2026Final\\stage2_full_20261121_223238 --input November_2025=D:\\BalticEstoniaExact_2025_11_27days\\stage2_full_20261121_223238 --output D:\\Baltic_Convergence_50_step10
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

# PRELIMINARY, UNCHANGED economic thresholds from the previous audit.
REVENUE_CI_REL = 0.05     # CI half-width <= 5% of abs(mean) for S2 and S3-net
GAP_CI_REL = 0.10         # paired-gap CI half-width <= 10% of abs(mean gap)
GAP_FLOOR_S2_REL = 0.02   # allow 2% abs(S2) scale for near-break-even gaps
MEAN_DRIFT_REL = 0.02    # adjacent-checkpoint drift <= 2%
PROB_LOW, PROB_HIGH = 0.20, 0.80
FIRST_N, STEP = 50, 10
STABILITY_WINDOW = 3      # three recent looks for mean/decision stability
CONFIRM_PASSES = 3        # three consecutive *full-criterion* passes
COLS = (
    'base_run', 'iteration', 'seed', 's2_total_eur', 's3_net_mean',
    'paired_delta_mean_eur', 'prob_s3_net_ge_s2'
)


def ci95(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    if n < 2:
        return float(np.mean(a)), float('nan'), float('nan'), float('nan')
    mean = float(np.mean(a))
    hw = float(t.ppf(.975, n - 1) * np.std(a, ddof=1) / math.sqrt(n))
    return mean, hw, mean - hw, mean + hw


def gap_decision(low, high, materiality):
    if low > 0:
        return 'S3_favoured'
    if high < 0:
        return 'S2_favoured'
    if low >= -materiality and high <= materiality:
        return 'near_break_even'
    return 'uncertain'


def probability_class(low, high):
    low, high = max(0., low), min(1., high)
    if high < PROB_LOW:
        return 'low'
    if low > PROB_HIGH:
        return 'high'
    if low >= PROB_LOW and high <= PROB_HIGH:
        return 'intermediate'
    return 'uncertain'


def read_base_summary(folder, regime):
    path = folder / regime / 'summary_by_base_scenario.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}')
    d = pd.read_csv(path)
    missing = set(COLS) - set(d.columns)
    if missing:
        raise ValueError(f'{path}: missing {sorted(missing)}')
    if len(d) < FIRST_N:
        raise ValueError(f'{path}: need at least {FIRST_N} completed base scenarios')
    for c in ('iteration', 'seed', 's2_total_eur', 's3_net_mean',
              'paired_delta_mean_eur', 'prob_s3_net_ge_s2'):
        d[c] = pd.to_numeric(d[c], errors='raise')
    if not np.isfinite(d[['iteration', 'seed', 's2_total_eur', 's3_net_mean',
                          'paired_delta_mean_eur', 'prob_s3_net_ge_s2']].to_numpy()).all():
        raise ValueError(f'{path}: nonfinite entries')
    if not (d['iteration'] == d['iteration'].astype(int)).all() or not (d['seed'] == d['seed'].astype(int)).all():
        raise ValueError(f'{path}: noninteger run identifiers')
    if d.duplicated('iteration').any() or d.duplicated('seed').any():
        raise ValueError(f'{path}: repeated base scenario identifier')
    if not d.apply(lambda r: r['base_run'] == f"run_{int(r['iteration']):04d}_seed_{int(r['seed'])}", axis=1).all():
        raise ValueError(f'{path}: run identifier disagrees with iteration/seed')
    if not d['prob_s3_net_ge_s2'].between(0., 1.).all():
        raise ValueError(f'{path}: probabilities outside [0,1]')
    discrepancy = abs(d['paired_delta_mean_eur'] - (d['s3_net_mean'] - d['s2_total_eur']))
    if float(discrepancy.max()) > 0.05:
        raise ValueError(f'{path}: S3-S2 paired mean inconsistency ({discrepancy.max():.2f} EUR)')
    return d.sort_values(['iteration', 'seed']).reset_index(drop=True)


def evaluate_at_n(d, n, month, regime, previous_looks=None, regular=True):
    chosen = d.iloc[:n]
    s2, h2, l2, u2 = ci95(chosen.s2_total_eur)
    s3, h3, l3, u3 = ci95(chosen.s3_net_mean)
    gap, hg, lg, ug = ci95(chosen.paired_delta_mean_eur)
    prob, hp, lp, up = ci95(chosen.prob_s3_net_ge_s2)
    lp, up = max(0., lp), min(1., up)
    materiality = GAP_FLOOR_S2_REL * abs(s2)
    rec = {
        'month': month, 'regime': regime, 'n_base': n,
        'regular_checkpoint': bool(regular),
        's2_mean_eur': s2, 's2_ci95_half_eur': h2,
        's2_ci95_lower_eur': l2, 's2_ci95_upper_eur': u2,
        's3_net_mean_eur': s3, 's3_ci95_half_eur': h3,
        's3_ci95_lower_eur': l3, 's3_ci95_upper_eur': u3,
        'paired_gap_mean_eur': gap, 'paired_gap_ci95_half_eur': hg,
        'paired_gap_ci95_lower_eur': lg, 'paired_gap_ci95_upper_eur': ug,
        'prob_s3_ge_s2': prob, 'prob_ci95_half': hp,
        'prob_ci95_lower': lp, 'prob_ci95_upper': up,
        'gap_decision': gap_decision(lg, ug, materiality),
        'probability_class': probability_class(lp, up),
        'materiality_eur': materiality,
        'revenue_precision_pass': bool(h2 <= REVENUE_CI_REL * abs(s2) and
                                       h3 <= REVENUE_CI_REL * abs(s3)),
        'gap_precision_pass': bool(hg <= max(GAP_CI_REL * abs(gap), materiality)),
        'drift_s2_rel': float('nan'), 'drift_s3_rel': float('nan'),
        'drift_gap_rel': float('nan'),
        'cumulative_stability_pass': False,
        'economic_decision_stability_pass': False,
        'all_criteria_pass': False,
    }
    if previous_looks:
        prior = previous_looks[-1]
        rec['drift_s2_rel'] = abs(s2 - prior['s2_mean_eur']) / max(abs(s2), 1.)
        rec['drift_s3_rel'] = abs(s3 - prior['s3_net_mean_eur']) / max(abs(s3), 1.)
        scale = max(abs(gap), materiality, 1.)
        rec['drift_gap_rel'] = abs(gap - prior['paired_gap_mean_eur']) / scale
    if previous_looks is not None and len(previous_looks) >= STABILITY_WINDOW - 1:
        last = [*previous_looks[-(STABILITY_WINDOW-1):], rec]
        rec['economic_decision_stability_pass'] = bool(
            rec['gap_decision'] != 'uncertain'
            and rec['probability_class'] != 'uncertain'
            and len({row['gap_decision'] for row in last}) == 1
            and len({row['probability_class'] for row in last}) == 1
        )
        rec['cumulative_stability_pass'] = bool(all(
            row['drift_s2_rel'] <= MEAN_DRIFT_REL
            and row['drift_s3_rel'] <= MEAN_DRIFT_REL
            and row['drift_gap_rel'] <= MEAN_DRIFT_REL
            for row in last[1:]
        ))
        rec['all_criteria_pass'] = bool(
            rec['revenue_precision_pass']
            and rec['gap_precision_pass']
            and rec['economic_decision_stability_pass']
            and rec['cumulative_stability_pass']
        )
    return rec


def reference_path(month, d):
    n_total = len(d)
    regular_ns = list(range(FIRST_N, n_total + 1, STEP))
    rows = []
    for n in regular_ns:
        rows.append(evaluate_at_n(d, n, month, 'reference', rows, True))
    # The incomplete last block is inspected as an additional robustness
    # diagnostic, not treated as a new 10-scenario checkpoint.
    if regular_ns[-1] != n_total:
        rows.append(evaluate_at_n(d, n_total, month, 'reference', rows, False))
    regular = [r for r in rows if r['regular_checkpoint']]
    confirmed = None
    onset = None
    # Certify at the END of the three successive full-criterion passes.
    # Subsequent regular looks and any final partial-block diagnostic must
    # keep passing, so a temporary 'lucky streak' is not reported.
    for end in range(CONFIRM_PASSES - 1, len(regular)):
        streak = regular[end - CONFIRM_PASSES + 1:end + 1]
        if all(r['all_criteria_pass'] for r in streak) and all(
                r['all_criteria_pass'] for r in regular[end + 1:]) and (
                rows[-1]['regular_checkpoint'] or rows[-1]['all_criteria_pass']):
            confirmed = regular[end]['n_base']
            onset = streak[0]['n_base']
            break
    end_regular = regular[-1]
    last = rows[-1]
    summary = {
        'month': month,
        'total_valid_bases': n_total,
        'first_checkpoint': FIRST_N,
        'checkpoint_step': STEP,
        'last_regular_checkpoint': end_regular['n_base'],
        'last_regular_pass': end_regular['all_criteria_pass'],
        'final_available_n': n_total,
        'final_available_pass': last['all_criteria_pass'],
        'first_three_pass_streak_n': onset,
        'reference_confirmed_n': confirmed,
        'reference_convergence_status': 'confirmed_retrospectively' if confirmed else 'not_confirmed',
        'reference_final_gap_decision': last['gap_decision'],
        'reference_final_probability_class': last['probability_class'],
    }
    return pd.DataFrame(rows), summary


def make_reference_plot(month, checkpoints, path, convergence_n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    shown = [
        ('s2_mean_eur', 's2_ci95_half_eur', 'S2 revenue (EUR)', axes[0, 0]),
        ('s3_net_mean_eur', 's3_ci95_half_eur', 'Reference S3-net (EUR)', axes[0, 1]),
        ('paired_gap_mean_eur', 'paired_gap_ci95_half_eur', 'Reference paired S3-net - S2 (EUR)', axes[1, 0]),
        ('prob_s3_ge_s2', 'prob_ci95_half', 'Reference P(S3-net >= S2)', axes[1, 1]),
    ]
    data = checkpoints.loc[checkpoints.regular_checkpoint].copy()
    x = data.n_base.to_numpy(dtype=float)
    for metric, half, title, ax in shown:
        y = data[metric].to_numpy(dtype=float)
        h = data[half].to_numpy(dtype=float)
        ax.plot(x, y, marker='.', label='estimate')
        ax.fill_between(x, y - h, y + h, alpha=.15, label='nominal 95% CI')
        if convergence_n is not None:
            ax.axvline(convergence_n, linestyle='--', label=f'confirmed N={convergence_n}')
        ax.set(title=title, xlabel='Independent completed base scenarios')
        ax.grid(alpha=.2)
    axes[1, 1].set_ylim(-.05, 1.05)
    axes[0, 0].legend(loc='best', fontsize='small')
    fig.suptitle(f'{month}: 50 + 10 Reference retrospective convergence checks')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, metavar='MONTH=STAGE2_DIR')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output is not empty; nothing overwritten: {out}')

    all_looks = []
    summary_rows = []
    stress_rows = []
    chosen_run_rows = []
    names = set()
    plots = []
    for spec in args.input:
        if '=' not in spec:
            raise ValueError(f'Expected MONTH=STAGE2_DIR, got: {spec}')
        month, root = spec.split('=', 1)
        month, root = month.strip(), Path(root.strip())
        if not month or month in names or not root.is_dir():
            raise ValueError(f'Invalid or duplicated month / nonexistent directory: {spec}')
        names.add(month)
        ref = read_base_summary(root, 'reference')
        stress = read_base_summary(root, 'stress')
        keys = ['base_run', 'iteration', 'seed']
        if not ref[keys].equals(stress[keys]):
            raise ValueError(f'{month}: Reference and Stress base scenarios are not paired')
        if not np.allclose(ref.s2_total_eur, stress.s2_total_eur, rtol=0, atol=.05):
            raise ValueError(f'{month}: Reference and Stress S2 values differ')
        looks, summary = reference_path(month, ref)
        all_looks.append(looks)
        summary_rows.append(summary)
        chosen_n = summary['reference_confirmed_n']
        if chosen_n is not None:
            st = evaluate_at_n(stress, chosen_n, month, 'stress_at_reference_N')
            # Only descriptive uncertainty: no Stress convergence criteria.
            stress_rows.append({key: value for key, value in st.items()
                                if key not in ('revenue_precision_pass', 'gap_precision_pass',
                                               'drift_s2_rel', 'drift_s3_rel', 'drift_gap_rel',
                                               'cumulative_stability_pass',
                                               'economic_decision_stability_pass', 'all_criteria_pass')})
            for _, r in ref.iloc[:chosen_n][keys].iterrows():
                chosen_run_rows.append(dict(month=month, **r.to_dict()))
        plots.append((month, looks, chosen_n))

    out.mkdir(parents=True, exist_ok=False) if not out.exists() else None
    checkpoints = pd.concat(all_looks, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    checkpoints.to_csv(out / 'reference_checkpoints_50_plus_10.csv', index=False)
    summary_df.to_csv(out / 'reference_convergence_summary.csv', index=False)
    pd.DataFrame(stress_rows).to_csv(out / 'stress_at_reference_convergence.csv', index=False)
    pd.DataFrame(chosen_run_rows, columns=['month', 'base_run', 'iteration', 'seed']).to_csv(
        out / 'selected_base_scenarios.csv', index=False)
    for month, looks, n in plots:
        make_reference_plot(month, looks, out / f'{month}_reference_convergence.png', n)
    methodology = {
        'purpose': 'Retrospective Reference-only convergence; Stress is sensitivity evaluated at selected Reference N',
        'first_checkpoint': FIRST_N,
        'step': STEP,
        'reference_stability_window_checkpoints': STABILITY_WINDOW,
        'consecutive_full_passes_required': CONFIRM_PASSES,
        'robustness_check': 'All subsequent regular checkpoints and final available partial block must pass',
        'revenue_ci_halfwidth_ratio': REVENUE_CI_REL,
        'paired_gap_ci_halfwidth_ratio': GAP_CI_REL,
        'near_zero_gap_floor_ratio_to_abs_s2': GAP_FLOOR_S2_REL,
        'adjacent_cumulative_mean_drift_ratio': MEAN_DRIFT_REL,
        'probability_class_boundaries': [PROB_LOW, PROB_HIGH],
        'confidence_interval_method': 'Nominal 95% t intervals of independent base-level outcomes; activation trials nested within base',
        'threshold_status': 'PROVISIONAL until critically reviewed and fixed for manuscript',
        'sequential_inference_caveat': 'Repeated nominal t CIs do not themselves establish family-wise 95% sequential coverage',
        'stress_convergence_tested': False,
        'online_stopping_claim': False,
    }
    (out / 'methodology.json').write_text(json.dumps(methodology, indent=2), encoding='utf-8')
    print('\nREFERENCE CONVERGENCE: START 50, STEP 10, THREE CONFIRMING PASSES')
    print(summary_df.to_string(index=False))
    if stress_rows:
        keys = ['month', 'n_base', 's3_net_mean_eur', 'paired_gap_mean_eur', 'paired_gap_ci95_half_eur',
                'prob_s3_ge_s2', 'gap_decision', 'probability_class']
        print('\nSTRESS: DESCRIPTIVE RESULTS AT SAME REFERENCE BASE-SCENARIO COUNT')
        print(pd.DataFrame(stress_rows)[keys].to_string(index=False))
    else:
        print('\nNo Reference convergence confirmed; Stress matched-count summary not generated.')
    print('\nOutput:', out)
    print('Retrospective assessment; no claim of online stopping. Thresholds remain provisional.')


if __name__ == '__main__':
    main()
