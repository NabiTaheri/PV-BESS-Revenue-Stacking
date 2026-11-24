#!/usr/bin/env python3
"""Read-only cumulative-mean drift check for completed May-2026 exact bases.

Checks the FIRST N COMPLETED base scenarios for N=50,60,70,... rather than
mistaking missing ordinal base numbers for completed samples. No MILP is run.
Only existing results are read. Activation estimates use fast_evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run from the actual May project folder, even when this file lives elsewhere.
PROJECT = Path.cwd().resolve()
sys.path.insert(0, str(PROJECT))
import run_may_exact as core
from fast_evaluator import eval_one_run

METRICS = (
    ("S2 revenue", "s2_total_eur"),
    ("S3 net", "s3_net_mean"),
    ("S3-S2 gap", "paired_delta_mean_eur"),
)


def pct_delta(current, former):
    if former is None or not np.isfinite(former) or abs(former) < 1e-10:
        return float("nan")
    return 100.0 * (current - former) / abs(former)


def display_pct(value):
    return "    ---" if not np.isfinite(value) else f"{value:+7.2f}%"


def verify_project():
    if core.ROOT.resolve() != PROJECT:
        raise RuntimeError(f"Wrong project imported: {core.ROOT}; current folder: {PROJECT}")
    core.check_sources()  # Verifies 2976 aligned 15-min timestamps, exactly May 2026.
    manifest_file = core.RESULT_DIR / "fast_run_manifest.json"
    if not manifest_file.is_file():
        raise RuntimeError(f"Manifest not found: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("engine") != "original_S2_MILP_reusable_sparse":
        raise RuntimeError("Manifest engine mismatch")
    if manifest.get("time_period") != "2026-05-01 to 2026-05-31":
        raise RuntimeError("Manifest month mismatch; refusing to analyze mixed months")
    for rel, sha in manifest.get("input_sha256", {}).items():
        file = core.ROOT / rel
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"Model/input hash mismatch: {rel}")
    print(f"[PROJECT] {PROJECT}")
    print(f"[RESULTS] {core.RESULT_DIR}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200,
                        help="New read-only activation samples per completed base/regime (default: 200; use 1000 for original interim precision)")
    parser.add_argument("--start", type=int, default=50)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--max-base", type=int, default=113,
                        help="Only inspect original 1..113; new extension 114..200 is excluded")
    args = parser.parse_args()
    if args.samples < 20 or args.start < 2 or args.step < 1 or not (1 <= args.max_base <= 113):
        parser.error("Invalid args: samples>=20, start>=2, step>=1, max-base=1..113")

    verify_project()
    complete = [i for i in range(1, args.max_base + 1)
                if core.is_complete(core.scenario_path(i))]
    missing = [i for i in range(1, args.max_base + 1) if i not in set(complete)]
    print(f"[STATUS] Complete: {len(complete)} / {args.max_base}; missing: {missing}")
    checkpoints = list(range(args.start, len(complete) + 1, args.step))
    if not checkpoints:
        print(f"[INFO] Fewer than {args.start} completed bases; no requested checkpoint yet.")
        return

    # No dispatch / MILP or base simulations. Evaluate only bases through the
    # last checkpoint, so extra available bases do not waste processing time.
    selected = complete[:checkpoints[-1]]
    data = {regime: [] for regime in ("reference", "stress")}
    for count, base in enumerate(selected, start=1):
        for regime in data:
            _, item = eval_one_run(core.scenario_path(base), regime, args.samples)
            for _, col in METRICS:
                if col not in item:
                    raise RuntimeError(f"Missing metric {col} in fast_evaluator output")
            data[regime].append(item)
        if count % 10 == 0 or count == len(selected):
            print(f"[READ] {count}/{len(selected)} complete bases evaluated (last base: {base})", flush=True)

    all_rows = []
    for regime, records in data.items():
        print(f"\n=== {regime.upper()} | cumulative averages (EUR), {args.samples} activations/base ===")
        print(f"{'Metric':<17} {'N':>3} {'Last base':>9} {'Mean EUR':>14} {'vs prior':>10} {'vs N=50':>10} {'CV%':>9}")
        for label, key in METRICS:
            previous_mean = None
            first_mean = None
            for n in checkpoints:
                values = np.asarray([r[key] for r in records[:n]], dtype=float)
                mean = float(np.mean(values))
                sd = float(np.std(values, ddof=1))
                cv = 100 * sd / abs(mean) if abs(mean) > 1e-10 else float("nan")
                prior_pct = pct_delta(mean, previous_mean)
                first_pct = pct_delta(mean, first_mean)
                print(f"{label:<17} {n:>3} {selected[n - 1]:>9} {mean:>14,.2f} {display_pct(prior_pct):>10} {display_pct(first_pct):>10} {cv:>8.2f}%")
                all_rows.append({"regime": regime, "metric": label,
                                 "n_completed": n, "last_base_number": selected[n - 1],
                                 "mean_eur": mean, "sd_eur": sd, "cv_percent": cv,
                                 "drift_vs_previous_percent": prior_pct,
                                 "drift_vs_first_percent": first_pct,
                                 "activation_samples_per_base": args.samples})
                if first_mean is None:
                    first_mean = mean
                previous_mean = mean

    out = PROJECT / "mean_drift_check"
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"may_cumulative_drift_{args.samples}_samples.csv"
    pd.DataFrame(all_rows).to_csv(outfile, index=False, encoding="utf-8-sig")
    print(f"\n[SAVED] {outfile}")
    print("[NOTE] 'vs prior' is the %% change of the cumulative mean since the previous 10-completed-base checkpoint.")
    print("[NOTE] 'CV%%' is scenario-to-scenario variation, not change in the cumulative mean.")
    print("[SAFE] No MILP, no new base simulations, and no changes to final_results.")
    print("[CAUTION] This is a descriptive stability check, not a formal convergence/stopping test.")


if __name__ == "__main__":
    main()
