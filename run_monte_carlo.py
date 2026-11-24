# run_monte_carlo.py
from __future__ import annotations

import csv
import json
import os
import re
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


# =============================================================================
# USER SETTINGS - فقط این بخش را تغییر بده
# =============================================================================

N_RUNS = int(os.environ.get("BALTIC_N_RUNS", "113"))
BASE_SEED = int(os.environ.get("BALTIC_BASE_SEED", "1010"))
CONTINUE_ON_ERROR = False

# اگر True باشد قبل از هر تکرار فولدرهای results قبلی پاک می‌شوند.
# این باعث می‌شود هر iteration خروجی تمیز خودش را داشته باشد.
CLEAN_RESULT_FOLDERS_BEFORE_EACH_RUN = True


# =============================================================================
# mFRR ENERGY ACTIVATION FRACTION SETTINGS
# =============================================================================
# This is inside the S2/S3 DA+mFRR-energy optimization, not a post-processing
# capacity-activation adjustment.
#
# For each Monte Carlo iteration, one random value is sampled and passed to both:
#   activation/DA/market_config.py
#   activation/MFRR&AFRR/market_config.py
# through the MFRR_ACTIVATION_FRACTION environment variable.
#
# Distribution options:
#   "uniform" -> uniform between MIN and MAX
#   "beta"    -> scaled beta distribution between MIN and MAX
MFRR_ACTIVATION_FRACTION_DISTRIBUTION = "beta"
MFRR_ACTIVATION_FRACTION_MIN = 0.0
MFRR_ACTIVATION_FRACTION_MAX = 1.00
MFRR_ACTIVATION_BETA_ALPHA = 2.0
MFRR_ACTIVATION_BETA_BETA = 6.0


def sample_mfrr_activation_fraction(seed: int) -> float:
    lo = float(MFRR_ACTIVATION_FRACTION_MIN)
    hi = float(MFRR_ACTIVATION_FRACTION_MAX)
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError("MFRR activation fraction bounds must satisfy 0 <= min <= max <= 1.")

    rng = random.Random(int(seed) + 7919)
    mode = str(MFRR_ACTIVATION_FRACTION_DISTRIBUTION).strip().lower()

    if mode == "uniform":
        value = rng.uniform(lo, hi)
    elif mode == "beta":
        raw = rng.betavariate(float(MFRR_ACTIVATION_BETA_ALPHA), float(MFRR_ACTIVATION_BETA_BETA))
        value = lo + (hi - lo) * raw
    else:
        raise ValueError(f"Unknown MFRR_ACTIVATION_FRACTION_DISTRIBUTION: {MFRR_ACTIVATION_FRACTION_DISTRIBUTION}")

    return max(0.0, min(1.0, float(value)))


def mfrr_activation_env(fraction: float) -> dict[str, str]:
    return {"MFRR_ACTIVATION_FRACTION": f"{float(fraction):.12f}"}



# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "Data"

MAKE_FORECAST_PY = DATA_DIR / "make_forecast.py"

SCENARIO_SCRIPTS = [
    ("S1_DA_only", ROOT_DIR / "run_S1_DA_only.py", ROOT_DIR / "results_S1_DA_only"),
    ("S2_DA_MFRR_energy", ROOT_DIR / "run_S2_DA_MFRR_energy.py", ROOT_DIR / "results_S2_DA_MFRR_energy"),
    ("S3_reserve_competition_full_stack", ROOT_DIR / "run_S3_reserve_competition_full_stack.py", ROOT_DIR / "results_S3_reserve_competition_full_stack"),
]

ACTUALS_CSV = DATA_DIR / "actuals.csv"
FORECAST_CSV = DATA_DIR / "forecasts_rolling.csv"

OUTPUT_ROOT = Path(os.environ.get("BALTIC_MC_ROOT", str(ROOT_DIR / "monte_carlo_results2"))).resolve()


# =============================================================================
# BASIC UTILS
# =============================================================================

def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    log_file: Path,
    env_overrides: dict[str, str] | None = None,
    stream_to_console: bool = False,
    console_prefixes: tuple[str, ...] | None = None,
) -> None:
    """Run a child command, write a full log, and optionally stream selected lines to terminal."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()

    child_env = os.environ.copy()
    if env_overrides:
        child_env.update({str(k): str(v) for k, v in env_overrides.items()})

    with log_file.open("w", encoding="utf-8", errors="replace") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n")
        f.write(f"CWD: {cwd}\n")
        f.write("=" * 100 + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()

            if stream_to_console:
                out = line.rstrip("\n")
                if console_prefixes is None or any(out.startswith(prefix) for prefix in console_prefixes):
                    print(out, flush=True)

        return_code = proc.wait()
        elapsed = time.time() - start

        f.write("\n" + "=" * 100 + "\n")
        f.write(f"RETURN_CODE: {return_code}\n")
        f.write(f"ELAPSED_SECONDS: {elapsed:.1f}\n")

    if return_code != 0:
        raise RuntimeError(f"Command failed with code {return_code}. See log: {log_file}")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing directory: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def remove_dir_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def norm(x: object) -> str:
    s = str(x).strip().lower()
    s = s.replace("€", "eur")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def to_float(x: object, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, str):
        x = x.replace(",", "").replace("EUR", "").replace("eur", "").strip()
    val = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(val):
        return float(default)
    return float(val)


# =============================================================================
# SUMMARY PARSING
# =============================================================================

SUMMARY_ALIASES = {
    "fcr_revenue_eur": [
        "fcr_revenue_eur",
        "fcr_revenue",
        "fcr revenue",
        "total_fcr_revenue_eur",
    ],
    "afrr_mfrr_capacity_revenue_eur": [
        "afrr_mfrr_capacity_revenue_eur",
        "afrr_mfrr_capacity_revenue",
        "afrr_mfrr capacity revenue",
        "afrr mfrr capacity revenue",
        "aFRR/mFRR capacity revenue",
        "capacity_revenue_eur",
        "reserve_capacity_revenue_eur",
        "total_afrr_mfrr_capacity_revenue_eur",
    ],
    "da_cashflow_actual_eur": [
        "da_cashflow_actual_eur",
        "da_cashflow_actual",
        "da cashflow actual",
        "da_cashflow_eur_actual",
    ],
    "mfrr_energy_cashflow_actual_eur": [
        "mfrr_energy_cashflow_actual_eur",
        "mfrr_energy_cashflow_actual",
        "mfrr energy cashflow actual",
        "mfrr_energy_cashflow_eur_actual",
    ],
    "imbalance_penalty_actual_eur": [
        "imbalance_penalty_actual_eur",
        "imbalance_penalty_actual",
        "imbalance penalty actual",
        "imbalance_penalty_eur_actual",
    ],
    "battery_degradation_eur": [
        "battery_degradation_eur",
        "battery_degradation",
        "battery degradation",
    ],
    "da_market_net_actual_eur": [
        "da_market_net_actual_eur",
        "da_market_net_actual",
        "da market net actual",
        "market_net_actual",
        "market_net_actual_eur",
        "market_net_cashflow_eur_actual",
        "net_cashflow_eur_actual",
        "da_net_cashflow_eur_actual",
        "da_mfrr_energy_net_eur",
        "da_net_eur",
    ],
    "total_market_actual_eur": [
        "total_market_actual_eur",
        "total_market_actual",
        "total market actual",
        "monthly_total_market_actual_eur",
        "total_revenue_market_actual_eur",
        "total_day_market_eur",
        "total_eur",
    ],
    "reserve_net_revenue_low_eur": [
        "reserve_net_revenue_low_eur",
        "total_reserve_net_revenue_eur_low",
    ],
    "reserve_net_revenue_medium_eur": [
        "reserve_net_revenue_medium_eur",
        "total_reserve_net_revenue_eur_medium",
    ],
    "reserve_net_revenue_high_eur": [
        "reserve_net_revenue_high_eur",
        "total_reserve_net_revenue_eur_high",
    ],
    "total_market_net_low_activation_eur": [
        "total_market_net_low_activation_eur",
        "total_market_actual_low_activation_eur",
        "s3_net_low_activation_eur",
    ],
    "total_market_net_medium_activation_eur": [
        "total_market_net_medium_activation_eur",
        "total_market_actual_medium_activation_eur",
        "s3_net_medium_activation_eur",
    ],
    "total_market_net_high_activation_eur": [
        "total_market_net_high_activation_eur",
        "total_market_actual_high_activation_eur",
        "s3_net_high_activation_eur",
    ],
}


def empty_summary() -> dict[str, float]:
    return {k: 0.0 for k in SUMMARY_ALIASES}


def _matched_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    norm_cols = {norm(c): c for c in df.columns}
    for alias in aliases:
        col = norm_cols.get(norm(alias))
        if col is not None:
            return col
    return None


def _is_daily_summary(df: pd.DataFrame) -> bool:
    """True when monthly_chain_summary.csv contains one row per day."""
    date_col = _matched_column(df, ["date", "day", "timestamp"])
    if date_col is None:
        return False

    dates = pd.to_datetime(df[date_col], errors="coerce")
    return int(dates.notna().sum()) > 1


def _sum_numeric_column(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())


def _last_numeric_value(df: pd.DataFrame, col: str) -> float:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.iloc[-1])


def read_summary_csv(summary_file: Path) -> dict[str, float]:
    """Read monthly_chain_summary.csv correctly.

    Important:
      - Your scenario summaries are daily rows, not a single monthly-total row.
      - Therefore the monthly result must be the SUM of daily revenue columns.
      - The old version read only the last row, so S3 looked like one day only.
      - It also missed some S1/S2 column names, so S1/S2 became zero.
    """
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_file}")

    df = pd.read_csv(summary_file)
    if df.empty:
        raise ValueError(f"Empty summary file: {summary_file}")

    values = empty_summary()
    daily_mode = _is_daily_summary(df)

    # ----- wide daily/monthly format -----
    wide_found = False

    for out_key, aliases in SUMMARY_ALIASES.items():
        col = _matched_column(df, aliases)
        if col is None:
            continue

        if daily_mode:
            values[out_key] = _sum_numeric_column(df, col)
        else:
            values[out_key] = _last_numeric_value(df, col)

        wide_found = True

    if wide_found:
        # If no direct total column exists, build total from reserve + DA/mFRR-energy net.
        if abs(values["total_market_actual_eur"]) < 1e-12:
            values["total_market_actual_eur"] = (
                values["fcr_revenue_eur"]
                + values["afrr_mfrr_capacity_revenue_eur"]
                + values["da_market_net_actual_eur"]
            )

        # If S3 net activation totals are missing, fall back to gross. This keeps
        # S1/S2 compatible and lets old result folders still parse.
        for scen in ["low", "medium", "high"]:
            key = f"total_market_net_{scen}_activation_eur"
            reserve_key = f"reserve_net_revenue_{scen}_eur"
            if abs(values.get(key, 0.0)) < 1e-12:
                if abs(values.get(reserve_key, 0.0)) > 1e-12:
                    values[key] = values[reserve_key] + values["da_market_net_actual_eur"]
                else:
                    values[key] = values["total_market_actual_eur"]

        # If DA net was not found but total exists and reserve is known, infer DA net.
        if abs(values["da_market_net_actual_eur"]) < 1e-12 and abs(values["total_market_actual_eur"]) > 1e-12:
            values["da_market_net_actual_eur"] = (
                values["total_market_actual_eur"]
                - values["fcr_revenue_eur"]
                - values["afrr_mfrr_capacity_revenue_eur"]
            )

        return values

    # ----- key-value format -----
    # Used only if a file is saved as Metric, Value.
    for metric_col in df.columns:
        for value_col in df.columns:
            if metric_col == value_col:
                continue

            trial = empty_summary()
            found = False

            for _, row in df.iterrows():
                metric_name = norm(row.get(metric_col))
                for out_key, aliases in SUMMARY_ALIASES.items():
                    if metric_name in {norm(a) for a in aliases}:
                        trial[out_key] = to_float(row.get(value_col))
                        found = True
                        break

            if found:
                if abs(trial["total_market_actual_eur"]) < 1e-12:
                    trial["total_market_actual_eur"] = (
                        trial["fcr_revenue_eur"]
                        + trial["afrr_mfrr_capacity_revenue_eur"]
                        + trial["da_market_net_actual_eur"]
                    )
                return trial

    raise ValueError(
        f"Could not parse summary file: {summary_file}\n"
        f"Columns: {list(df.columns)}\n"
        f"Head:\n{df.head(5)}"
    )


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# ONE ITERATION
# =============================================================================

def run_one_iteration(iteration: int, seed: int, output_root: Path) -> None:
    iter_dir = output_root / f"run_{iteration:04d}_seed_{seed}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 100, flush=True)
    print(f"ITERATION {iteration}/{N_RUNS} | seed={seed}", flush=True)
    print("#" * 100, flush=True)

    mfrr_activation_fraction = sample_mfrr_activation_fraction(seed)
    print(f"  mFRR energy activation fraction = {mfrr_activation_fraction:.4f}", flush=True)

    # 0) clean old result folders
    if CLEAN_RESULT_FOLDERS_BEFORE_EACH_RUN:
        for _scenario_name, _script, result_dir in SCENARIO_SCRIPTS:
            remove_dir_if_exists(result_dir)

    # 1) run Data/make_forecast.py and create a new Data/forecasts_rolling.csv
    forecast_log = iter_dir / "logs" / "01_make_forecast.log"
    run_command(
        [
            sys.executable,
            str(MAKE_FORECAST_PY),
            "--input-file",
            str(ACTUALS_CSV),
            "--output-file",
            str(FORECAST_CSV),
            "--seed",
            str(seed),
        ],
        cwd=DATA_DIR,
        log_file=forecast_log,
    )

    copy_file(FORECAST_CSV, iter_dir / "data" / "forecasts_rolling.csv")
    copy_file(ACTUALS_CSV, iter_dir / "data" / "actuals.csv")

    # 2) run S1, S2, S3 sequentially
    scenario_rows: list[dict[str, object]] = []

    for scenario_name, script_path, result_dir in SCENARIO_SCRIPTS:
        if not script_path.exists():
            raise FileNotFoundError(f"Missing scenario script: {script_path}")

        log_file = iter_dir / "logs" / f"02_{scenario_name}.log"

        print(f"  Running {scenario_name} ...", flush=True)

        is_s3 = scenario_name == "S3_reserve_competition_full_stack"
        scenario_env = mfrr_activation_env(mfrr_activation_fraction) if scenario_name in {
            "S2_DA_MFRR_energy",
            "S3_reserve_competition_full_stack",
        } else None

        run_command(
            [sys.executable, str(script_path)],
            cwd=ROOT_DIR,
            log_file=log_file,
            env_overrides=scenario_env,
            stream_to_console=is_s3,
            console_prefixes=(
                "DAY ",
                "   S3 gross revenue breakdown:",
                "ERROR on ",
            ),
        )

        # 3) copy each scenario result folder into the iteration folder
        result_copy_dir = iter_dir / "results" / result_dir.name
        copy_dir(result_dir, result_copy_dir)

        summary_file = result_dir / "monthly_chain_summary.csv"
        summary_copy = result_copy_dir / "monthly_chain_summary.csv"

        values = read_summary_csv(summary_file)

        row = {
            "iteration": iteration,
            "seed": seed,
            "mfrr_energy_activation_fraction": round(mfrr_activation_fraction, 12),
            "scenario": scenario_name,
            "result_dir": str(result_copy_dir),
            "summary_file": str(summary_copy),
            **values,
        }
        scenario_rows.append(row)

    # 4) save per-iteration scenario totals
    write_csv(iter_dir / "scenario_totals.csv", scenario_rows)

    # 5) save comparison row
    by_scenario = {r["scenario"]: r for r in scenario_rows}

    s1 = float(by_scenario["S1_DA_only"]["total_market_actual_eur"])
    s2 = float(by_scenario["S2_DA_MFRR_energy"]["total_market_actual_eur"])
    s3 = float(by_scenario["S3_reserve_competition_full_stack"]["total_market_actual_eur"])

    comparison = {
        "iteration": iteration,
        "seed": seed,
        "mfrr_energy_activation_fraction": round(mfrr_activation_fraction, 12),
        "S1_total_eur": round(s1, 6),
        "S2_total_eur": round(s2, 6),
        "S3_total_eur": round(s3, 6),
        "S2_minus_S1_eur": round(s2 - s1, 6),
        "S3_minus_S2_eur": round(s3 - s2, 6),
        "S3_minus_S1_eur": round(s3 - s1, 6),
        "S2_ge_S1": bool(s2 >= s1),
        "S3_ge_S2": bool(s3 >= s2),
        "ranking_S3_ge_S2_ge_S1": bool(s3 >= s2 >= s1),
        "iteration_dir": str(iter_dir),
    }

    write_csv(iter_dir / "comparison.csv", [comparison])

    # 6) append to global summaries
    for row in scenario_rows:
        append_csv(output_root / "all_iterations_long.csv", row)

    append_csv(output_root / "all_iterations_comparison.csv", comparison)

    s3_row = by_scenario["S3_reserve_competition_full_stack"]
    s3_da_mfrr_energy = float(s3_row.get("da_market_net_actual_eur") or 0.0)
    s3_fcr = float(s3_row.get("fcr_revenue_eur") or 0.0)
    s3_afrr_mfrr_capacity = float(s3_row.get("afrr_mfrr_capacity_revenue_eur") or 0.0)
    s3_reserve_total = s3_fcr + s3_afrr_mfrr_capacity
    s3_component_total = s3_da_mfrr_energy + s3_reserve_total
    s3_unclassified = s3 - s3_component_total

    print(
        f"OK | mFRR activation fraction={mfrr_activation_fraction:.4f} | S1={s1:.2f} EUR | S2={s2:.2f} EUR | S3 gross={s3:.2f} EUR",
        flush=True,
    )
    print(
        "   S3 gross revenue breakdown: "
        f"DA+mFRR energy net={s3_da_mfrr_energy:.2f} EUR | "
        f"aFRR+mFRR capacity={s3_afrr_mfrr_capacity:.2f} EUR | "
        f"FCR capacity={s3_fcr:.2f} EUR | "
        f"Reserve capacity total={s3_reserve_total:.2f} EUR",
        flush=True,
    )
    if abs(s3_unclassified) > 0.01:
        print(
            f"   WARNING: {s3_unclassified:.2f} EUR of S3 gross revenue is not classified in the breakdown above.",
            flush=True,
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    if not MAKE_FORECAST_PY.exists():
        raise FileNotFoundError(f"Missing make_forecast.py: {MAKE_FORECAST_PY}")
    if not ACTUALS_CSV.exists():
        raise FileNotFoundError(f"Missing actuals.csv: {ACTUALS_CSV}")

    # start clean aggregate files
    for aggregate in [
        output_root / "all_iterations_long.csv",
        output_root / "all_iterations_comparison.csv",
        output_root / "failed_iterations.csv",
    ]:
        if aggregate.exists():
            aggregate.unlink()

    print("=" * 100, flush=True)
    print("MONTE CARLO SCENARIO LOOP", flush=True)
    print(f"Root      : {ROOT_DIR}", flush=True)
    print(f"Data      : {DATA_DIR}", flush=True)
    print(f"Runs      : {N_RUNS}", flush=True)
    print(f"Base seed : {BASE_SEED}", flush=True)
    print(f"Output    : {output_root}", flush=True)
    print("=" * 100, flush=True)

    for i in range(1, N_RUNS + 1):
        seed = BASE_SEED + i - 1

        try:
            run_one_iteration(i, seed, output_root)
        except Exception as exc:
            iter_dir = output_root / f"run_{i:04d}_seed_{seed}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "ERROR.txt").write_text(str(exc), encoding="utf-8")

            append_csv(
                output_root / "failed_iterations.csv",
                {
                    "iteration": i,
                    "seed": seed,
                    "error": str(exc),
                    "iteration_dir": str(iter_dir),
                },
            )

            print(f"FAILED iteration {i}: {exc}", flush=True)

            if not CONTINUE_ON_ERROR:
                raise

    print("\n" + "=" * 100, flush=True)
    print("FINISHED", flush=True)
    print(f"Comparison file: {output_root / 'all_iterations_comparison.csv'}", flush=True)
    print(f"Long file      : {output_root / 'all_iterations_long.csv'}", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
