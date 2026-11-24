#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Second-level reserve activation settlement sensitivity over forecast scenarios.

Two S3 metrics are output for the SAME activation trajectories:
    S3_cost_only = S3_gross - C_recovery - C_degradation
    S3_settlement = S3_cost_only + R_aFRR_activation + R_mFRR_capacity_activation

FCR has no separate energy settlement here. Direction-specific aFRR/mFRR
clearing prices are required from Data/settlement_prices.csv, obtained from
a published historical market source (no price imputation).

This is a stylized, price-conditioned scenario model, not a reconstruction of
actual BSP energy bid acceptance. See README_FA.md for interpretation.
"""

from __future__ import annotations

from pathlib import Path
import os
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent
MC_ROOT = Path(os.environ.get("BALTIC_MC_ROOT", str(ROOT_DIR / "monte_carlo_results2"))).resolve()
OUTPUT_DIR = MC_ROOT / "random_activation_sensitivity"

# =============================================================================
# FAIR / NEUTRAL ACTIVATION SENSITIVITY SETTINGS
# =============================================================================

N_ACTIVATION_SAMPLES_PER_BASE_SCENARIO = int(os.environ.get("BALTIC_N_ACTIVATION", "1000"))
RANDOM_SEED = 20260622

# Activation-factor distribution:
# activation_fraction = activated reserve energy MWh / accepted reserve capacity-hours
# This is not an activation probability.
ACTIVATION_FRACTION_DISTRIBUTION = "beta"

FCR_ACTIVATION_FRACTION_RANGE = (0.0, 1.0)
AFRR_ACTIVATION_FRACTION_RANGE = (0.0, 1.0)
MFRR_ACTIVATION_FRACTION_RANGE = (0.0, 1.0)

# Fair/neutral activation intensity:
# Beta(10,10) has mean = 0.50 and keeps most samples near the middle.
FCR_ACTIVATION_BETA_ALPHA = 10.0
FCR_ACTIVATION_BETA_BETA = 10.0

AFRR_ACTIVATION_BETA_ALPHA = 10.0
AFRR_ACTIVATION_BETA_BETA = 10.0

MFRR_ACTIVATION_BETA_ALPHA = 10.0
MFRR_ACTIVATION_BETA_BETA = 10.0

# FCR direction bias:
# UP = battery discharges
# DOWN = battery charges
# This setting gives approximately 90% discharge and 10% charge.
FCR_UP_SHARE_RANGE = (0.85, 0.95)

# Keep aFRR/mFRR direction tied to the observed balancing direction.
USE_BALANCING_FLAG_FOR_AFRR_MFRR = True

# Recovery price mode:
#   "imbalance_single"   -> buy = sell = imbalance_price_actual
#   "conservative_vs_da" -> buy=max(imbalance, DA), sell=min(imbalance, DA)
RECOVERY_PRICE_MODE = "conservative_vs_da"

# Battery assumptions used only for ex-post activation/recovery sensitivity.
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95

# Degradation cost applied to activation + recovery grid throughput.
DEGRADATION_EUR_PER_MWH = float(os.environ.get("BALTIC_CYCLE_COST_EUR_PER_MWH", "25"))


# =============================================================================
# PATHS / COLUMN HELPERS
# =============================================================================

S3_DIR_NAME = "results_S3_reserve_competition_full_stack"
S2_DIR_NAME = "results_S2_DA_MFRR_energy"

CAPACITY_RESULTS_NAME = "monthly_afrr_mfrr_capacity_results.csv"
S3_DAILY_SUMMARY_NAME = "monthly_chain_summary.csv"
S2_DAILY_SUMMARY_NAME = "monthly_chain_summary.csv"

EPS = 1e-12


def list_base_run_dirs() -> list[Path]:
    if not MC_ROOT.exists():
        raise FileNotFoundError(f"Missing Monte Carlo root: {MC_ROOT}")

    runs = []
    for p in MC_ROOT.iterdir():
        if p.is_dir() and p.name.startswith("run_") and "_seed_" in p.name:
            s3_dir = p / "results" / S3_DIR_NAME
            if s3_dir.exists():
                runs.append(p)

    return sorted(runs)


def parse_iteration_seed(run_dir: Path) -> tuple[int | None, int | None]:
    m = re.search(r"run_(\d+)_seed_(\d+)", run_dir.name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def series_or_zero(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    c = first_existing_column(df, candidates)
    if c is None:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)


def sum_column_from_csv(path: Path, candidates: list[str]) -> float | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    c = first_existing_column(df, candidates)

    if c is None:
        return None

    return float(pd.to_numeric(df[c], errors="coerce").fillna(0.0).sum())


def infer_interval_hours(timestamps: pd.Series) -> float:
    ts = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()

    if len(ts) < 2:
        return 0.25

    diffs = ts.diff().dropna().dt.total_seconds() / 3600.0
    diffs = diffs[(diffs > 0) & (diffs < 6)]

    if len(diffs) == 0:
        return 0.25

    return float(diffs.median())


def load_capacity_results(s3_dir: Path) -> pd.DataFrame:
    path = s3_dir / CAPACITY_RESULTS_NAME

    if not path.exists():
        raise FileNotFoundError(f"Missing capacity results: {path}")

    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError(f"Capacity results need timestamp or date column: {path}")

    return df


def find_actuals_csv(run_dir: Path) -> Path:
    candidates = [
        run_dir / "data" / "actuals.csv",
        ROOT_DIR / "Data" / "actuals.csv",
        ROOT_DIR / "data" / "actuals.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find actuals.csv. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )


def attach_settlement_prices(df: pd.DataFrame, actuals_csv: Path) -> pd.DataFrame:
    out = df.copy()

    required = [
        "imbalance_buy_price",
        "imbalance_sell_price",
        "balancing_flag_for_activation",
    ]

    if all(c in out.columns for c in required):
        return attach_product_settlement_prices(out)

    ac = pd.read_csv(actuals_csv)

    if "timestamp" not in ac.columns:
        raise ValueError(f"actuals.csv has no timestamp column: {actuals_csv}")

    ac["timestamp"] = pd.to_datetime(ac["timestamp"], errors="coerce")

    keep = ["timestamp"]
    for c in [
        "imbalance_price_actual",
        "da_clearing_price",
        "balancing_flag",
    ]:
        if c in ac.columns:
            keep.append(c)

    out = out.merge(ac[keep], on="timestamp", how="left")

    imbalance = pd.to_numeric(
        out.get("imbalance_price_actual", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)

    da_price = pd.to_numeric(
        out.get("da_clearing_price", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)

    if RECOVERY_PRICE_MODE == "conservative_vs_da":
        out["imbalance_buy_price"] = np.maximum(imbalance, da_price)
        out["imbalance_sell_price"] = np.minimum(imbalance, da_price)
    elif RECOVERY_PRICE_MODE == "imbalance_single":
        out["imbalance_buy_price"] = imbalance
        out["imbalance_sell_price"] = imbalance
    else:
        raise ValueError(f"Unknown RECOVERY_PRICE_MODE: {RECOVERY_PRICE_MODE}")

    out["balancing_flag_for_activation"] = pd.to_numeric(
        out.get("balancing_flag", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)

    return attach_product_settlement_prices(out)


def attach_product_settlement_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Distinct aFRR/mFRR marginal clearing prices are required. Imbalance prices
    # above are ONLY for recovery, never used as balancing-energy remuneration.
    settlement_csv = ROOT_DIR / "Data" / "settlement_prices.csv"
    if not settlement_csv.exists():
        raise FileNotFoundError(
            f"Missing {settlement_csv}. Run download_settlement_prices.py first."
        )
    settle = pd.read_csv(settlement_csv)
    cols = ["afrr_up_price_eur_mwh", "afrr_down_price_eur_mwh",
            "mfrr_up_price_eur_mwh", "mfrr_down_price_eur_mwh"]
    if any(c not in settle.columns for c in ["timestamp", *cols]):
        raise ValueError("settlement_prices.csv must have timestamp and all four product/direction price columns")
    settle["timestamp"] = pd.to_datetime(settle["timestamp"], errors="raise")
    if settle["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamp in settlement_prices.csv")
    before = len(out)
    out = out.merge(settle[["timestamp", *cols]], on="timestamp", how="left", validate="many_to_one")
    if len(out) != before:
        raise ValueError("Settlement merge changed interval count")
    if out[cols].notna().sum().sum() == 0:
        raise ValueError("Settlement price data are completely empty; cannot infer balancing-energy cashflows")
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =============================================================================
# RANDOM SAMPLING / EVALUATION
# =============================================================================

@dataclass(frozen=True)
class ActivationSample:
    sample_id: int
    fcr_activation_fraction: float
    afrr_activation_fraction: float
    mfrr_activation_fraction: float
    fcr_up_share: float


def validate_range(name: str, bounds: tuple[float, float]) -> None:
    lo, hi = bounds
    if lo < 0.0 or hi > 1.0 or lo > hi:
        raise ValueError(f"{name} must satisfy 0 <= lo <= hi <= 1. Got {bounds}")


def make_random_samples(rng: np.random.Generator, n: int) -> list[ActivationSample]:
    validate_range("FCR_ACTIVATION_FRACTION_RANGE", FCR_ACTIVATION_FRACTION_RANGE)
    validate_range("AFRR_ACTIVATION_FRACTION_RANGE", AFRR_ACTIVATION_FRACTION_RANGE)
    validate_range("MFRR_ACTIVATION_FRACTION_RANGE", MFRR_ACTIVATION_FRACTION_RANGE)
    validate_range("FCR_UP_SHARE_RANGE", FCR_UP_SHARE_RANGE)

    def uniform_scaled(bounds: tuple[float, float]) -> np.ndarray:
        return rng.uniform(bounds[0], bounds[1], size=n)

    def beta_scaled(bounds: tuple[float, float], alpha: float, beta: float) -> np.ndarray:
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("Beta parameters must be positive.")

        lo, hi = bounds
        raw = rng.beta(float(alpha), float(beta), size=n)
        return lo + (hi - lo) * raw

    mode = str(ACTIVATION_FRACTION_DISTRIBUTION).strip().lower()

    if mode == "uniform":
        fcr = uniform_scaled(FCR_ACTIVATION_FRACTION_RANGE)
        afrr = uniform_scaled(AFRR_ACTIVATION_FRACTION_RANGE)
        mfrr = uniform_scaled(MFRR_ACTIVATION_FRACTION_RANGE)

    elif mode == "beta":
        fcr = beta_scaled(
            FCR_ACTIVATION_FRACTION_RANGE,
            FCR_ACTIVATION_BETA_ALPHA,
            FCR_ACTIVATION_BETA_BETA,
        )
        afrr = beta_scaled(
            AFRR_ACTIVATION_FRACTION_RANGE,
            AFRR_ACTIVATION_BETA_ALPHA,
            AFRR_ACTIVATION_BETA_BETA,
        )
        mfrr = beta_scaled(
            MFRR_ACTIVATION_FRACTION_RANGE,
            MFRR_ACTIVATION_BETA_ALPHA,
            MFRR_ACTIVATION_BETA_BETA,
        )

    else:
        raise ValueError(
            f"Unknown ACTIVATION_FRACTION_DISTRIBUTION: "
            f"{ACTIVATION_FRACTION_DISTRIBUTION}"
        )

    fcr_up = uniform_scaled(FCR_UP_SHARE_RANGE)

    return [
        ActivationSample(
            sample_id=i + 1,
            fcr_activation_fraction=float(fcr[i]),
            afrr_activation_fraction=float(afrr[i]),
            mfrr_activation_fraction=float(mfrr[i]),
            fcr_up_share=float(fcr_up[i]),
        )
        for i in range(n)
    ]


def recovery_cost_and_degradation(
    up_mwh: pd.Series,
    down_mwh: pd.Series,
    buy_price: pd.Series,
    sell_price: pd.Series,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Calculates recovery cost and degradation after reserve activation.

    up_mwh:
        Activation that discharges the battery.
    down_mwh:
        Activation that charges the battery.

    Battery energy drift:
        positive drift -> battery has surplus stored energy after activation
        negative drift -> battery has energy deficit after activation

    Recovery:
        negative drift -> buy energy to recover
        positive drift -> sell energy to recover

    Returns
    -------
    net_recovery_cost_eur:
        recovery_buy_cost - recovery_sell_revenue
    degradation_eur:
        throughput_mwh * degradation_cost
    recovery_buy_mwh
    recovery_sell_mwh
    recovery_buy_cost_eur
    recovery_sell_revenue_eur
    throughput_mwh
    """

    eta_c = max(float(CHARGE_EFFICIENCY), EPS)
    eta_d = max(float(DISCHARGE_EFFICIENCY), EPS)

    # Battery energy drift after activation.
    drift_mwh = eta_c * down_mwh - up_mwh / eta_d

    # Energy bought/sold at grid side to restore battery state.
    recovery_buy_mwh = (-drift_mwh.clip(upper=0.0)) / eta_c
    recovery_sell_mwh = drift_mwh.clip(lower=0.0) * eta_d

    recovery_buy_cost = recovery_buy_mwh * buy_price
    recovery_sell_revenue = recovery_sell_mwh * sell_price

    net_recovery_cost_eur = float(
        recovery_buy_cost.sum() - recovery_sell_revenue.sum()
    )

    throughput_mwh = (
        up_mwh.abs()
        + down_mwh.abs()
        + recovery_buy_mwh.abs()
        + recovery_sell_mwh.abs()
    )

    degradation_eur = float(
        throughput_mwh.sum() * float(DEGRADATION_EUR_PER_MWH)
    )

    return (
        net_recovery_cost_eur,
        degradation_eur,
        float(recovery_buy_mwh.sum()),
        float(recovery_sell_mwh.sum()),
        float(recovery_buy_cost.sum()),
        float(recovery_sell_revenue.sum()),
        float(throughput_mwh.sum()),
    )


def evaluate_activation_sample(
    cap: pd.DataFrame,
    sample: ActivationSample,
) -> dict[str, float | int]:
    dt = infer_interval_hours(cap["timestamp"])

    buy_price = pd.to_numeric(
        cap["imbalance_buy_price"],
        errors="coerce",
    ).fillna(0.0)

    sell_price = pd.to_numeric(
        cap["imbalance_sell_price"],
        errors="coerce",
    ).fillna(0.0)

    flag = pd.to_numeric(
        cap.get("balancing_flag_for_activation", pd.Series(0.0, index=cap.index)),
        errors="coerce",
    ).fillna(0.0)

    # -------------------------------------------------------------------------
    # FCR activation energy.
    # No activation payment is calculated.
    # -------------------------------------------------------------------------
    accepted_fcr = series_or_zero(
        cap,
        ["Accepted_FCR_MW", "accepted_fcr_mw"],
    ).clip(lower=0.0)

    fcr_total = sample.fcr_activation_fraction * accepted_fcr * dt
    fcr_up = fcr_total * sample.fcr_up_share
    fcr_down = fcr_total * (1.0 - sample.fcr_up_share)

    (
        fcr_recovery_cost,
        fcr_deg,
        fcr_rec_buy_mwh,
        fcr_rec_sell_mwh,
        fcr_rec_buy_cost,
        fcr_rec_sell_revenue,
        fcr_throughput,
    ) = recovery_cost_and_degradation(fcr_up, fcr_down, buy_price, sell_price)

    # -------------------------------------------------------------------------
    # aFRR activation energy.
    # Price-based settlement cashflow calculated below.
    # -------------------------------------------------------------------------
    afrr_up_accepted = series_or_zero(
        cap,
        ["Accepted_aFRR_UP_MW", "accepted_afrr_up_mw"],
    ).clip(lower=0.0)

    afrr_down_accepted = series_or_zero(
        cap,
        ["Accepted_aFRR_DOWN_MW", "accepted_afrr_down_mw"],
    ).clip(lower=0.0)

    if USE_BALANCING_FLAG_FOR_AFRR_MFRR:
        afrr_up_accepted = afrr_up_accepted.where(flag > 0.0, 0.0)
        afrr_down_accepted = afrr_down_accepted.where(flag < 0.0, 0.0)

    # A null market clearing price denotes no observed activation in that
    # product/direction/interval. Do not invent a settlement price. Restrict
    # simulated activation to intervals with an observed clearing price.
    afrr_up_eligible = afrr_up_accepted.where(cap["afrr_up_price_eur_mwh"].notna(), 0.0)
    afrr_down_eligible = afrr_down_accepted.where(cap["afrr_down_price_eur_mwh"].notna(), 0.0)
    afrr_up = sample.afrr_activation_fraction * afrr_up_eligible * dt
    afrr_down = sample.afrr_activation_fraction * afrr_down_eligible * dt

    (
        afrr_recovery_cost,
        afrr_deg,
        afrr_rec_buy_mwh,
        afrr_rec_sell_mwh,
        afrr_rec_buy_cost,
        afrr_rec_sell_revenue,
        afrr_throughput,
    ) = recovery_cost_and_degradation(afrr_up, afrr_down, buy_price, sell_price)

    # -------------------------------------------------------------------------
    # mFRR capacity-linked activation energy.
    # Price-based settlement cashflow calculated below.
    # -------------------------------------------------------------------------
    mfrr_up_accepted = series_or_zero(
        cap,
        ["Accepted_mFRR_UP_MW", "accepted_mfrr_up_mw"],
    ).clip(lower=0.0)

    mfrr_down_accepted = series_or_zero(
        cap,
        ["Accepted_mFRR_DOWN_MW", "accepted_mfrr_down_mw"],
    ).clip(lower=0.0)

    if USE_BALANCING_FLAG_FOR_AFRR_MFRR:
        mfrr_up_accepted = mfrr_up_accepted.where(flag > 0.0, 0.0)
        mfrr_down_accepted = mfrr_down_accepted.where(flag < 0.0, 0.0)

    mfrr_up_eligible = mfrr_up_accepted.where(cap["mfrr_up_price_eur_mwh"].notna(), 0.0)
    mfrr_down_eligible = mfrr_down_accepted.where(cap["mfrr_down_price_eur_mwh"].notna(), 0.0)
    mfrr_up = sample.mfrr_activation_fraction * mfrr_up_eligible * dt
    mfrr_down = sample.mfrr_activation_fraction * mfrr_down_eligible * dt

    (
        mfrr_recovery_cost,
        mfrr_deg,
        mfrr_rec_buy_mwh,
        mfrr_rec_sell_mwh,
        mfrr_rec_buy_cost,
        mfrr_rec_sell_revenue,
        mfrr_throughput,
    ) = recovery_cost_and_degradation(mfrr_up, mfrr_down, buy_price, sell_price)

    # -------------------------------------------------------------------------
    # Aggregate activation effect.
    #
    # Settlement-aware formula:
    # net_activation_adjustment = aFRR_cashflow + mFRR_cashflow
    #                             - recovery_cost - degradation
    # Downward activation is energy PURCHASE by BSP, so its sign is negative.
    # Prices may be negative, and results retain the corresponding sign.
    # -------------------------------------------------------------------------
    net_recovery_cost = (
        fcr_recovery_cost
        + afrr_recovery_cost
        + mfrr_recovery_cost
    )

    degradation = fcr_deg + afrr_deg + mfrr_deg

    # No aFRR/mFRR remuneration is buried in the existing S3 base: base
    # includes independently dispatched mFRR energy, which must NOT be added again.
    afrr_energy_cashflow = float((
        afrr_up * cap["afrr_up_price_eur_mwh"].fillna(0.0)
        - afrr_down * cap["afrr_down_price_eur_mwh"].fillna(0.0)
    ).sum())
    mfrr_energy_cashflow = float((
        mfrr_up * cap["mfrr_up_price_eur_mwh"].fillna(0.0)
        - mfrr_down * cap["mfrr_down_price_eur_mwh"].fillna(0.0)
    ).sum())
    activation_energy_cashflow = afrr_energy_cashflow + mfrr_energy_cashflow
    cost_only_adjustment = -net_recovery_cost - degradation
    net_activation_adjustment = activation_energy_cashflow + cost_only_adjustment

    recovery_buy_mwh = fcr_rec_buy_mwh + afrr_rec_buy_mwh + mfrr_rec_buy_mwh
    recovery_sell_mwh = fcr_rec_sell_mwh + afrr_rec_sell_mwh + mfrr_rec_sell_mwh

    recovery_buy_cost = fcr_rec_buy_cost + afrr_rec_buy_cost + mfrr_rec_buy_cost
    recovery_sell_revenue = (
        fcr_rec_sell_revenue
        + afrr_rec_sell_revenue
        + mfrr_rec_sell_revenue
    )

    throughput_mwh = fcr_throughput + afrr_throughput + mfrr_throughput

    return {
        "sample_id": sample.sample_id,
        "fcr_activation_fraction": sample.fcr_activation_fraction,
        "afrr_activation_fraction": sample.afrr_activation_fraction,
        "mfrr_activation_fraction": sample.mfrr_activation_fraction,
        "fcr_up_share": sample.fcr_up_share,

        # Main correction terms.
        "net_recovery_cost_eur": net_recovery_cost,
        "recovery_buy_cost_eur": recovery_buy_cost,
        "recovery_sell_revenue_eur": recovery_sell_revenue,
        "degradation_eur": degradation,
        "net_activation_adjustment_eur": net_activation_adjustment,
        "cost_only_adjustment_eur": cost_only_adjustment,
        "afrr_activation_energy_cashflow_eur": afrr_energy_cashflow,
        "mfrr_activation_energy_cashflow_eur": mfrr_energy_cashflow,
        "activation_energy_cashflow_eur": activation_energy_cashflow,
        "afrr_up_excluded_capacity_hours": float(((afrr_up_accepted - afrr_up_eligible) * dt).sum()),
        "afrr_down_excluded_capacity_hours": float(((afrr_down_accepted - afrr_down_eligible) * dt).sum()),
        "mfrr_up_excluded_capacity_hours": float(((mfrr_up_accepted - mfrr_up_eligible) * dt).sum()),
        "mfrr_down_excluded_capacity_hours": float(((mfrr_down_accepted - mfrr_down_eligible) * dt).sum()),
        "afrr_up_available_intervals": int(cap["afrr_up_price_eur_mwh"].notna().sum()),
        "afrr_down_available_intervals": int(cap["afrr_down_price_eur_mwh"].notna().sum()),
        "mfrr_up_available_intervals": int(cap["mfrr_up_price_eur_mwh"].notna().sum()),
        "mfrr_down_available_intervals": int(cap["mfrr_down_price_eur_mwh"].notna().sum()),

        # Product-level recovery costs.
        "fcr_net_recovery_cost_eur": fcr_recovery_cost,
        "afrr_net_recovery_cost_eur": afrr_recovery_cost,
        "mfrr_net_recovery_cost_eur": mfrr_recovery_cost,

        # Product-level degradation.
        "fcr_degradation_eur": fcr_deg,
        "afrr_degradation_eur": afrr_deg,
        "mfrr_degradation_eur": mfrr_deg,

        # Activated reserve energy.
        "fcr_activated_up_mwh": float(fcr_up.sum()),
        "fcr_activated_down_mwh": float(fcr_down.sum()),
        "afrr_activated_up_mwh": float(afrr_up.sum()),
        "afrr_activated_down_mwh": float(afrr_down.sum()),
        "mfrr_activated_up_mwh": float(mfrr_up.sum()),
        "mfrr_activated_down_mwh": float(mfrr_down.sum()),

        # Recovery and throughput.
        "recovery_buy_mwh": recovery_buy_mwh,
        "recovery_sell_mwh": recovery_sell_mwh,
        "throughput_mwh": throughput_mwh,
    }


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def summarize_series(x: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "p05": float(x.quantile(0.05)),
        "p25": float(x.quantile(0.25)),
        "p50": float(x.quantile(0.50)),
        "p75": float(x.quantile(0.75)),
        "p95": float(x.quantile(0.95)),
        "max": float(x.max()),
    }


def beta_scaled_mean(bounds: tuple[float, float], alpha: float, beta: float) -> float:
    lo, hi = bounds
    return float(lo + (hi - lo) * (float(alpha) / (float(alpha) + float(beta))))


def activation_assumption_table(samples_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    market_specs = [
        (
            "FCR",
            "fcr_activation_fraction",
            FCR_ACTIVATION_FRACTION_RANGE,
            FCR_ACTIVATION_BETA_ALPHA,
            FCR_ACTIVATION_BETA_BETA,
            "activation fraction; no direct activation payment",
        ),
        (
            "aFRR",
            "afrr_activation_fraction",
            AFRR_ACTIVATION_FRACTION_RANGE,
            AFRR_ACTIVATION_BETA_ALPHA,
            AFRR_ACTIVATION_BETA_BETA,
            "activation fraction; energy settlement included where observed clearing price exists",
        ),
        (
            "mFRR",
            "mfrr_activation_fraction",
            MFRR_ACTIVATION_FRACTION_RANGE,
            MFRR_ACTIVATION_BETA_ALPHA,
            MFRR_ACTIVATION_BETA_BETA,
            "activation fraction; energy settlement included where observed clearing price exists",
        ),
    ]

    for market, col, bounds, alpha, beta, note in market_specs:
        s = summarize_series(samples_df[col])

        rows.append(
            {
                "market": market,
                "distribution": ACTIVATION_FRACTION_DISTRIBUTION,
                "range_low": bounds[0],
                "range_high": bounds[1],
                "beta_alpha": alpha,
                "beta_beta": beta,
                "theoretical_mean": (
                    beta_scaled_mean(bounds, alpha, beta)
                    if ACTIVATION_FRACTION_DISTRIBUTION.strip().lower() == "beta"
                    else float(np.mean(bounds))
                ),
                "sample_mean": s["mean"],
                "sample_p05": s["p05"],
                "sample_p50": s["p50"],
                "sample_p95": s["p95"],
                "sample_max": s["max"],
                "note": note,
            }
        )

    fcr_dir = summarize_series(samples_df["fcr_up_share"])

    rows.append(
        {
            "market": "FCR direction",
            "distribution": "uniform",
            "range_low": FCR_UP_SHARE_RANGE[0],
            "range_high": FCR_UP_SHARE_RANGE[1],
            "beta_alpha": np.nan,
            "beta_beta": np.nan,
            "theoretical_mean": float(np.mean(FCR_UP_SHARE_RANGE)),
            "sample_mean": fcr_dir["mean"],
            "sample_p05": fcr_dir["p05"],
            "sample_p50": fcr_dir["p50"],
            "sample_p95": fcr_dir["p95"],
            "sample_max": fcr_dir["max"],
            "note": "UP share only; not activation probability",
        }
    )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_runs = list_base_run_dirs()

    if not base_runs:
        raise RuntimeError(
            f"No completed base runs found under {MC_ROOT}. "
            "Run Monte Carlo first."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    sample_rows: list[dict[str, Any]] = []
    base_summary_rows: list[dict[str, Any]] = []

    print("Random reserve-activation sensitivity over completed base scenarios")
    print("Settlement-corrected S3: aFRR + mFRR activation energy cashflows included.")
    print(f"Base runs found: {len(base_runs)}")
    print(f"Activation samples per base run: {N_ACTIVATION_SAMPLES_PER_BASE_SCENARIO}")
    print(f"Output: {OUTPUT_DIR}")

    for run_dir in base_runs:
        iteration, seed = parse_iteration_seed(run_dir)

        s3_dir = run_dir / "results" / S3_DIR_NAME
        s2_dir = run_dir / "results" / S2_DIR_NAME

        cap = load_capacity_results(s3_dir)
        cap = attach_settlement_prices(cap, find_actuals_csv(run_dir))

        s3_daily_summary = s3_dir / S3_DAILY_SUMMARY_NAME
        s2_daily_summary = s2_dir / S2_DAILY_SUMMARY_NAME

        s3_gross = sum_column_from_csv(
            s3_daily_summary,
            ["total_revenue_market_actual_eur"],
        )

        s3_energy = sum_column_from_csv(
            s3_daily_summary,
            ["da_net_cashflow_eur_actual"],
        )

        s3_reserve = sum_column_from_csv(
            s3_daily_summary,
            ["reserve_capacity_revenue_gross_eur"],
        )

        s2_total = sum_column_from_csv(
            s2_daily_summary,
            ["total_revenue_market_actual_eur"],
        )

        if s3_gross is None:
            raise ValueError(f"Could not read S3 gross from: {s3_daily_summary}")
        if s2_total is None:
            raise ValueError(f"Could not read S2 total from: {s2_daily_summary}")

        samples = make_random_samples(
            rng,
            N_ACTIVATION_SAMPLES_PER_BASE_SCENARIO,
        )

        for sample in samples:
            row = evaluate_activation_sample(cap, sample)

            s3_net = s3_gross + float(row["net_activation_adjustment_eur"])
            s3_cost_only = s3_gross + float(row["cost_only_adjustment_eur"])

            row.update(
                {
                    "base_run": run_dir.name,
                    "iteration": iteration,
                    "seed": seed,
                    "s3_gross_no_activation_eur": s3_gross,
                    "s3_energy_net_eur": s3_energy,
                    "s3_reserve_capacity_gross_eur": s3_reserve,
                    "s2_total_eur": s2_total,
                    "s3_activation_adjusted_net_eur": s3_net,
                    "s3_cost_only_net_eur": s3_cost_only,
                }
            )

            if s2_total is not None:
                row["s3_minus_s2_eur"] = s3_net - s2_total
                row["s3_cost_only_minus_s2_eur"] = s3_cost_only - s2_total

            sample_rows.append(row)

        recent = pd.DataFrame(sample_rows[-N_ACTIVATION_SAMPLES_PER_BASE_SCENARIO:])

        base_row: dict[str, Any] = {
            "base_run": run_dir.name,
            "iteration": iteration,
            "seed": seed,
            "s3_gross_no_activation_eur": s3_gross,
            "s3_energy_net_eur": s3_energy,
            "s3_reserve_capacity_gross_eur": s3_reserve,
            "s2_total_eur": s2_total,
        }

        for k, v in summarize_series(recent["s3_activation_adjusted_net_eur"]).items():
            base_row[f"s3_net_{k}"] = v
        for k, v in summarize_series(recent["s3_cost_only_net_eur"]).items():
            base_row[f"s3_cost_only_{k}"] = v
        for k, v in summarize_series(recent["activation_energy_cashflow_eur"]).items():
            base_row[f"activation_energy_cashflow_{k}"] = v

        for k, v in summarize_series(recent["net_recovery_cost_eur"]).items():
            base_row[f"net_recovery_cost_{k}"] = v

        for k, v in summarize_series(recent["degradation_eur"]).items():
            base_row[f"degradation_{k}"] = v

        if s2_total is not None:
            base_row["prob_s3_net_ge_s2"] = float(
                (recent["s3_activation_adjusted_net_eur"] >= s2_total).mean()
            )
            base_row["prob_s3_cost_only_ge_s2"] = float(
                (recent["s3_cost_only_net_eur"] >= s2_total).mean()
            )

        base_summary_rows.append(base_row)

        print(
            f"  {run_dir.name}: "
            f"S3 gross={s3_gross:.2f} EUR | "
            f"S3 net p05/p50/p95="
            f"{base_row['s3_net_p05']:.2f}/"
            f"{base_row['s3_net_p50']:.2f}/"
            f"{base_row['s3_net_p95']:.2f} EUR"
        )

    samples_df = pd.DataFrame(sample_rows)
    by_base_df = pd.DataFrame(base_summary_rows)

    # -------------------------------------------------------------------------
    # Overall summaries.
    # -------------------------------------------------------------------------
    overall_rows: list[dict[str, Any]] = []

    for metric in [
        "s3_activation_adjusted_net_eur",
        "s3_cost_only_net_eur",
        "s3_cost_only_minus_s2_eur",
        "afrr_activation_energy_cashflow_eur",
        "mfrr_activation_energy_cashflow_eur",
        "activation_energy_cashflow_eur",
        "net_activation_adjustment_eur",
        "net_recovery_cost_eur",
        "recovery_buy_cost_eur",
        "recovery_sell_revenue_eur",
        "degradation_eur",
        "throughput_mwh",
    ]:
        row = {"metric": metric}
        row.update(summarize_series(samples_df[metric]))
        overall_rows.append(row)

    if "s3_minus_s2_eur" in samples_df.columns:
        row = {"metric": "s3_minus_s2_eur"}
        row.update(summarize_series(samples_df["s3_minus_s2_eur"]))
        overall_rows.append(row)

        overall_rows.append(
            {
                "metric": "probability_s3_activation_adjusted_net_ge_s2",
                "mean": float((samples_df["s3_minus_s2_eur"] >= 0.0).mean()),
                "std": np.nan,
                "min": np.nan,
                "p05": np.nan,
                "p25": np.nan,
                "p50": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "max": np.nan,
            }
        )

    overall_df = pd.DataFrame(overall_rows)

    assumptions_df = activation_assumption_table(samples_df)

    # -------------------------------------------------------------------------
    # Save outputs.
    # -------------------------------------------------------------------------
    samples_path = OUTPUT_DIR / "all_random_activation_samples.csv"
    by_base_path = OUTPUT_DIR / "summary_by_base_scenario.csv"
    overall_path = OUTPUT_DIR / "summary_overall.csv"
    assumptions_path = OUTPUT_DIR / "activation_assumptions_summary.csv"
    config_path = OUTPUT_DIR / "random_activation_config.json"

    samples_df.to_csv(samples_path, index=False, encoding="utf-8-sig")
    by_base_df.to_csv(by_base_path, index=False, encoding="utf-8-sig")
    overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
    assumptions_df.to_csv(assumptions_path, index=False, encoding="utf-8-sig")

    config = {
        "n_base_runs": len(base_runs),
        "n_activation_samples_per_base_scenario": N_ACTIVATION_SAMPLES_PER_BASE_SCENARIO,
        "total_rows": int(len(samples_df)),
        "random_seed": RANDOM_SEED,
        "activation_fraction_distribution": ACTIVATION_FRACTION_DISTRIBUTION,
        "fcr_activation_fraction_range": FCR_ACTIVATION_FRACTION_RANGE,
        "afrr_activation_fraction_range": AFRR_ACTIVATION_FRACTION_RANGE,
        "mfrr_activation_fraction_range": MFRR_ACTIVATION_FRACTION_RANGE,
        "fcr_activation_beta_alpha": FCR_ACTIVATION_BETA_ALPHA,
        "fcr_activation_beta_beta": FCR_ACTIVATION_BETA_BETA,
        "afrr_activation_beta_alpha": AFRR_ACTIVATION_BETA_ALPHA,
        "afrr_activation_beta_beta": AFRR_ACTIVATION_BETA_BETA,
        "mfrr_activation_beta_alpha": MFRR_ACTIVATION_BETA_ALPHA,
        "mfrr_activation_beta_beta": MFRR_ACTIVATION_BETA_BETA,
        "fcr_activation_theoretical_mean": beta_scaled_mean(
            FCR_ACTIVATION_FRACTION_RANGE,
            FCR_ACTIVATION_BETA_ALPHA,
            FCR_ACTIVATION_BETA_BETA,
        ),
        "afrr_activation_theoretical_mean": beta_scaled_mean(
            AFRR_ACTIVATION_FRACTION_RANGE,
            AFRR_ACTIVATION_BETA_ALPHA,
            AFRR_ACTIVATION_BETA_BETA,
        ),
        "mfrr_activation_theoretical_mean": beta_scaled_mean(
            MFRR_ACTIVATION_FRACTION_RANGE,
            MFRR_ACTIVATION_BETA_ALPHA,
            MFRR_ACTIVATION_BETA_BETA,
        ),
        "fcr_up_share_range": FCR_UP_SHARE_RANGE,
        "use_balancing_flag_for_afrr_mfrr": USE_BALANCING_FLAG_FOR_AFRR_MFRR,
        "recovery_price_mode": RECOVERY_PRICE_MODE,
        "charge_efficiency": CHARGE_EFFICIENCY,
        "discharge_efficiency": DISCHARGE_EFFICIENCY,
        "degradation_eur_per_mwh": DEGRADATION_EUR_PER_MWH,
        "activation_energy_payment_included": True,
        "settlement_data_source": "Volton Public Data API (Elering/Baltic Transparency Dashboard)",
        "settlement_price_file": str(ROOT_DIR / "Data" / "settlement_prices.csv"),
        "null_price_policy": "no simulated FRR activation when the corresponding clearing price is missing",
        "s3_net_formula": (
            "s3_activation_adjusted_net_eur = s3_gross_no_activation_eur "
            "+ afrr_activation_energy_cashflow_eur + mfrr_activation_energy_cashflow_eur "
            "- net_recovery_cost_eur - degradation_eur"
        ),
        "mc_root": str(MC_ROOT),
        "output_dir": str(OUTPUT_DIR),
    }

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    net = samples_df["s3_activation_adjusted_net_eur"]

    print("\nDone.")
    print(f"Total activation samples: {len(samples_df)}")
    print(
        "Overall S3 activation-adjusted net: "
        f"mean={net.mean():.2f} EUR | "
        f"p05={net.quantile(0.05):.2f} | "
        f"p50={net.quantile(0.50):.2f} | "
        f"p95={net.quantile(0.95):.2f}"
    )

    if "s3_minus_s2_eur" in samples_df.columns:
        prob = float((samples_df["s3_minus_s2_eur"] >= 0.0).mean())
        print(f"Overall P(S3 activation-adjusted net >= S2): {prob:.3f}")

    print("\nFiles written:")
    print(f"Samples: {samples_path}")
    print(f"By-base summary: {by_base_path}")
    print(f"Overall summary: {overall_path}")
    print(f"Activation assumptions summary: {assumptions_path}")
    print(f"Config: {config_path}")

    print(
        "\nActivation means used: "
        f"FCR={beta_scaled_mean(FCR_ACTIVATION_FRACTION_RANGE, FCR_ACTIVATION_BETA_ALPHA, FCR_ACTIVATION_BETA_BETA):.3f}, "
        f"aFRR={beta_scaled_mean(AFRR_ACTIVATION_FRACTION_RANGE, AFRR_ACTIVATION_BETA_ALPHA, AFRR_ACTIVATION_BETA_BETA):.3f}, "
        f"mFRR={beta_scaled_mean(MFRR_ACTIVATION_FRACTION_RANGE, MFRR_ACTIVATION_BETA_ALPHA, MFRR_ACTIVATION_BETA_BETA):.3f}"
    )


if __name__ == "__main__":
    main()