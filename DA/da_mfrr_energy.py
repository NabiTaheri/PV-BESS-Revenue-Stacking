# DA/da_mfrr_energy.py
# Mixed-integer linear DA + mFRR energy optimizer.
# This replaces the previous marginal/greedy DA+mFRR stage with one horizon-level
# optimization problem. S2 and S3 then use the same objective; S3 differs only by
# tighter reservation constraints imported from the reservation CSV.
from __future__ import annotations

import json
import math
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
except Exception as exc:  # pragma: no cover
    milp = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

from market_config import *  # noqa: F401,F403

# -----------------------------------------------------------------------------
# Robust defaults when a specific market_config does not define a parameter.
# -----------------------------------------------------------------------------
EPS = float(globals().get("EPS", 1e-9))
DT_H = float(globals().get("DT_H", globals().get("DT", 0.25)))
FORECAST_FILE = Path(globals().get("FORECAST_FILE", "forecasts_rolling.csv"))
ACTUALS_FILE = Path(globals().get("ACTUALS_FILE", "actuals.csv"))
RESERVATION_FILE = Path(globals().get("RESERVATION_FILE", "battery_reservations_after_fcr_afrr_mfrr_capacity.csv"))
OUTPUT_REPORT = Path(globals().get("OUTPUT_REPORT", "da_energy_report.csv"))
OUTPUT_SUMMARY = globals().get("OUTPUT_SUMMARY", Path("da_energy_summary.csv"))
HORIZON = int(globals().get("HORIZON", globals().get("DEFAULT_HORIZON", 96)))

BATTERY_P_MAX_MW = float(globals().get("BATTERY_P_MAX_MW", 1.0))
BATTERY_E_MWH = float(globals().get("BATTERY_E_MWH", 2.0))
EFF = float(globals().get("EFF", 0.95))
INV_EFF = 1.0 / max(EFF, EPS)
PHYSICAL_SOC_MIN_PCT = float(globals().get("PHYSICAL_SOC_MIN_PCT", 20.0))
PHYSICAL_SOC_MAX_PCT = float(globals().get("PHYSICAL_SOC_MAX_PCT", 80.0))
SOC_INIT_PCT = float(globals().get("SOC_INIT_PCT", 50.0))
SOC_TARGET_PCT = float(globals().get("SOC_TARGET_PCT", 50.0))
SOC_TERMINAL_SYNC = bool(globals().get("SOC_TERMINAL_SYNC", True))
PV_MAX_MW = float(globals().get("PV_MAX_MW", 5.0))
GRID_LIMIT_MW = float(globals().get("GRID_LIMIT_MW", 10.0))
GRID_CHARGING_ALLOWED = bool(globals().get("GRID_CHARGING_ALLOWED", True))
ENABLE_BALANCING_MARKET = bool(globals().get("ENABLE_BALANCING_MARKET", globals().get("ENABLE_MFRR_ENERGY_MARKET", True)))
MFRR_ACTIVATION_FRACTION = float(globals().get("MFRR_ACTIVATION_FRACTION", os.environ.get("MFRR_ACTIVATION_FRACTION", 1.0)))
CYCLE_COST_EUR_PER_MWH = float(globals().get("CYCLE_COST_EUR_PER_MWH", 0.0))
PRINT_INTERVAL_TABLE = bool(globals().get("PRINT_INTERVAL_TABLE", False))


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _to_float_series(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(float)


def _first_existing(df: pd.DataFrame, names: list[str], default: float | str = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _must_have(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _cfg_value(cfg: Any | None, name: str, default: Any) -> Any:
    return getattr(cfg, name) if cfg is not None and hasattr(cfg, name) else default


def apply_config(cfg: Any | None) -> None:
    """Copy runner dataclass values into module-level constants."""
    if cfg is None:
        return

    mapping = {
        "forecast_file": "FORECAST_FILE",
        "actuals_file": "ACTUALS_FILE",
        "reservation_file": "RESERVATION_FILE",
        "output_report": "OUTPUT_REPORT",
        "output_summary": "OUTPUT_SUMMARY",
        "soc_init_pct": "SOC_INIT_PCT",
        "soc_target_pct": "SOC_TARGET_PCT",
        "soc_terminal_sync": "SOC_TERMINAL_SYNC",
        "rolling_terminal_soc_sync": "SOC_TERMINAL_SYNC",
        "battery_p_max_mw": "BATTERY_P_MAX_MW",
        "battery_e_mwh": "BATTERY_E_MWH",
        "eff": "EFF",
        "physical_soc_min_pct": "PHYSICAL_SOC_MIN_PCT",
        "physical_soc_max_pct": "PHYSICAL_SOC_MAX_PCT",
        "pv_max_mw": "PV_MAX_MW",
        "grid_limit_mw": "GRID_LIMIT_MW",
        "grid_charging_allowed": "GRID_CHARGING_ALLOWED",
        "cycle_cost_eur_per_mwh": "CYCLE_COST_EUR_PER_MWH",
        "enable_mfrr_energy_market": "ENABLE_BALANCING_MARKET",
        "enable_balancing_market": "ENABLE_BALANCING_MARKET",
        "mfrr_activation_fraction": "MFRR_ACTIVATION_FRACTION",
        "print_interval_table": "PRINT_INTERVAL_TABLE",
        "horizon": "HORIZON",
    }

    g = globals()
    for cfg_name, global_name in mapping.items():
        if hasattr(cfg, cfg_name):
            value = getattr(cfg, cfg_name)
            if global_name in {"FORECAST_FILE", "ACTUALS_FILE", "RESERVATION_FILE", "OUTPUT_REPORT", "OUTPUT_SUMMARY"}:
                value = _as_path(value) if value is not None else value
            g[global_name] = value

    g["MFRR_ACTIVATION_FRACTION"] = max(0.0, min(1.0, float(g.get("MFRR_ACTIVATION_FRACTION", os.environ.get("MFRR_ACTIVATION_FRACTION", 1.0)))))
    g["EFF"] = float(g.get("EFF", 0.95))
    g["INV_EFF"] = 1.0 / max(float(g["EFF"]), EPS)
    g["SOC_TARGET_PCT"] = float(g.get("SOC_TARGET_PCT", 50.0))
    # The daily chain expects exact terminal synchronisation.
    g["SOC_TERMINAL_SYNC"] = True


def load_data() -> pd.DataFrame:
    forecast_file = _as_path(FORECAST_FILE)
    actuals_file = _as_path(ACTUALS_FILE)
    reservation_file = _as_path(RESERVATION_FILE)

    if not forecast_file.exists():
        raise FileNotFoundError(f"Forecast file not found: {forecast_file.resolve()}")
    if not actuals_file.exists():
        raise FileNotFoundError(f"Actuals file not found: {actuals_file.resolve()}")

    fc = pd.read_csv(forecast_file)
    ac = pd.read_csv(actuals_file)
    _must_have(fc, ["timestamp", "pv_forecast_mw", "da_price_forecast"], "forecast")
    _must_have(ac, ["timestamp", "pv_actual_mw", "da_clearing_price"], "actuals")

    fc = fc.copy()
    ac = ac.copy()
    fc["timestamp"] = pd.to_datetime(fc["timestamp"], errors="coerce")
    ac["timestamp"] = pd.to_datetime(ac["timestamp"], errors="coerce")

    df = pd.merge(fc, ac, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("No matching timestamps between forecast and actuals.")

    if int(globals().get("HORIZON", 96)) > 0:
        df = df.iloc[: int(globals().get("HORIZON", 96))].copy().reset_index(drop=True)

    if reservation_file.exists():
        rs = pd.read_csv(reservation_file)
        if "timestamp" in rs.columns:
            rs = rs.copy()
            rs["timestamp"] = pd.to_datetime(rs["timestamp"], errors="coerce")
            cols = ["timestamp"] + [c for c in ["p_charge_max_mw", "p_discharge_max_mw", "soc_min_pct", "soc_max_pct"] if c in rs.columns]
            df = pd.merge(df, rs[cols], on="timestamp", how="left")
        elif len(rs) == len(df):
            for col in ["p_charge_max_mw", "p_discharge_max_mw", "soc_min_pct", "soc_max_pct"]:
                if col in rs.columns:
                    df[col] = rs[col].values

    for col, default in [
        ("p_charge_max_mw", BATTERY_P_MAX_MW),
        ("p_discharge_max_mw", BATTERY_P_MAX_MW),
        ("soc_min_pct", PHYSICAL_SOC_MIN_PCT),
        ("soc_max_pct", PHYSICAL_SOC_MAX_PCT),
    ]:
        if col not in df.columns:
            df[col] = default

    df["p_charge_max_mw"] = _to_float_series(df["p_charge_max_mw"], BATTERY_P_MAX_MW).clip(0.0, BATTERY_P_MAX_MW)
    df["p_discharge_max_mw"] = _to_float_series(df["p_discharge_max_mw"], BATTERY_P_MAX_MW).clip(0.0, BATTERY_P_MAX_MW)
    df["soc_min_pct"] = _to_float_series(df["soc_min_pct"], PHYSICAL_SOC_MIN_PCT).clip(0.0, 100.0)
    df["soc_max_pct"] = _to_float_series(df["soc_max_pct"], PHYSICAL_SOC_MAX_PCT).clip(0.0, 100.0)

    df["pv_forecast_mw"] = _to_float_series(df["pv_forecast_mw"], 0.0).clip(0.0, PV_MAX_MW)
    df["pv_actual_mw"] = _to_float_series(df["pv_actual_mw"], 0.0).clip(0.0, PV_MAX_MW)
    df["da_price_forecast"] = _to_float_series(df["da_price_forecast"], 0.0)
    df["da_clearing_price"] = _to_float_series(df["da_clearing_price"], 0.0)

    df["balancing_price_forecast"] = _to_float_series(
        _first_existing(df, ["balancing_price_forecast", "mfrr_energy_price_forecast", "mfrr_price_forecast", "imbalance_price_forecast"], 0.0),
        0.0,
    )
    df["balancing_price"] = _to_float_series(
        _first_existing(df, ["balancing_price", "balancing_price_actual", "mfrr_energy_price_actual", "mfrr_price_actual", "imbalance_price_actual"], 0.0),
        0.0,
    )
    df["imbalance_price_actual"] = _to_float_series(
        _first_existing(df, ["imbalance_price_actual", "imbalance_price", "balancing_price", "balancing_price_actual"], 0.0),
        0.0,
    )

    if "balancing_flag_forecast" in df.columns:
        df["balancing_flag_forecast"] = _to_float_series(df["balancing_flag_forecast"], 0.0).clip(-1.0, 1.0)
    elif "balancing_flag" in df.columns:
        df["balancing_flag_forecast"] = _to_float_series(df["balancing_flag"], 0.0).clip(-1.0, 1.0)
    else:
        df["balancing_flag_forecast"] = 0.0
        df.loc[df["balancing_price_forecast"] > 0.0, "balancing_flag_forecast"] = 1.0
        df.loc[df["balancing_price_forecast"] < 0.0, "balancing_flag_forecast"] = -1.0

    if "balancing_flag" in df.columns:
        df["balancing_flag"] = _to_float_series(df["balancing_flag"], 0.0).clip(-1.0, 1.0)
    else:
        df["balancing_flag"] = df["balancing_flag_forecast"]

    df["soc_min_mwh_used"] = df["soc_min_pct"] / 100.0 * BATTERY_E_MWH
    df["soc_max_mwh_used"] = df["soc_max_pct"] / 100.0 * BATTERY_E_MWH
    return df.reset_index(drop=True)


def _target_soc_mwh() -> float:
    return SOC_TARGET_PCT / 100.0 * BATTERY_E_MWH


class _VarIndex:
    names = [
        "grid_charge",
        "pv_charge",
        "battery_discharge",
        "pv_export",
        "pv_curtail",
        "mfrr_up",
        "mfrr_down",
        "charge_mode",
    ]

    def __init__(self, n: int):
        self.n = n
        self.block = len(self.names)
        self.offset_soc = self.block * n
        self.total = self.offset_soc + n + 1
        self.name_to_pos = {name: i for i, name in enumerate(self.names)}

    def x(self, name: str, t: int) -> int:
        return t * self.block + self.name_to_pos[name]

    def soc(self, t: int) -> int:
        return self.offset_soc + t


def optimize_dispatch(df: pd.DataFrame) -> pd.DataFrame:
    if milp is None:
        raise RuntimeError(
            "scipy is required for the MILP DA+mFRR optimizer. Install it with: pip install scipy"
        ) from _SCIPY_IMPORT_ERROR

    n = len(df)
    vi = _VarIndex(n)
    c = [0.0] * vi.total
    lb = [0.0] * vi.total
    ub = [math.inf] * vi.total
    integrality = [0] * vi.total

    deg_cost = CYCLE_COST_EUR_PER_MWH / 2.0
    balancing_enabled = bool(ENABLE_BALANCING_MARKET)
    mfrr_activation_fraction = max(0.0, min(1.0, float(MFRR_ACTIVATION_FRACTION)))

    for t, row in df.iterrows():
        p_ch = max(0.0, float(row["p_charge_max_mw"]))
        p_dis = max(0.0, float(row["p_discharge_max_mw"]))
        pv = max(0.0, float(row["pv_forecast_mw"]))
        da_price = float(row["da_price_forecast"])
        bal_price = float(row["balancing_price_forecast"])
        flag = float(row["balancing_flag_forecast"])

        ub[vi.x("grid_charge", t)] = min(p_ch, GRID_LIMIT_MW) if GRID_CHARGING_ALLOWED else 0.0
        ub[vi.x("pv_charge", t)] = min(p_ch, pv)
        ub[vi.x("battery_discharge", t)] = p_dis
        ub[vi.x("pv_export", t)] = pv
        ub[vi.x("pv_curtail", t)] = pv
        ub[vi.x("mfrr_up", t)] = p_dis if balancing_enabled and mfrr_activation_fraction > EPS and flag > 0.0 else 0.0
        ub[vi.x("mfrr_down", t)] = p_ch if balancing_enabled and mfrr_activation_fraction > EPS and flag < 0.0 else 0.0
        ub[vi.x("charge_mode", t)] = 1.0
        integrality[vi.x("charge_mode", t)] = 1

        # Maximize forecast cashflow minus degradation, implemented as minimization.
        c[vi.x("pv_export", t)] = -da_price * DT_H
        c[vi.x("battery_discharge", t)] = -da_price * DT_H + deg_cost * DT_H
        c[vi.x("grid_charge", t)] = da_price * DT_H + deg_cost * DT_H
        c[vi.x("pv_charge", t)] = deg_cost * DT_H
        c[vi.x("mfrr_up", t)] = -mfrr_activation_fraction * bal_price * DT_H + mfrr_activation_fraction * deg_cost * DT_H
        c[vi.x("mfrr_down", t)] = mfrr_activation_fraction * bal_price * DT_H + mfrr_activation_fraction * deg_cost * DT_H
        c[vi.x("pv_curtail", t)] = 0.0

    init_soc = SOC_INIT_PCT / 100.0 * BATTERY_E_MWH
    target_soc = _target_soc_mwh()
    for t in range(n + 1):
        if t == 0:
            lb[vi.soc(t)] = init_soc
            ub[vi.soc(t)] = init_soc
        elif t == n and SOC_TERMINAL_SYNC:
            lb[vi.soc(t)] = target_soc
            ub[vi.soc(t)] = target_soc
        else:
            row_idx = min(t, n - 1)
            lb[vi.soc(t)] = float(df.loc[row_idx, "soc_min_mwh_used"])
            ub[vi.soc(t)] = float(df.loc[row_idx, "soc_max_mwh_used"])

    A_eq: list[list[float]] = []
    b_eq: list[float] = []
    A_ub: list[list[float]] = []
    b_ub: list[float] = []

    def row0() -> list[float]:
        return [0.0] * vi.total

    for t, row in df.iterrows():
        # PV allocation balance.
        eq = row0()
        eq[vi.x("pv_export", t)] = 1.0
        eq[vi.x("pv_charge", t)] = 1.0
        eq[vi.x("pv_curtail", t)] = 1.0
        A_eq.append(eq)
        b_eq.append(float(row["pv_forecast_mw"]))

        # SOC dynamics.
        eq = row0()
        eq[vi.soc(t + 1)] = 1.0
        eq[vi.soc(t)] = -1.0
        eq[vi.x("grid_charge", t)] = -EFF * DT_H
        eq[vi.x("pv_charge", t)] = -EFF * DT_H
        eq[vi.x("mfrr_down", t)] = -EFF * DT_H * mfrr_activation_fraction
        eq[vi.x("battery_discharge", t)] = INV_EFF * DT_H
        eq[vi.x("mfrr_up", t)] = INV_EFF * DT_H * mfrr_activation_fraction
        A_eq.append(eq)
        b_eq.append(0.0)

        p_ch = float(row["p_charge_max_mw"])
        p_dis = float(row["p_discharge_max_mw"])

        # Battery charge power limit.
        ub_row = row0()
        ub_row[vi.x("grid_charge", t)] = 1.0
        ub_row[vi.x("pv_charge", t)] = 1.0
        ub_row[vi.x("mfrr_down", t)] = mfrr_activation_fraction
        A_ub.append(ub_row)
        b_ub.append(p_ch)

        # Battery discharge power limit.
        ub_row = row0()
        ub_row[vi.x("battery_discharge", t)] = 1.0
        ub_row[vi.x("mfrr_up", t)] = mfrr_activation_fraction
        A_ub.append(ub_row)
        b_ub.append(p_dis)

        # Grid export/import limits.
        ub_row = row0()
        ub_row[vi.x("pv_export", t)] = 1.0
        ub_row[vi.x("battery_discharge", t)] = 1.0
        ub_row[vi.x("mfrr_up", t)] = mfrr_activation_fraction
        A_ub.append(ub_row)
        b_ub.append(GRID_LIMIT_MW)

        ub_row = row0()
        ub_row[vi.x("grid_charge", t)] = 1.0
        ub_row[vi.x("mfrr_down", t)] = mfrr_activation_fraction
        A_ub.append(ub_row)
        b_ub.append(GRID_LIMIT_MW)

        # Binary operating mode: charge-side actions and discharge-side actions
        # cannot be active simultaneously in the same interval.
        mode = row0()
        mode[vi.x("grid_charge", t)] = 1.0
        mode[vi.x("pv_charge", t)] = 1.0
        mode[vi.x("mfrr_down", t)] = 1.0
        mode[vi.x("charge_mode", t)] = -p_ch
        A_ub.append(mode)
        b_ub.append(0.0)

        mode = row0()
        mode[vi.x("battery_discharge", t)] = 1.0
        mode[vi.x("mfrr_up", t)] = 1.0
        mode[vi.x("charge_mode", t)] = p_dis
        A_ub.append(mode)
        b_ub.append(p_dis)

    rows_l = []
    rows_u = []
    mats = []
    for row, rhs in zip(A_eq, b_eq):
        mats.append(row)
        rows_l.append(rhs)
        rows_u.append(rhs)
    for row, rhs in zip(A_ub, b_ub):
        mats.append(row)
        rows_l.append(-math.inf)
        rows_u.append(rhs)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(mats, rows_l, rows_u),
        options={"time_limit": 300.0, "mip_rel_gap": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"DA+mFRR MILP optimization failed: {result.message}")

    x = result.x
    rows: list[dict[str, Any]] = []
    for t, row in df.iterrows():
        grid_charge = max(0.0, float(x[vi.x("grid_charge", t)]))
        pv_charge = max(0.0, float(x[vi.x("pv_charge", t)]))
        dis = max(0.0, float(x[vi.x("battery_discharge", t)]))
        pv_export = max(0.0, float(x[vi.x("pv_export", t)]))
        pv_curtail = max(0.0, float(x[vi.x("pv_curtail", t)]))
        mfup = max(0.0, float(x[vi.x("mfrr_up", t)]))
        mfdown = max(0.0, float(x[vi.x("mfrr_down", t)]))
        soc_next = float(x[vi.soc(t + 1)])

        da_net = pv_export + dis - grid_charge
        da_cash_actual = da_net * float(row["da_clearing_price"]) * DT_H
        da_cash_forecast = da_net * float(row["da_price_forecast"]) * DT_H
        mfrr_up_activated = mfrr_activation_fraction * mfup
        mfrr_down_activated = mfrr_activation_fraction * mfdown
        mfrr_cash_actual = (mfrr_up_activated - mfrr_down_activated) * float(row["balancing_price"]) * DT_H
        mfrr_cash_forecast = (mfrr_up_activated - mfrr_down_activated) * float(row["balancing_price_forecast"]) * DT_H
        throughput = DT_H * (grid_charge + pv_charge + dis + mfrr_up_activated + mfrr_down_activated)
        degradation = deg_cost * throughput
        net_actual = da_cash_actual + mfrr_cash_actual - degradation
        net_forecast = da_cash_forecast + mfrr_cash_forecast - degradation

        rows.append(
            {
                "interval": int(t) + 1,
                "timestamp": row["timestamp"],
                "engine": "MILP_DA_MFRR_ENERGY_SHARED_OBJECTIVE",
                "da_price_forecast": float(row["da_price_forecast"]),
                "da_clearing_price": float(row["da_clearing_price"]),
                "balancing_price_forecast": float(row["balancing_price_forecast"]),
                "balancing_price": float(row["balancing_price"]),
                "balancing_flag_forecast": float(row["balancing_flag_forecast"]),
                "balancing_flag": float(row["balancing_flag"]),
                "mfrr_energy_activation_fraction": float(mfrr_activation_fraction),
                "p_charge_max_mw": float(row["p_charge_max_mw"]),
                "p_discharge_max_mw": float(row["p_discharge_max_mw"]),
                "soc_min_pct": float(row["soc_min_pct"]),
                "soc_max_pct": float(row["soc_max_pct"]),
                "pv_forecast_mw": float(row["pv_forecast_mw"]),
                "pv_actual_mw": float(row["pv_actual_mw"]),
                "da_capacity_value_eur": 0.0,
                "balancing_capacity_value_eur": 0.0,
                "da_reserved_power_mw": float(row["p_discharge_max_mw"]),
                "balancing_reserved_power_mw": float(row["p_discharge_max_mw"] if ENABLE_BALANCING_MARKET else 0.0),
                "da_bid_min_mw": -float(row["p_charge_max_mw"]),
                "da_bid_max_mw": float(row["p_discharge_max_mw"]) + float(row["pv_forecast_mw"]),
                "da_bid_mw": float(da_net),
                "da_accepted_mw": float(da_net),
                "da_delivered_mw": float(da_net),
                "da_undelivered_mw": 0.0,
                "mfrr_up_max_bid_mw": float(row["p_discharge_max_mw"]),
                "mfrr_down_max_bid_mw": float(row["p_charge_max_mw"]),
                "mfrr_up_accepted_mw": float(mfup),
                "mfrr_down_accepted_mw": float(mfdown),
                "mfrr_up_activated_mw": float(mfrr_up_activated),
                "mfrr_down_activated_mw": float(mfrr_down_activated),
                "battery_grid_charge_mw": float(grid_charge),
                "battery_pv_charge_mw": float(pv_charge),
                "battery_discharge_mw": float(dis),
                "pv_export_mw": float(pv_export),
                "pv_curtail_mw": float(pv_curtail),
                "terminal_recovery_charge_mw": 0.0,
                "terminal_recovery_discharge_mw": 0.0,
                "da_cashflow_eur_forecast": float(da_cash_forecast),
                "mfrr_up_cashflow_eur_forecast": float(mfrr_up_activated * float(row["balancing_price_forecast"]) * DT_H),
                "mfrr_down_cashflow_eur_forecast": float(-mfrr_down_activated * float(row["balancing_price_forecast"]) * DT_H),
                "mfrr_energy_cashflow_eur_forecast": float(mfrr_cash_forecast),
                "imbalance_penalty_eur_forecast": 0.0,
                "battery_degradation_eur_forecast": float(degradation),
                "net_cashflow_eur_forecast": float(net_forecast),
                "da_cashflow_eur_actual": float(da_cash_actual),
                "mfrr_up_cashflow_eur_actual": float(mfrr_up_activated * float(row["balancing_price"]) * DT_H),
                "mfrr_down_cashflow_eur_actual": float(-mfrr_down_activated * float(row["balancing_price"]) * DT_H),
                "mfrr_energy_cashflow_eur_actual": float(mfrr_cash_actual),
                "imbalance_penalty_eur_actual": 0.0,
                "battery_degradation_eur": float(degradation),
                "battery_degradation_eur_actual": float(degradation),
                "net_cashflow_eur_actual": float(net_actual),
                "market_net_cashflow_eur_actual": float(net_actual),
                "market_net_eur_actual": float(net_actual),
                "soc_target_mwh": float(_target_soc_mwh()),
                "soc_target_pct": float(SOC_TARGET_PCT),
                "soc_after_all_mwh": float(soc_next),
                "soc_after_all_pct": 100.0 * float(soc_next) / BATTERY_E_MWH,
            }
        )

    return pd.DataFrame(rows)


def _sum(report: pd.DataFrame, col: str) -> float:
    if col not in report.columns:
        return 0.0
    return float(pd.to_numeric(report[col], errors="coerce").fillna(0.0).sum())


def _summary_from_report(report: pd.DataFrame) -> dict[str, Any]:
    return {
        "status": "Optimal",
        "engine": "MILP_DA_MFRR_ENERGY_SHARED_OBJECTIVE",
        "mfrr_energy_activation_fraction": float(MFRR_ACTIVATION_FRACTION),
        "intervals": int(len(report)),
        "soc_init_pct": float(SOC_INIT_PCT),
        "soc_final_pct": float(report["soc_after_all_pct"].iloc[-1]) if len(report) else float(SOC_INIT_PCT),
        "da_cashflow_eur_actual": _sum(report, "da_cashflow_eur_actual"),
        "mfrr_energy_cashflow_eur_actual": _sum(report, "mfrr_energy_cashflow_eur_actual"),
        "imbalance_penalty_eur_actual": _sum(report, "imbalance_penalty_eur_actual"),
        "battery_degradation_eur": _sum(report, "battery_degradation_eur"),
        "market_net_eur_actual": _sum(report, "market_net_cashflow_eur_actual"),
        "market_net_cashflow_eur_actual": _sum(report, "market_net_cashflow_eur_actual"),
        "net_cashflow_eur_actual": _sum(report, "net_cashflow_eur_actual"),
        "da_cashflow_eur_forecast": _sum(report, "da_cashflow_eur_forecast"),
        "mfrr_energy_cashflow_eur_forecast": _sum(report, "mfrr_energy_cashflow_eur_forecast"),
        "battery_degradation_eur_forecast": _sum(report, "battery_degradation_eur_forecast"),
        "net_cashflow_eur_forecast": _sum(report, "net_cashflow_eur_forecast"),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n========== MILP DA + mFRR ENERGY SUMMARY ==========")
    print(f"Intervals: {summary['intervals']}")
    print(f"DA cashflow actual: {summary['da_cashflow_eur_actual']:.2f} EUR")
    print(f"mFRR energy cashflow actual: {summary['mfrr_energy_cashflow_eur_actual']:.2f} EUR")
    print(f"Battery degradation: {summary['battery_degradation_eur']:.2f} EUR")
    print(f"Market net actual: {summary['market_net_eur_actual']:.2f} EUR")
    print(f"Final SoC: {summary['soc_final_pct']:.4f}%")


def run(cfg: Any | None = None) -> pd.DataFrame:
    """Run one horizon-level MILP for DA + optional mFRR energy.

    S2 and S3 use this exact same objective. S3 only differs because the
    reservation CSV reduces the available battery power/SOC range. Therefore,
    under perfect forecasts, S3's DA+mFRR energy value cannot exceed S2's.
    """
    apply_config(cfg)
    df = load_data()
    report = optimize_dispatch(df)

    output_report = _as_path(OUTPUT_REPORT)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_report, index=False, encoding="utf-8-sig")

    summary = _summary_from_report(report)
    report.attrs["summary"] = summary

    output_summary = globals().get("OUTPUT_SUMMARY", None)
    if output_summary is not None:
        output_summary = _as_path(output_summary)
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(output_summary, index=False, encoding="utf-8-sig")
        with open(output_summary.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if bool(_cfg_value(cfg, "print_interval_table", PRINT_INTERVAL_TABLE)):
        cols = [
            "timestamp", "da_accepted_mw", "mfrr_up_accepted_mw", "mfrr_down_accepted_mw",
            "battery_grid_charge_mw", "battery_pv_charge_mw", "battery_discharge_mw",
            "soc_after_all_pct", "net_cashflow_eur_actual",
        ]
        print(report[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print_summary(summary)
    print(f"\nDetailed CSV saved to: {output_report}")
    return report


def main() -> None:
    run(None)


if __name__ == "__main__":
    main()
