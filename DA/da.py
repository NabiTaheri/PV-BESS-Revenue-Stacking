# DA/da_mfrr_energy.py
# Scenario-runner-compatible DA-only engine.
from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pulp

from market_config import *


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def floor_to_lot(x: float, lot: float) -> float:
    return math.floor(float(x) / lot + EPS) * lot


def ceil_to_lot(x: float, lot: float) -> float:
    return math.ceil(float(x) / lot - EPS) * lot


def round_to_lot(x: float, lot: float) -> float:
    return round(float(x) / lot) * lot


def quantize_to_lot(x: float, lot: float, mode: str = "nearest") -> float:
    if abs(x) < EPS:
        return 0.0
    if mode == "floor":
        return floor_to_lot(x, lot)
    if mode == "ceil":
        return ceil_to_lot(x, lot)
    return round_to_lot(x, lot)


def must_have(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _cfg_value(cfg: Any | None, name: str, default: Any) -> Any:
    if cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    return default


def apply_config(cfg: Any | None) -> None:
    """Copy a runner dataclass config into module-level constants.

    This keeps compatibility with `run_S1_DA_only.py`, which imports
    `MarketConfig` and calls `run(cfg)`.
    """
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
        "battery_p_max_mw": "BATTERY_P_MAX_MW",
        "battery_e_mwh": "BATTERY_E_MWH",
        "eff": "EFF",
        "physical_soc_min_pct": "PHYSICAL_SOC_MIN_PCT",
        "physical_soc_max_pct": "PHYSICAL_SOC_MAX_PCT",
        "pv_max_mw": "PV_MAX_MW",
        "grid_limit_mw": "GRID_LIMIT_MW",
        "grid_charging_allowed": "GRID_CHARGING_ALLOWED",
        "da_trade_lot_mw": "DA_TRADE_LOT_MW",
        "cycle_cost_eur_per_mwh": "CYCLE_COST_EUR_PER_MWH",
        "block_da_export_below_floor": "BLOCK_DA_EXPORT_BELOW_FLOOR",
        "da_export_price_floor_eur_per_mwh": "DA_EXPORT_PRICE_FLOOR_EUR_PER_MWH",
        "da_backtest_physical_delivery_guard": "DA_BACKTEST_PHYSICAL_DELIVERY_GUARD",
        "da_backtest_no_loss_guard": "DA_BACKTEST_NO_LOSS_GUARD",
        "optional_participation_no_loss": "OPTIONAL_PARTICIPATION_NO_LOSS",
        "auto_charge_excess_pv": "AUTO_CHARGE_EXCESS_PV",
        "no_trade_if_forecast_objective_nonpositive": "NO_TRADE_IF_FORECAST_OBJECTIVE_NONPOSITIVE",
    }

    g = globals()
    for cfg_name, global_name in mapping.items():
        if hasattr(cfg, cfg_name):
            value = getattr(cfg, cfg_name)
            if global_name.endswith("_FILE") or global_name in {"OUTPUT_REPORT", "OUTPUT_SUMMARY"}:
                value = _as_path(value) if value is not None else value
            g[global_name] = value

    # Daily operation requirement: the DA stage must always end at SOC_TARGET_PCT.
    # The scenario runner may pass final_da_terminal_sync=False for chaining, but
    # this engine intentionally chains around the fixed daily target instead.
    g["SOC_TARGET_PCT"] = float(g.get("SOC_TARGET_PCT", 50.0))
    g["SOC_TERMINAL_SYNC"] = True

    g["INV_EFF"] = 1.0 / float(g["EFF"])


def make_solver_safe() -> pulp.LpSolver:
    try:
        return make_solver()
    except Exception:
        pass

    try:
        return pulp.PULP_CBC_CMD(msg=False)
    except Exception as exc:
        raise RuntimeError("No usable PuLP MILP solver found. Install GLPK or CBC.") from exc


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

    must_have(fc, ["timestamp", "pv_forecast_mw", "da_price_forecast"], "forecast")
    must_have(ac, ["timestamp", "pv_actual_mw", "da_clearing_price"], "actuals")

    fc = fc.copy()
    ac = ac.copy()
    fc["timestamp"] = pd.to_datetime(fc["timestamp"])
    ac["timestamp"] = pd.to_datetime(ac["timestamp"])

    df = pd.merge(fc, ac, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("No matching timestamps between forecast and actuals.")

    if reservation_file.exists():
        rs = pd.read_csv(reservation_file)
        must_have(rs, ["timestamp", "p_charge_max_mw", "p_discharge_max_mw"], "reservation")
        rs = rs.copy()
        rs["timestamp"] = pd.to_datetime(rs["timestamp"], errors="coerce")
        rs = rs[["timestamp", "p_charge_max_mw", "p_discharge_max_mw"]]
        df = pd.merge(df, rs, on="timestamp", how="left")
    else:
        df["p_charge_max_mw"] = BATTERY_P_MAX_MW
        df["p_discharge_max_mw"] = BATTERY_P_MAX_MW

    df["p_charge_max_mw"] = (
        pd.to_numeric(df["p_charge_max_mw"], errors="coerce")
        .fillna(BATTERY_P_MAX_MW)
        .clip(0.0, BATTERY_P_MAX_MW)
    )
    df["p_discharge_max_mw"] = (
        pd.to_numeric(df["p_discharge_max_mw"], errors="coerce")
        .fillna(BATTERY_P_MAX_MW)
        .clip(0.0, BATTERY_P_MAX_MW)
    )

    df["pv_forecast_mw"] = pd.to_numeric(df["pv_forecast_mw"], errors="coerce").fillna(0.0).clip(0.0, PV_MAX_MW)
    df["pv_actual_mw"] = pd.to_numeric(df["pv_actual_mw"], errors="coerce").fillna(0.0).clip(0.0, PV_MAX_MW)
    df["da_price_forecast"] = pd.to_numeric(df["da_price_forecast"], errors="coerce").fillna(0.0)
    df["da_clearing_price"] = pd.to_numeric(df["da_clearing_price"], errors="coerce").fillna(0.0)

    df["soc_min_mwh_used"] = PHYSICAL_SOC_MIN_PCT / 100.0 * BATTERY_E_MWH
    df["soc_max_mwh_used"] = PHYSICAL_SOC_MAX_PCT / 100.0 * BATTERY_E_MWH
    return df


def _physical_net_bounds(
    *,
    soc_mwh: float,
    pv_mw: float,
    p_charge_avail_mw: float,
    p_discharge_avail_mw: float,
    soc_min_mwh: float,
    soc_max_mwh: float,
) -> tuple[float, float]:
    charge_room_mw = max(0.0, (soc_max_mwh - soc_mwh) / max(EFF * DT_H, EPS))
    discharge_room_mw = max(0.0, (soc_mwh - soc_min_mwh) * EFF / max(DT_H, EPS))

    max_import_mw = min(GRID_LIMIT_MW, p_charge_avail_mw, charge_room_mw)
    max_battery_discharge_mw = min(p_discharge_avail_mw, discharge_room_mw)
    max_export_mw = min(GRID_LIMIT_MW, pv_mw + max_battery_discharge_mw)

    return -max_import_mw, max_export_mw


def solve_da_plan(df: pd.DataFrame) -> pd.DataFrame:
    t_idx = list(range(len(df)))
    model = pulp.LpProblem("DA_only_forecast_plan", pulp.LpMaximize)

    pv_grid = pulp.LpVariable.dicts("pv_grid", t_idx, lowBound=0)
    pv_batt = pulp.LpVariable.dicts("pv_batt", t_idx, lowBound=0)
    pv_curt = pulp.LpVariable.dicts("pv_curt", t_idx, lowBound=0)
    grid_charge = pulp.LpVariable.dicts("grid_charge", t_idx, lowBound=0)
    discharge = pulp.LpVariable.dicts("discharge", t_idx, lowBound=0)
    soc = pulp.LpVariable.dicts("soc", t_idx, lowBound=0, upBound=BATTERY_E_MWH)
    net = pulp.LpVariable.dicts("da_optimizer_plan_mw", t_idx, lowBound=-GRID_LIMIT_MW, upBound=GRID_LIMIT_MW)

    u_charge = pulp.LpVariable.dicts("u_charge", t_idx, lowBound=0, upBound=1, cat="Binary")
    u_discharge = pulp.LpVariable.dicts("u_discharge", t_idx, lowBound=0, upBound=1, cat="Binary")
    u_import = pulp.LpVariable.dicts("u_import", t_idx, lowBound=0, upBound=1, cat="Binary")
    u_export = pulp.LpVariable.dicts("u_export", t_idx, lowBound=0, upBound=1, cat="Binary")

    soc_init = SOC_INIT_PCT / 100.0 * BATTERY_E_MWH
    soc_target = SOC_TARGET_PCT / 100.0 * BATTERY_E_MWH

    for t in t_idx:
        pv = float(df.loc[t, "pv_forecast_mw"])
        p_ch_av = float(df.loc[t, "p_charge_max_mw"])
        p_dis_av = float(df.loc[t, "p_discharge_max_mw"])

        model += pv_grid[t] + pv_batt[t] + pv_curt[t] == pv, f"pv_balance_{t}"

        model += pv_batt[t] + grid_charge[t] <= p_ch_av, f"charge_avail_{t}"
        model += discharge[t] <= p_dis_av, f"discharge_avail_{t}"

        if not GRID_CHARGING_ALLOWED:
            model += grid_charge[t] == 0.0, f"no_grid_charge_{t}"

        model += pv_batt[t] + grid_charge[t] <= BATTERY_P_MAX_MW * u_charge[t], f"charge_bin_{t}"
        model += discharge[t] <= BATTERY_P_MAX_MW * u_discharge[t], f"discharge_bin_{t}"
        model += u_charge[t] + u_discharge[t] <= 1, f"no_charge_discharge_{t}"

        export = pv_grid[t] + discharge[t]
        imp = grid_charge[t]
        model += export <= GRID_LIMIT_MW * u_export[t], f"export_bin_{t}"
        model += imp <= GRID_LIMIT_MW * u_import[t], f"import_bin_{t}"
        model += u_export[t] + u_import[t] <= 1, f"no_import_export_{t}"
        model += net[t] == export - imp, f"net_{t}"

        model += soc[t] >= float(df.loc[t, "soc_min_mwh_used"]), f"soc_min_{t}"
        model += soc[t] <= float(df.loc[t, "soc_max_mwh_used"]), f"soc_max_{t}"

        if t == 0:
            model += soc[t] == soc_init + DT_H * (EFF * (pv_batt[t] + grid_charge[t]) - INV_EFF * discharge[t]), f"soc_dyn_{t}"
        else:
            model += soc[t] == soc[t - 1] + DT_H * (EFF * (pv_batt[t] + grid_charge[t]) - INV_EFF * discharge[t]), f"soc_dyn_{t}"

        if BLOCK_DA_EXPORT_BELOW_FLOOR and float(df.loc[t, "da_price_forecast"]) < DA_EXPORT_PRICE_FLOOR_EUR_PER_MWH:
            model += net[t] <= 0.0, f"block_export_below_floor_{t}"

    if SOC_TERMINAL_SYNC and t_idx:
        model += soc[t_idx[-1]] == soc_target, "terminal_soc_target"

    da_revenue = pulp.lpSum(float(df.loc[t, "da_price_forecast"]) * net[t] * DT_H for t in t_idx)
    degradation = pulp.lpSum(
        (CYCLE_COST_EUR_PER_MWH / 2.0) * (pv_batt[t] + grid_charge[t] + discharge[t]) * DT_H
        for t in t_idx
    )

    model += da_revenue - degradation

    status_code = model.solve(make_solver_safe())
    status_name = pulp.LpStatus[model.status]
    if status_name != "Optimal":
        raise RuntimeError(f"DA-only MILP not optimal: {status_name} (code={status_code})")

    objective_value = float(pulp.value(model.objective) or 0.0)

    out = df.copy()
    if NO_TRADE_IF_FORECAST_OBJECTIVE_NONPOSITIVE and objective_value <= EPS:
        out["da_optimizer_plan_mw"] = 0.0
        out["da_plan_soc_mwh"] = SOC_INIT_PCT / 100.0 * BATTERY_E_MWH
    else:
        out["da_optimizer_plan_mw"] = [
            quantize_to_lot(float(pulp.value(net[t]) or 0.0), DA_TRADE_LOT_MW, CURVE_VOLUME_ROUND_MODE)
            for t in t_idx
        ]
        out["da_plan_soc_mwh"] = [float(pulp.value(soc[t]) or 0.0) for t in t_idx]

    out.attrs["solver_status_code"] = int(status_code)
    out.attrs["solver_status"] = status_name
    out.attrs["forecast_objective_eur"] = objective_value
    return out


def _deliver_one_interval(
    *,
    soc_mwh: float,
    pv_actual_mw: float,
    requested_net_mw: float,
    p_charge_avail_mw: float,
    p_discharge_avail_mw: float,
    soc_min_mwh: float,
    soc_max_mwh: float,
) -> dict[str, float]:
    requested_net_mw = float(requested_net_mw)
    pv_used_for_export_mw = 0.0
    battery_discharge_mw = 0.0
    grid_charge_mw = 0.0
    pv_charge_mw = 0.0
    pv_curtail_mw = 0.0

    if requested_net_mw >= 0.0:
        export_target_mw = requested_net_mw
        pv_used_for_export_mw = min(pv_actual_mw, export_target_mw)
        remaining_export_mw = export_target_mw - pv_used_for_export_mw

        max_discharge_mw = min(
            p_discharge_avail_mw,
            max(0.0, (soc_mwh - soc_min_mwh) * EFF / max(DT_H, EPS)),
            GRID_LIMIT_MW,
        )
        battery_discharge_mw = min(remaining_export_mw, max_discharge_mw)
        delivered_net_mw = pv_used_for_export_mw + battery_discharge_mw

        unused_pv_mw = max(0.0, pv_actual_mw - pv_used_for_export_mw)
        if AUTO_CHARGE_EXCESS_PV and unused_pv_mw > EPS:
            max_pv_charge_mw = min(
                p_charge_avail_mw,
                max(0.0, (soc_max_mwh - soc_mwh + battery_discharge_mw * DT_H * INV_EFF) / max(EFF * DT_H, EPS)),
            )
            pv_charge_mw = min(unused_pv_mw, max_pv_charge_mw)
            pv_curtail_mw = max(0.0, unused_pv_mw - pv_charge_mw)
        else:
            pv_curtail_mw = unused_pv_mw

    else:
        import_target_mw = -requested_net_mw
        max_charge_mw = min(
            p_charge_avail_mw,
            max(0.0, (soc_max_mwh - soc_mwh) / max(EFF * DT_H, EPS)),
            GRID_LIMIT_MW,
        )
        grid_charge_mw = min(import_target_mw, max_charge_mw)
        delivered_net_mw = -grid_charge_mw
        pv_curtail_mw = pv_actual_mw

    soc_next_mwh = soc_mwh + DT_H * EFF * (grid_charge_mw + pv_charge_mw) - DT_H * INV_EFF * battery_discharge_mw
    soc_next_mwh = clamp(soc_next_mwh, soc_min_mwh, soc_max_mwh)

    throughput_mwh = DT_H * (grid_charge_mw + pv_charge_mw + battery_discharge_mw)

    return {
        "delivered_net_mw": delivered_net_mw,
        "pv_used_for_export_mw": pv_used_for_export_mw,
        "battery_discharge_mw": battery_discharge_mw,
        "grid_charge_mw": grid_charge_mw,
        "pv_charge_mw": pv_charge_mw,
        "pv_curtail_mw": pv_curtail_mw,
        "battery_throughput_mwh": throughput_mwh,
        "soc_next_mwh": soc_next_mwh,
    }


def _forecast_bid_bounds(
    *,
    soc_mwh: float,
    row: pd.Series,
    target_floor_mwh: float | None = None,
    target_ceiling_mwh: float | None = None,
) -> tuple[float, float]:
    """Return bid limits using forecast information only.

    DA bidding must not look at actual PV. Actuals are only used later for
    delivery, imbalance, and realized cashflow.
    """
    soc_min_mwh = float(row["soc_min_mwh_used"])
    soc_max_mwh = float(row["soc_max_mwh_used"])

    if target_floor_mwh is not None:
        soc_min_mwh = max(soc_min_mwh, float(target_floor_mwh))
    if target_ceiling_mwh is not None:
        soc_max_mwh = min(soc_max_mwh, float(target_ceiling_mwh))
    if soc_min_mwh > soc_max_mwh + EPS:
        soc_min_mwh = float(row["soc_min_mwh_used"])
        soc_max_mwh = float(row["soc_max_mwh_used"])

    q_min, q_max = _physical_net_bounds(
        soc_mwh=soc_mwh,
        pv_mw=float(row["pv_forecast_mw"]),
        p_charge_avail_mw=float(row["p_charge_max_mw"]),
        p_discharge_avail_mw=float(row["p_discharge_max_mw"]),
        soc_min_mwh=soc_min_mwh,
        soc_max_mwh=soc_max_mwh,
    )
    q_min = quantize_to_lot(q_min, DA_TRADE_LOT_MW, "ceil")
    q_max = quantize_to_lot(q_max, DA_TRADE_LOT_MW, "floor")
    if q_min > q_max:
        return 0.0, 0.0
    return q_min, q_max


def _accepted_da_volume(
    *,
    soc_mwh: float,
    row: pd.Series,
    requested_net_mw: float,
    target_floor_mwh: float | None = None,
    target_ceiling_mwh: float | None = None,
) -> tuple[float, float, float, dict[str, float]]:
    """Create a DA bid from forecast and deliver it against actuals.

    Critical rule:
      * forecast data decide the bid quantity.
      * actual price decides whether the bid clears and realized revenue.
      * actual PV/battery state decide delivery and imbalance.

    This prevents the backtest from using actual PV to silently cancel bad bids.
    """
    q_min, q_max = _forecast_bid_bounds(
        soc_mwh=soc_mwh,
        row=row,
        target_floor_mwh=target_floor_mwh,
        target_ceiling_mwh=target_ceiling_mwh,
    )

    requested = quantize_to_lot(requested_net_mw, DA_TRADE_LOT_MW, CURVE_VOLUME_ROUND_MODE)
    bid_mw = clamp(requested, q_min, q_max)

    if BLOCK_DA_EXPORT_BELOW_FLOOR and bid_mw > 0.0 and float(row["da_clearing_price"]) < DA_EXPORT_PRICE_FLOOR_EUR_PER_MWH:
        accepted = 0.0
    else:
        accepted = bid_mw

    soc_min_mwh = float(row["soc_min_mwh_used"])
    soc_max_mwh = float(row["soc_max_mwh_used"])
    if target_floor_mwh is not None:
        soc_min_mwh = max(soc_min_mwh, float(target_floor_mwh))
    if target_ceiling_mwh is not None:
        soc_max_mwh = min(soc_max_mwh, float(target_ceiling_mwh))
    if soc_min_mwh > soc_max_mwh + EPS:
        soc_min_mwh = float(row["soc_min_mwh_used"])
        soc_max_mwh = float(row["soc_max_mwh_used"])

    trial = _deliver_one_interval(
        soc_mwh=soc_mwh,
        pv_actual_mw=float(row["pv_actual_mw"]),
        requested_net_mw=accepted,
        p_charge_avail_mw=float(row["p_charge_max_mw"]),
        p_discharge_avail_mw=float(row["p_discharge_max_mw"]),
        soc_min_mwh=soc_min_mwh,
        soc_max_mwh=soc_max_mwh,
    )

    return accepted, q_min, q_max, trial

def _future_charge_energy_mwh(plan: pd.DataFrame, start_idx: int) -> float:
    total = 0.0
    for k in range(start_idx, len(plan)):
        p = min(float(plan.loc[k, "p_charge_max_mw"]), BATTERY_P_MAX_MW, GRID_LIMIT_MW)
        total += DT_H * EFF * max(0.0, p)
    return total


def _future_discharge_energy_mwh(plan: pd.DataFrame, start_idx: int) -> float:
    total = 0.0
    for k in range(start_idx, len(plan)):
        p = min(float(plan.loc[k, "p_discharge_max_mw"]), BATTERY_P_MAX_MW, GRID_LIMIT_MW)
        total += DT_H * INV_EFF * max(0.0, p)
    return total


def _terminal_reachable_band_after_interval(plan: pd.DataFrame, idx: int, row: pd.Series) -> tuple[float, float]:
    if not SOC_TERMINAL_SYNC:
        return float(row["soc_min_mwh_used"]), float(row["soc_max_mwh_used"])

    target = SOC_TARGET_PCT / 100.0 * BATTERY_E_MWH
    next_idx = int(idx) + 1
    floor = max(float(row["soc_min_mwh_used"]), target - _future_charge_energy_mwh(plan, next_idx))
    ceiling = min(float(row["soc_max_mwh_used"]), target + _future_discharge_energy_mwh(plan, next_idx))

    if floor > ceiling + EPS:
        return float(row["soc_min_mwh_used"]), float(row["soc_max_mwh_used"])
    return floor, ceiling


def simulate_da_delivery(plan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    soc_mwh = SOC_INIT_PCT / 100.0 * BATTERY_E_MWH

    for i, row in plan.iterrows():
        target_floor_mwh, target_ceiling_mwh = _terminal_reachable_band_after_interval(plan, int(i), row)
        accepted, q_min, q_max, delivery = _accepted_da_volume(
            soc_mwh=soc_mwh,
            row=row,
            requested_net_mw=float(row["da_optimizer_plan_mw"]),
            target_floor_mwh=target_floor_mwh,
            target_ceiling_mwh=target_ceiling_mwh,
        )

        delivered = float(delivery["delivered_net_mw"])
        mismatch_mw = accepted - delivered

        da_cash = accepted * float(row["da_clearing_price"]) * DT_H
        degradation = (CYCLE_COST_EUR_PER_MWH / 2.0) * float(delivery["battery_throughput_mwh"])

        imbalance_penalty = 0.0
        if abs(mismatch_mw) > DA_TRADE_LOT_MW / 2:
            imbalance_penalty = abs(mismatch_mw) * abs(float(row["da_clearing_price"])) * DT_H

        net_cash = da_cash - degradation - imbalance_penalty
        if OPTIONAL_PARTICIPATION_NO_LOSS and net_cash < 0.0 and abs(accepted) <= EPS:
            net_cash = 0.0

        soc_mwh = float(delivery["soc_next_mwh"])

        rows.append(
            {
                "interval": int(i) + 1,
                "timestamp": row["timestamp"],
                "da_price_forecast": float(row["da_price_forecast"]),
                "da_clearing_price": float(row["da_clearing_price"]),
                "pv_forecast_mw": float(row["pv_forecast_mw"]),
                "pv_actual_mw": float(row["pv_actual_mw"]),
                "da_optimizer_plan_mw": float(row["da_optimizer_plan_mw"]),
                "da_bid_min_mw": float(q_min),
                "da_bid_max_mw": float(q_max),
                "da_bid_mw": float(accepted),
                "da_accepted_mw": float(accepted),
                "da_delivered_mw": float(delivered),
                "da_undelivered_mw": float(mismatch_mw),
                "battery_grid_charge_mw": float(delivery["grid_charge_mw"]),
                "battery_pv_charge_mw": float(delivery["pv_charge_mw"]),
                "battery_discharge_mw": float(delivery["battery_discharge_mw"]),
                "pv_export_mw": float(delivery["pv_used_for_export_mw"]),
                "pv_curtail_mw": float(delivery["pv_curtail_mw"]),
                "da_cashflow_eur_actual": float(da_cash),
                "mfrr_up_cashflow_eur_actual": 0.0,
                "mfrr_down_cashflow_eur_actual": 0.0,
                "mfrr_energy_cashflow_eur_actual": 0.0,
                "imbalance_penalty_eur_actual": float(imbalance_penalty),
                "battery_degradation_eur": float(degradation),
                "net_cashflow_eur_actual": float(net_cash),
                "market_net_cashflow_eur_actual": float(net_cash),
                "market_net_eur_actual": float(net_cash),
                "soc_target_mwh": float(SOC_TARGET_PCT / 100.0 * BATTERY_E_MWH),
                "soc_target_pct": float(SOC_TARGET_PCT),
                "soc_terminal_floor_mwh": float(target_floor_mwh),
                "soc_terminal_ceiling_mwh": float(target_ceiling_mwh),
                "soc_after_all_mwh": float(soc_mwh),
                "soc_after_all_pct": 100.0 * float(soc_mwh) / BATTERY_E_MWH,
            }
        )

    report = pd.DataFrame(rows)
    return report


def _summary_from_report(report: pd.DataFrame, plan: pd.DataFrame) -> dict[str, Any]:
    total_da = float(report["da_cashflow_eur_actual"].sum()) if len(report) else 0.0
    total_deg = float(report["battery_degradation_eur"].sum()) if len(report) else 0.0
    total_imb = float(report["imbalance_penalty_eur_actual"].sum()) if len(report) else 0.0
    total_net = float(report["net_cashflow_eur_actual"].sum()) if len(report) else 0.0
    return {
        "status": "Optimal",
        "engine": "DA_ONLY_SAFE_DAILY_SOC50",
        "intervals": int(len(report)),
        "forecast_objective_eur": float(plan.attrs.get("forecast_objective_eur", 0.0)),
        "solver_status": str(plan.attrs.get("solver_status", "")),
        "solver_status_code": int(plan.attrs.get("solver_status_code", 0)),
        "soc_init_pct": float(SOC_INIT_PCT),
        "soc_final_pct": float(report["soc_after_all_pct"].iloc[-1]) if len(report) else float(SOC_INIT_PCT),
        "da_cashflow_eur_actual": total_da,
        "mfrr_energy_cashflow_eur_actual": 0.0,
        "imbalance_penalty_eur_actual": total_imb,
        "battery_degradation_eur": total_deg,
        "market_net_eur_actual": total_net,
        "market_net_cashflow_eur_actual": total_net,
        "net_cashflow_eur_actual": total_net,
    }


def print_interval_report(report: pd.DataFrame) -> None:
    print("\nInterval-level DA-only report:")
    cols = [
        "timestamp",
        "da_optimizer_plan_mw",
        "da_bid_min_mw",
        "da_bid_max_mw",
        "da_accepted_mw",
        "da_delivered_mw",
        "soc_after_all_pct",
        "pv_curtail_mw",
        "da_cashflow_eur_actual",
        "battery_degradation_eur",
        "imbalance_penalty_eur_actual",
        "net_cashflow_eur_actual",
    ]
    printable = report[cols].copy()
    print(printable.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_summary(summary: dict[str, Any]) -> None:
    print("\n========== DA-ONLY SUMMARY ==========")
    print(f"Intervals: {summary['intervals']}")
    print(f"DA cashflow actual: {summary['da_cashflow_eur_actual']:.2f} EUR")
    print(f"Imbalance penalty actual: {summary['imbalance_penalty_eur_actual']:.2f} EUR")
    print(f"Battery degradation: {summary['battery_degradation_eur']:.2f} EUR")
    print(f"Market net actual: {summary['market_net_eur_actual']:.2f} EUR")
    print(f"Final SoC: {summary['soc_final_pct']:.4f}%")


def run(cfg: Any | None = None) -> pd.DataFrame:
    """Run the DA-only engine and return the interval report.

    This is the interface expected by the scenario runner:
    `MarketConfig()` is created in the child process, then `run(cfg)` is called,
    and the returned value must be a DataFrame with a final SoC column.
    """
    apply_config(cfg)

    df = load_data()
    plan = solve_da_plan(df)
    report = simulate_da_delivery(plan)

    output_report = _as_path(OUTPUT_REPORT)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_report, index=False, encoding="utf-8-sig")

    summary = _summary_from_report(report, plan)
    report.attrs["summary"] = summary

    output_summary = globals().get("OUTPUT_SUMMARY", None)
    if output_summary is not None:
        output_summary = _as_path(output_summary)
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(output_summary, index=False, encoding="utf-8-sig")
        with open(output_summary.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if bool(_cfg_value(cfg, "print_interval_table", True)):
        print_interval_report(report)
    print_summary(summary)
    print(f"\nDetailed CSV saved to: {output_report}")

    return report


def main() -> None:
    run(None)


if __name__ == "__main__":
    main()
