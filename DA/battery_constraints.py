from __future__ import annotations

import pandas as pd
import pulp

from market_config import (
    BATTERY_E_MWH,
    BATTERY_P_MAX_MW,
    DT_H,
    EFF,
    GRID_CHARGING_ALLOWED,
    GRID_LIMIT_MW,
    INV_EFF,
    PHYSICAL_SOC_MAX_PCT,
    PHYSICAL_SOC_MIN_PCT,
    RESERVE_ENERGY_DURATION_H,
    SOC_INIT_PCT,
    SOC_TERMINAL_SYNC,
)


def recompute_soc_limits_from_existing_reserves(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute SoC min/max using current config and remaining power from reservation file."""
    out = df.copy()

    sold_up_mw = (BATTERY_P_MAX_MW - out["p_discharge_avail_mw"]).clip(lower=0.0)
    sold_down_mw = (BATTERY_P_MAX_MW - out["p_charge_avail_mw"]).clip(lower=0.0)

    physical_min_mwh = PHYSICAL_SOC_MIN_PCT / 100.0 * BATTERY_E_MWH
    physical_max_mwh = PHYSICAL_SOC_MAX_PCT / 100.0 * BATTERY_E_MWH

    up_energy_buffer_mwh = sold_up_mw * RESERVE_ENERGY_DURATION_H * INV_EFF
    down_energy_room_mwh = sold_down_mw * RESERVE_ENERGY_DURATION_H * EFF

    out["sold_up_reserve_mw_reconstructed"] = sold_up_mw
    out["sold_down_reserve_mw_reconstructed"] = sold_down_mw
    out["soc_min_mwh_used"] = (physical_min_mwh + up_energy_buffer_mwh).clip(upper=physical_max_mwh)
    out["soc_max_mwh_used"] = (physical_max_mwh - down_energy_room_mwh).clip(lower=physical_min_mwh)

    bad = out["soc_min_mwh_used"] > out["soc_max_mwh_used"]
    if bad.any():
        raise ValueError("Infeasible reserve SoC band in some intervals.")

    out["soc_min_pct_used"] = 100.0 * out["soc_min_mwh_used"] / BATTERY_E_MWH
    out["soc_max_pct_used"] = 100.0 * out["soc_max_mwh_used"] / BATTERY_E_MWH
    return out


def add_da_milp_constraints(
    model: pulp.LpProblem,
    df: pd.DataFrame,
    pv_grid: dict,
    pv_batt: dict,
    pv_curt: dict,
    grid_charge: dict,
    discharge: dict,
    soc: dict,
    u_charge: dict,
    u_discharge: dict,
    u_import: dict,
    u_export: dict,
    net: dict,
):
    """Add DA-stage physical battery, PV, grid, reserve, and SoC constraints."""
    T = list(range(len(df)))
    soc_init = SOC_INIT_PCT / 100.0 * BATTERY_E_MWH

    for t in T:
        p_ch_av = float(df.loc[t, "p_charge_avail_mw"])
        p_dis_av = float(df.loc[t, "p_discharge_avail_mw"])
        pv = float(df.loc[t, "pv_bid_basis_mw"])

        # PV balance.
        model += pv_grid[t] + pv_batt[t] + pv_curt[t] == pv, f"pv_balance_{t}"

        # Remaining power after already-sold FCR/aFRR/mFRR reserves.
        model += pv_batt[t] + grid_charge[t] <= p_ch_av, f"charge_avail_{t}"
        model += discharge[t] <= p_dis_av, f"discharge_avail_{t}"

        if not GRID_CHARGING_ALLOWED:
            model += grid_charge[t] == 0, f"no_grid_charge_{t}"

        # No simultaneous charge/discharge.
        model += pv_batt[t] + grid_charge[t] <= BATTERY_P_MAX_MW * u_charge[t], f"charge_bin_{t}"
        model += discharge[t] <= BATTERY_P_MAX_MW * u_discharge[t], f"discharge_bin_{t}"
        model += u_charge[t] + u_discharge[t] <= 1, f"no_charge_discharge_{t}"

        # Grid import/export consistency.
        export = pv_grid[t] + discharge[t]
        imp = grid_charge[t]
        model += export <= GRID_LIMIT_MW * u_export[t], f"export_bin_{t}"
        model += imp <= GRID_LIMIT_MW * u_import[t], f"import_bin_{t}"
        model += u_export[t] + u_import[t] <= 1, f"no_import_export_{t}"
        model += net[t] == export - imp, f"net_{t}"

        # SoC constraints including reserve energy buffer.
        model += soc[t] >= float(df.loc[t, "soc_min_mwh_used"]), f"soc_min_{t}"
        model += soc[t] <= float(df.loc[t, "soc_max_mwh_used"]), f"soc_max_{t}"

        if t == 0:
            model += soc[t] == soc_init + DT_H * (EFF * (pv_batt[t] + grid_charge[t]) - INV_EFF * discharge[t]), f"soc_dyn_{t}"
        else:
            model += soc[t] == soc[t - 1] + DT_H * (EFF * (pv_batt[t] + grid_charge[t]) - INV_EFF * discharge[t]), f"soc_dyn_{t}"

    if SOC_TERMINAL_SYNC:
        model += soc[T[-1]] == soc_init, "terminal_soc"
