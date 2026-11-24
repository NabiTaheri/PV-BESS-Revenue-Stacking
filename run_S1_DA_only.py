from __future__ import annotations

"""
Monthly scenario runner with chained SoC.

    Day D:
        Optional FCR capacity stage.
        Optional aFRR/mFRR capacity stage.
        Optional DA / mFRR-energy dispatch stage.
        End-of-day SoC from the DA stage is chained into day D+1.

    Scenario switches:
        DA                  -> final day-ahead dispatch stage.
        mFRR energy          -> enabled inside the DA stage.
        FCR                 -> symmetric FCR capacity stage before capacity stacking.
        aFRR/mFRR capacity   -> capacity stacking stage after FCR.

    Built-in market-participation scenarios:
        S1: DA only
        S2: DA + mFRR energy
        S3: DA + mFRR energy + aFRR/mFRR capacity
        S4: DA + mFRR energy + aFRR/mFRR capacity + FCR

Expected folder layout by default:

    i-em/
      data/
        forecasts_rolling.csv
        actuals.csv
      DA/
      FCR/
      MFRR&AFRR/

Each stage folder should contain its own importable project files, for example:

    FCR/
      config.py
      FCR.py
      da_mfrr.py
      afrr_mfrr_capacity_value.py

    MFRR&AFRR/
      market_config.py
      config_capacity.py
      afrr_mfrr.py
      da_mfrr.py

    DA/
      either:
        da.py + market_config.py              [DA-only engine]
        da_mfrr_energy.py + market_config.py   [DA + mFRR/balancing engine]
      or:
        da_mfrr.py + market_config.py          [forecast-only fallback]

This runner deliberately runs each stage in a separate Python subprocess to avoid
module-cache/config contamination between folders and days.

The monthly source data are read from i-em/data by default. The source files in
data/ are never overwritten. Daily slices are copied into the stage folders
(FCR, MFRR&AFRR, DA), where the existing project code can safely overwrite its
own working input files.
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# USER SCENARIO TABLE - EDIT ONLY THIS BLOCK FOR NORMAL USE
# =============================================================================
# Put this file in the main i-em folder and run it with no command-line arguments.
# The script will run every row whose RUN value is True.
#
# Market switches:
#   DA                  = final DA dispatch stage
#   MFRR_ENERGY         = mFRR balancing-energy inside DA
#   FCR                 = FCR capacity stage
#   AFRR_MFRR_CAPACITY  = aFRR/mFRR capacity stage after FCR/free reservation
#
# Important: MFRR_ENERGY=True requires DA=True.
RUN_SELECTED_SCENARIOS_FROM_TABLE = True

DEFAULT_ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = DEFAULT_ROOT_DIR / "Data"
DEFAULT_INITIAL_SOC_PCT = 50.0
DEFAULT_FINAL_DA_TERMINAL_SYNC = False
DEFAULT_SOC_TARGET_PCT = 50.0
# Current project uses only 3 scenarios:
#   S1 = DA only
#   S2 = DA + mFRR energy
#   S3 = DA + mFRR energy + reserve competition
# Separate root/FCR stage is not used, so daily inputs must not be written there.
RUN_SEPARATE_FCR_STAGE = False


SCENARIO_RUN_TABLE: dict[str, dict[str, Any]] = {
    # S1_DA_only
    # DA only.
    #
    # This file runs ONLY this scenario. To execute it, run:
    #     python run_S1_DA_only.py
    #
    # Normal edit points:
    #     INITIAL_SOC_PCT          -> first-day battery SoC, default 50%
    #     FINAL_DA_TERMINAL_SYNC   -> keep False if end-of-day SoC must chain to next day
    #     OUTPUT_DIR               -> scenario output folder
    "S1_DA_only": {
        "RUN": True,
        "DA": True,
        "MFRR_ENERGY": False,
        "FCR": False,
        "AFRR_MFRR_CAPACITY": False,
        "INITIAL_SOC_PCT": 50.0,
        "FINAL_DA_TERMINAL_SYNC": False,
        "SOC_TARGET_PCT": 50.0,
        "DA_ENERGY_MODULE": "da",
        "OUTPUT_DIR": "results_S1_DA_only",
        "CONTINUE_ON_ERROR": False,
        "SHOW_STAGE_CONSOLE": False,
    },
}


# =============================================================================
# Child process code templates
# =============================================================================

_COMMON_CHILD = r'''
from __future__ import annotations

import importlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


def _to_jsonable(x: Any) -> Any:
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
    except Exception:
        pass
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items() if not hasattr(v, "to_csv")}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return x


def _bool_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _patch_config_like_module(m: Any, soc_init_pct: float, *,
                              forecast_name: str = "forecasts_rolling.csv",
                              actuals_name: str = "actuals.csv",
                              horizon: int | None = None,
                              terminal_sync: bool | None = None,
                              enable_da: bool | None = None,
                              enable_fcr: bool | None = None,
                              enable_mfrr_energy: bool | None = None,
                              enable_afrr_mfrr_capacity: bool | None = None) -> None:
    """Patch module-level config objects before importing/running a stage."""
    soc = float(soc_init_pct)

    # File names are kept relative to the stage working directory.
    for name, value in [
        ("FORECAST_FILE", Path(forecast_name)),
        ("FORECASTS_FILE", Path(forecast_name)),
        ("ACTUALS_FILE", Path(actuals_name)),
        ("ACTUAL_FILE", Path(actuals_name)),
    ]:
        if hasattr(m, name):
            try:
                setattr(m, name, value)
            except Exception:
                pass

    if horizon is not None and hasattr(m, "HORIZON"):
        try:
            setattr(m, "HORIZON", int(horizon))
        except Exception:
            pass

    if hasattr(m, "SOC_INIT_PCT"):
        try:
            setattr(m, "SOC_INIT_PCT", soc)
        except Exception:
            pass

    if terminal_sync is not None and hasattr(m, "SOC_TERMINAL_SYNC"):
        try:
            setattr(m, "SOC_TERMINAL_SYNC", bool(terminal_sync))
        except Exception:
            pass

    # Market participation switches. These are patched before stage modules
    # import their config, so modules using `from market_config import *` or
    # `from config import *` receive the intended scenario state.
    if enable_mfrr_energy is not None:
        for name in ["ENABLE_MFRR_ENERGY_MARKET", "ENABLE_MFRR_MARKET"]:
            if hasattr(m, name):
                try:
                    setattr(m, name, bool(enable_mfrr_energy))
                except Exception:
                    pass

    markets = getattr(m, "MARKETS", None)
    if isinstance(markets, dict):
        if enable_da is not None:
            markets["DA"] = bool(enable_da)
        if enable_fcr is not None:
            markets["FCR"] = bool(enable_fcr)
        if enable_mfrr_energy is not None:
            markets["mFRR_Energy"] = bool(enable_mfrr_energy)
        if enable_afrr_mfrr_capacity is not None:
            markets["aFRR_Capacity"] = bool(enable_afrr_mfrr_capacity)
            markets["mFRR_Capacity"] = bool(enable_afrr_mfrr_capacity)

    # Dictionary-style BATTERY used by capacity modules.
    b = getattr(m, "BATTERY", None)
    if isinstance(b, dict):
        b["soc_init_pct"] = soc

    # Recalculate headroom/risk numbers when the starting SoC changes.
    try:
        e_mwh = float(getattr(m, "BATTERY_E_MWH", b.get("e_max_mwh")))
        p_max = float(getattr(m, "BATTERY_P_MAX_MW", b.get("p_max_mw")))
        eff = float(getattr(m, "EFF", b.get("efficiency", 1.0)))
        dt = float(getattr(m, "DT", getattr(m, "DT_H", 0.25)))
        soc_min = float(getattr(m, "PHYSICAL_SOC_MIN_PCT", b.get("soc_min_pct", 20.0)))
        soc_max = float(getattr(m, "PHYSICAL_SOC_MAX_PCT", b.get("soc_max_pct", 80.0)))
        reserve_h = float(getattr(m, "RESERVE_ENERGY_DURATION_H", 0.25))

        up_headroom_mwh = max(0.0, (soc - soc_min) / 100.0 * e_mwh)
        down_headroom_mwh = max(0.0, (soc_max - soc) / 100.0 * e_mwh)
        symmetric_headroom_mwh = min(up_headroom_mwh, down_headroom_mwh)

        for name, value in [
            ("UP_SOC_HEADROOM_MWH", up_headroom_mwh),
            ("DOWN_SOC_HEADROOM_MWH", down_headroom_mwh),
            ("FCR_MAX_CUMULATIVE_EXPOSURE_MWH", symmetric_headroom_mwh),
            ("AFRR_MFRR_UP_MAX_CUMULATIVE_EXPOSURE_MWH", up_headroom_mwh),
            ("AFRR_MFRR_DOWN_MAX_CUMULATIVE_EXPOSURE_MWH", down_headroom_mwh),
        ]:
            if hasattr(m, name):
                try:
                    setattr(m, name, value)
                except Exception:
                    pass

        risk = getattr(m, "RISK_MANAGEMENT", None)
        if isinstance(risk, dict):
            # FCR symmetric risk.
            risk["FCR_MAX_CUMULATIVE_EXPOSURE_MWH"] = symmetric_headroom_mwh
            risk["MAX_CUMULATIVE_EXPOSURE_MWH"] = symmetric_headroom_mwh
            risk["FCR_EXPOSURE_PER_MW_INTERVAL_MWH"] = reserve_h / max(eff, 1e-9)
            risk["FCR_RECOVERY_PER_INTERVAL_MWH"] = p_max * dt * eff

            # Directional aFRR/mFRR risk.
            risk["AFRR_MFRR_UP_MAX_CUMULATIVE_EXPOSURE_MWH"] = up_headroom_mwh
            risk["AFRR_MFRR_DOWN_MAX_CUMULATIVE_EXPOSURE_MWH"] = down_headroom_mwh
            risk["AFRR_MFRR_MAX_CUMULATIVE_EXPOSURE_MWH"] = symmetric_headroom_mwh
            risk["AFRR_MFRR_RECOVERY_PER_INTERVAL_MWH"] = p_max * dt * eff
    except Exception:
        # Some config modules are dataclass-only or incomplete; ignore safely.
        pass


def _write_summary(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2, ensure_ascii=False)
'''


FCR_CHILD_CODE = _COMMON_CHILD + r'''

def main(payload_path: str) -> None:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    os.chdir(payload["stage_dir"])
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    soc = float(payload["soc_init_pct"])
    horizon = int(payload["horizon"])

    cfg = importlib.import_module(payload.get("config_module", "config"))
    _patch_config_like_module(
        cfg, soc, horizon=horizon, terminal_sync=_bool_or_none(payload.get("terminal_sync")),
        enable_da=_bool_or_none(payload.get("enable_da")),
        enable_fcr=_bool_or_none(payload.get("enable_fcr")),
        enable_mfrr_energy=_bool_or_none(payload.get("enable_mfrr_energy")),
        enable_afrr_mfrr_capacity=_bool_or_none(payload.get("enable_afrr_mfrr_capacity")),
    )

    stage = importlib.import_module(payload.get("module_name", "FCR"))
    # The FCR module keeps a direct reference to cfg_module and some module-level paths.
    if hasattr(stage, "cfg_module"):
        _patch_config_like_module(
            stage.cfg_module, soc, horizon=horizon, terminal_sync=_bool_or_none(payload.get("terminal_sync")),
            enable_da=_bool_or_none(payload.get("enable_da")),
            enable_fcr=_bool_or_none(payload.get("enable_fcr")),
            enable_mfrr_energy=_bool_or_none(payload.get("enable_mfrr_energy")),
            enable_afrr_mfrr_capacity=_bool_or_none(payload.get("enable_afrr_mfrr_capacity")),
        )

    if hasattr(stage, "FCR_SUBMITTED_WORST_CASE_RESERVATIONS_FILE"):
        stage.FCR_SUBMITTED_WORST_CASE_RESERVATIONS_FILE = str(payload["submitted_worst_case_file"])

    result = stage.run_fcr_case(
        print_console=bool(payload.get("print_console", False)),
        save_csv=True,
        results_file=str(payload["results_file"]),
        fcr_reservation_output_file=str(payload["reservation_file"]),
    )

    summary = {
        "status": result.get("status"),
        "soc_init_pct": soc,
        "total_fcr_revenue_eur": result.get("total_fcr_revenue_eur"),
        "fcr_forecast_scheduling_objective_eur": result.get("fcr_forecast_scheduling_objective_eur"),
        "base_da_mfrr_forecast_value_eur": result.get("base_da_mfrr_forecast_value_eur"),
        "base_afrr_mfrr_capacity_value_eur": result.get("base_afrr_mfrr_capacity_value_eur"),
        "total_uncovered_charge_recovery_mwh": result.get("total_uncovered_charge_recovery_mwh"),
        "total_uncovered_discharge_recovery_mwh": result.get("total_uncovered_discharge_recovery_mwh"),
    }
    _write_summary(payload["summary_json"], summary)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception:
        traceback.print_exc()
        raise
'''


CAPACITY_CHILD_CODE = _COMMON_CHILD + r'''

def main(payload_path: str) -> None:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    os.chdir(payload["stage_dir"])
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    soc = float(payload["soc_init_pct"])
    horizon = int(payload["horizon"])

    # config_capacity imports from market_config, so patch market_config first when present.
    try:
        mc = importlib.import_module(payload.get("market_config_module", "market_config"))
        _patch_config_like_module(
            mc, soc, horizon=horizon, terminal_sync=_bool_or_none(payload.get("terminal_sync")),
            enable_da=_bool_or_none(payload.get("enable_da")),
            enable_fcr=_bool_or_none(payload.get("enable_fcr")),
            enable_mfrr_energy=_bool_or_none(payload.get("enable_mfrr_energy")),
            enable_afrr_mfrr_capacity=_bool_or_none(payload.get("enable_afrr_mfrr_capacity")),
        )
    except Exception:
        mc = None

    cc = importlib.import_module(payload.get("config_module", "config_capacity"))
    _patch_config_like_module(
        cc, soc, horizon=horizon, terminal_sync=_bool_or_none(payload.get("terminal_sync")),
        enable_da=_bool_or_none(payload.get("enable_da")),
        enable_fcr=_bool_or_none(payload.get("enable_fcr")),
        enable_mfrr_energy=_bool_or_none(payload.get("enable_mfrr_energy")),
        enable_afrr_mfrr_capacity=_bool_or_none(payload.get("enable_afrr_mfrr_capacity")),
    )

    stage = importlib.import_module(payload.get("module_name", "afrr_mfrr"))

    if hasattr(stage, "cfg_module"):
        _patch_config_like_module(
            stage.cfg_module, soc, horizon=horizon, terminal_sync=_bool_or_none(payload.get("terminal_sync")),
            enable_da=_bool_or_none(payload.get("enable_da")),
            enable_fcr=_bool_or_none(payload.get("enable_fcr")),
            enable_mfrr_energy=_bool_or_none(payload.get("enable_mfrr_energy")),
            enable_afrr_mfrr_capacity=_bool_or_none(payload.get("enable_afrr_mfrr_capacity")),
        )
    if hasattr(stage, "da_module"):
        _patch_config_like_module(
            stage.da_module, soc, horizon=horizon, terminal_sync=_bool_or_none(payload.get("terminal_sync")),
            enable_da=_bool_or_none(payload.get("enable_da")),
            enable_fcr=_bool_or_none(payload.get("enable_fcr")),
            enable_mfrr_energy=_bool_or_none(payload.get("enable_mfrr_energy")),
            enable_afrr_mfrr_capacity=_bool_or_none(payload.get("enable_afrr_mfrr_capacity")),
        )

    # Force the stage to read the FCR reservation copied into this folder.
    if hasattr(stage, "FCR_RESERVATIONS_FILE"):
        stage.FCR_RESERVATIONS_FILE = Path("battery_reservations_fcr_symmetric.csv")

    result = stage.run_afrr_mfrr_capacity_case(
        print_console=bool(payload.get("print_console", False)),
        save_csv=True,
        results_file=str(payload["results_file"]),
        reservation_output_file=str(payload["reservation_file"]),
        submitted_worst_case_reservation_file=str(payload["submitted_worst_case_file"]),
        use_actuals_for_backtest_clearing=bool(payload.get("use_actuals_for_backtest_clearing", True)),
    )

    summary = {
        "status": result.get("status"),
        "soc_init_pct": soc,
        "base_da_mfrr_forecast_value_eur": result.get("base_da_mfrr_forecast_value_eur"),
        "dp_forecast_scheduling_objective_eur": result.get("dp_forecast_scheduling_objective_eur"),
        "total_afrr_mfrr_capacity_revenue_eur": result.get("total_afrr_mfrr_capacity_revenue_eur"),
        "total_uncovered_charge_recovery_mwh": result.get("total_uncovered_charge_recovery_mwh"),
        "total_uncovered_discharge_recovery_mwh": result.get("total_uncovered_discharge_recovery_mwh"),
        "submitted_worst_case_repair_removed_steps": result.get("submitted_worst_case_repair_removed_steps"),
    }
    _write_summary(payload["summary_json"], summary)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception:
        traceback.print_exc()
        raise
'''


DA_CHILD_CODE = _COMMON_CHILD + r'''
from dataclasses import fields, is_dataclass, replace


def _final_soc_pct_from_report(report: pd.DataFrame, battery_e_mwh: float | None = None) -> float:
    candidates_pct = [
        "soc_after_all_pct",
        "soc_pct",
        "soc_after_pct",
        "da_plan_soc_pct",
        "soc_end_pct",
    ]
    for col in candidates_pct:
        if col in report.columns and len(report):
            return float(pd.to_numeric(report[col], errors="coerce").dropna().iloc[-1])

    candidates_mwh = [
        "soc_after_all_mwh",
        "soc_mwh",
        "soc_after_mwh",
        "da_plan_soc_mwh",
        "soc_end_mwh",
    ]
    for col in candidates_mwh:
        if col in report.columns and len(report):
            e = float(battery_e_mwh or 1.0)
            return 100.0 * float(pd.to_numeric(report[col], errors="coerce").dropna().iloc[-1]) / e

    raise ValueError(
        "Could not find final SoC column in DA report. Expected one of: "
        + ", ".join(candidates_pct + candidates_mwh)
    )


def _sum_col(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())


def _sum_first_existing_col(df: pd.DataFrame, cols: list[str]) -> float | None:
    for col in cols:
        value = _sum_col(df, col)
        if value is not None:
            return value
    return None


def _try_run_da_mfrr_energy(payload: dict[str, Any], soc: float, horizon: int) -> dict[str, Any] | None:
    """Preferred path for the DA folder: dataclass MarketConfig + da_mfrr_energy.run."""
    try:
        mc = importlib.import_module(payload.get("market_config_module", "market_config"))
        if not hasattr(mc, "MarketConfig"):
            return None
        da_mod = importlib.import_module(payload.get("da_energy_module", "da_mfrr_energy"))
        if not hasattr(da_mod, "run"):
            return None

        MarketConfig = mc.MarketConfig
        base_cfg = MarketConfig()
        supported = {f.name for f in fields(base_cfg)} if is_dataclass(base_cfg) else set()

        kwargs = {
            "forecast_file": Path("forecasts_rolling.csv"),
            "actuals_file": Path("actuals.csv"),
            "reservation_file": Path("battery_reservations_after_fcr_afrr_mfrr_capacity.csv"),
            "output_report": Path(payload["report_file"]),
            "output_summary": Path(payload.get("summary_csv", str(Path(payload["report_file"]).with_name(Path(payload["report_file"]).stem + "_summary.csv")))),
            "soc_init_pct": soc,
            "soc_target_pct": float(payload.get("soc_target_pct", 50.0)),
            "rolling_soc_reference_pct": float(payload.get("soc_target_pct", 50.0)),
            "rolling_recovery_target_soc_pct": float(payload.get("soc_target_pct", 50.0)),
            "reservation_file_required": True,
            "allow_reservation_row_order_fallback": False,
            "print_interval_table": bool(payload.get("print_console", False)),
        }
        mfrr_energy_enabled = _bool_or_none(payload.get("enable_mfrr_energy"))
        if mfrr_energy_enabled is not None:
            kwargs["enable_mfrr_energy_market"] = bool(mfrr_energy_enabled)
            kwargs["include_hypothetical_mfrr_in_da_stage"] = bool(mfrr_energy_enabled)
        ts = _bool_or_none(payload.get("final_da_terminal_sync"))
        if ts is not None:
            kwargs["soc_terminal_sync"] = bool(ts)
            kwargs["rolling_terminal_soc_sync"] = bool(ts)
        kwargs = {k: v for k, v in kwargs.items() if k in supported}
        cfg = replace(base_cfg, **kwargs)

        report = da_mod.run(cfg)
        if report is None:
            report_path = Path(payload["report_file"])
            report = pd.read_csv(report_path)

        battery_e = float(getattr(cfg, "battery_e_mwh", getattr(mc, "BATTERY_E_MWH", 10.0)))
        end_soc_pct = _final_soc_pct_from_report(report, battery_e)
        return {"report": report, "end_soc_pct": end_soc_pct, "engine": "da_mfrr_energy"}
    except ModuleNotFoundError:
        return None


def _run_forecast_da_fallback(payload: dict[str, Any], soc: float, horizon: int) -> dict[str, Any]:
    """Fallback path: da_mfrr.run_da_mfrr_case(reservation_override_df=...)."""
    try:
        mc = importlib.import_module(payload.get("market_config_module", "market_config"))
        _patch_config_like_module(
            mc, soc, horizon=horizon, terminal_sync=_bool_or_none(payload.get("final_da_terminal_sync")),
            enable_da=_bool_or_none(payload.get("enable_da")),
            enable_fcr=_bool_or_none(payload.get("enable_fcr")),
            enable_mfrr_energy=_bool_or_none(payload.get("enable_mfrr_energy")),
            enable_afrr_mfrr_capacity=_bool_or_none(payload.get("enable_afrr_mfrr_capacity")),
        )
    except Exception:
        mc = None

    da_mod = importlib.import_module(payload.get("da_module", "da_mfrr"))
    _patch_config_like_module(
        da_mod, soc, horizon=horizon, terminal_sync=_bool_or_none(payload.get("final_da_terminal_sync")),
        enable_da=_bool_or_none(payload.get("enable_da")),
        enable_fcr=_bool_or_none(payload.get("enable_fcr")),
        enable_mfrr_energy=_bool_or_none(payload.get("enable_mfrr_energy")),
        enable_afrr_mfrr_capacity=_bool_or_none(payload.get("enable_afrr_mfrr_capacity")),
    )

    if hasattr(da_mod, "OUTPUT_REPORT"):
        da_mod.OUTPUT_REPORT = Path(payload["report_file"])
    if hasattr(da_mod, "RESERVATION_FILE"):
        da_mod.RESERVATION_FILE = Path("battery_reservations_after_fcr_afrr_mfrr_capacity.csv")

    reservation = pd.read_csv("battery_reservations_after_fcr_afrr_mfrr_capacity.csv")
    result = da_mod.run_da_mfrr_case(
        reservation_override_df=reservation,
        save_csv=True,
        print_console=bool(payload.get("print_console", False)),
        print_solver=bool(payload.get("print_solver", False)),
    )
    report = result["report"]
    battery_e = float(getattr(da_mod, "BATTERY_E_MWH", 10.0))
    end_soc_pct = _final_soc_pct_from_report(report, battery_e)
    return {"report": report, "end_soc_pct": end_soc_pct, "engine": "da_mfrr_forecast_fallback", "result": result}


def main(payload_path: str) -> None:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    os.chdir(payload["stage_dir"])
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    soc = float(payload["soc_init_pct"])
    horizon = int(payload["horizon"])

    run_result = _try_run_da_mfrr_energy(payload, soc, horizon)
    if run_result is None:
        run_result = _run_forecast_da_fallback(payload, soc, horizon)

    report = run_result["report"]
    report_path = Path(payload["report_file"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not report_path.exists():
        report.to_csv(report_path, index=False, encoding="utf-8-sig")

    end_soc_pct = float(run_result["end_soc_pct"])
    summary = {
        "status": "Optimal",
        "engine": run_result["engine"],
        "soc_init_pct": soc,
        "soc_end_pct": end_soc_pct,
        # Actual/rolling DA engine columns.
        "da_cashflow_eur_actual": _sum_col(report, "da_cashflow_eur_actual"),
        "mfrr_energy_cashflow_eur_actual": _sum_col(report, "mfrr_energy_cashflow_eur_actual"),
        "imbalance_penalty_eur_actual": _sum_col(report, "imbalance_penalty_eur_actual"),
        "battery_degradation_eur": _sum_col(report, "battery_degradation_eur"),
        "net_cashflow_eur_actual": _sum_col(report, "net_cashflow_eur_actual"),
        "market_net_cashflow_eur_actual": _sum_first_existing_col(report, ["market_net_cashflow_eur_actual", "net_cashflow_eur_actual"]),
        # Forecast-only fallback columns.
        "da_cashflow_eur_forecast": _sum_col(report, "da_cashflow_eur_forecast"),
        "mfrr_up_cashflow_eur_forecast": _sum_col(report, "mfrr_up_cashflow_eur_forecast"),
        "mfrr_down_cashflow_eur_forecast": _sum_col(report, "mfrr_down_cashflow_eur_forecast"),
        "mfrr_energy_cashflow_eur_forecast": _sum_col(report, "mfrr_energy_cashflow_eur_forecast"),
        "imbalance_penalty_eur_forecast": _sum_col(report, "imbalance_penalty_eur_forecast"),
        "battery_degradation_eur_forecast": _sum_col(report, "battery_degradation_eur_forecast"),
        "net_cashflow_eur_forecast": _sum_col(report, "net_cashflow_eur_forecast"),
        "rows": int(len(report)),
    }
    _write_summary(payload["summary_json"], summary)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception:
        traceback.print_exc()
        raise
'''


# =============================================================================
# Parent runner helpers
# =============================================================================


def parse_bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"none", "default", ""}:
        return None
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false/none.")


SCENARIOS: dict[str, dict[str, bool]] = {
    # Final market-participation scenarios.
    "S1": {"enable_da": True, "enable_mfrr_energy": False, "enable_fcr": False, "enable_afrr_mfrr_capacity": False},
    "S2": {"enable_da": True, "enable_mfrr_energy": True, "enable_fcr": False, "enable_afrr_mfrr_capacity": False},
    "S3": {"enable_da": True, "enable_mfrr_energy": True, "enable_fcr": False, "enable_afrr_mfrr_capacity": True},
    "S4": {"enable_da": True, "enable_mfrr_energy": True, "enable_fcr": True, "enable_afrr_mfrr_capacity": True},
}


def scenario_switches(args: argparse.Namespace) -> dict[str, bool]:
    scenario = str(args.scenario).strip().upper()
    if scenario == "CUSTOM":
        # Default custom = DA-only unless explicitly switched on. This prevents
        # accidentally running the full stack when the user intended a narrow test.
        switches = {
            "enable_da": True,
            "enable_mfrr_energy": False,
            "enable_fcr": False,
            "enable_afrr_mfrr_capacity": False,
        }
    else:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {args.scenario!r}. Use S1..S4 or custom.")
        switches = dict(SCENARIOS[scenario])

    overrides = {
        "enable_da": args.enable_da,
        "enable_mfrr_energy": args.enable_mfrr_energy,
        "enable_fcr": args.enable_fcr,
        "enable_afrr_mfrr_capacity": args.enable_afrr_mfrr_capacity,
    }
    for key, value in overrides.items():
        if value is not None:
            switches[key] = bool(value)

    if switches["enable_mfrr_energy"] and not switches["enable_da"]:
        raise ValueError("mFRR energy is implemented inside the DA stage, so --enable-mfrr-energy true requires --enable-da true.")

    return switches


def scenario_label(args: argparse.Namespace, switches: dict[str, bool]) -> str:
    s = str(args.scenario).strip().upper()
    if s in SCENARIOS and switches == SCENARIOS[s]:
        return s
    parts = []
    if switches["enable_da"]:
        parts.append("DA")
    if switches["enable_mfrr_energy"]:
        parts.append("mFRR_E")
    if switches["enable_fcr"]:
        parts.append("FCR")
    if switches["enable_afrr_mfrr_capacity"]:
        parts.append("AFRR_MFRR_CAP")
    return "CUSTOM_" + ("_".join(parts) if parts else "NO_MARKET")


def read_with_timestamp(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{label} file must contain a timestamp column: {path}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].duplicated().any():
        dupes = df.loc[df["timestamp"].duplicated(), "timestamp"].head(10).tolist()
        raise ValueError(f"{label} file has duplicate timestamps, examples: {dupes}")
    return df


def slice_day(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(day)
    end = start + pd.Timedelta(days=1)
    out = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


def resolve_input_file(root: Path, explicit: str | None, candidates: list[Path]) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (root / p).resolve()
    for c in candidates:
        p = c if c.is_absolute() else root / c
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("Could not resolve input file from candidates: " + ", ".join(map(str, candidates)))


def write_daily_inputs(stage_dirs: list[Path], fc_day: pd.DataFrame, ac_day: pd.DataFrame) -> None:
    for d in stage_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fc_day.to_csv(d / "forecasts_rolling.csv", index=False, encoding="utf-8-sig")
        ac_day.to_csv(d / "actuals.csv", index=False, encoding="utf-8-sig")


def make_free_reservation(
    timestamps: list[pd.Timestamp] | pd.Series,
    *,
    p_max_mw: float,
    soc_min_pct: float,
    soc_max_pct: float,
) -> pd.DataFrame:
    """Reservation profile when a capacity stage is disabled.

    It uses the common reservation schema expected by FCR, aFRR/mFRR, and DA:
    timestamp, p_charge_max_mw, p_discharge_max_mw, soc_min_pct, soc_max_pct.
    """
    ts = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    return pd.DataFrame({
        "timestamp": ts,
        "p_charge_max_mw": float(p_max_mw),
        "p_discharge_max_mw": float(p_max_mw),
        "soc_min_pct": float(soc_min_pct),
        "soc_max_pct": float(soc_max_pct),
    })


def copy_or_write_reservation(df_or_path: pd.DataFrame | Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df_or_path, pd.DataFrame):
        df_or_path.to_csv(dst, index=False, encoding="utf-8-sig")
    else:
        copy_required(Path(df_or_path), dst, "reservation")


def run_child(
    *,
    stage_dir: Path,
    code: str,
    payload: dict[str, Any],
    log_path: Path,
    script_name: str,
    show_console: bool,
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload_path = log_path.with_suffix(".payload.json")
    script_path = stage_dir / script_name
    summary_path = Path(payload["summary_json"])

    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    script_path.write_text(code, encoding="utf-8")

    cmd = [sys.executable, str(script_path), str(payload_path)]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(stage_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - t0

    log_text = (
        f"COMMAND: {' '.join(cmd)}\n"
        f"CWD: {stage_dir}\n"
        f"RETURN_CODE: {proc.returncode}\n"
        f"ELAPSED_SECONDS: {elapsed:.3f}\n\n"
        "--- STDOUT ---\n"
        f"{proc.stdout}\n\n"
        "--- STDERR ---\n"
        f"{proc.stderr}\n"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")

    if show_console:
        print(log_text)

    if proc.returncode != 0:
        raise RuntimeError(f"Stage failed. See log: {log_path}")

    if not summary_path.exists():
        raise RuntimeError(f"Stage did not create summary JSON: {summary_path}. See log: {log_path}")

    out = json.loads(summary_path.read_text(encoding="utf-8"))
    out["elapsed_seconds"] = elapsed
    return out


def copy_required(src: Path, dst: Path, label: str) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing {label}: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def last_soc_from_summary(summary: dict[str, Any]) -> float:
    if "soc_end_pct" not in summary:
        raise ValueError(f"DA summary does not contain soc_end_pct: {summary}")
    soc = float(summary["soc_end_pct"])
    if not (0.0 <= soc <= 100.0):
        raise ValueError(f"Invalid end SoC percentage from DA stage: {soc}")
    return soc


def sum_csv_col(path: Path, col: str) -> float | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if col not in df.columns:
        return None
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())


def _main_single_run() -> None:
    parser = argparse.ArgumentParser(
        description="Run monthly market scenarios with switches for DA, mFRR energy, FCR, and aFRR/mFRR capacity, chaining end-of-day SoC."
    )
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root-dir", default=".", help="Root folder containing DA, FCR, and MFRR&AFRR folders.")
    parser.add_argument("--fcr-dir", default="FCR", help="FCR stage folder relative to root-dir, or absolute path.")
    parser.add_argument("--capacity-dir", default="MFRR&AFRR", help="aFRR/mFRR capacity folder relative to root-dir, or absolute path.")
    parser.add_argument("--da-dir", default="DA", help="DA stage folder relative to root-dir, or absolute path.")
    parser.add_argument("--data-dir", default="Data", help="Read original monthly CSVs from this folder relative to root-dir. Default: Data.")
    parser.add_argument("--forecast-file", default=None, help="Monthly forecast CSV. Default: data/forecasts_rolling.csv. The source file is never overwritten.")
    parser.add_argument("--actuals-file", default=None, help="Monthly actuals CSV. Default: data/actuals.csv. The source file is never overwritten.")
    parser.add_argument("--output-dir", default="monthly_full_chain_results", help="Output folder for chained monthly results.")
    parser.add_argument("--initial-soc-pct", type=float, default=50.0, help="First day start SoC percentage. Default: 50.")
    parser.add_argument(
        "--scenario",
        default="S5",
        help="Built-in scenario: S1=DA only, S2=DA+FCR, S3=DA+FCR+aFRR/mFRR capacity, S4=DA+mFRR energy, S5=full stack, or custom.",
    )
    parser.add_argument("--enable-da", type=parse_bool_or_none, default=None, help="Override DA stage: true/false/none.")
    parser.add_argument("--enable-mfrr-energy", type=parse_bool_or_none, default=None, help="Override mFRR balancing-energy participation inside DA: true/false/none.")
    parser.add_argument("--enable-fcr", type=parse_bool_or_none, default=None, help="Override FCR capacity stage: true/false/none.")
    parser.add_argument("--enable-afrr-mfrr-capacity", type=parse_bool_or_none, default=None, help="Override aFRR/mFRR capacity stage: true/false/none.")
    parser.add_argument("--free-reservation-p-max-mw", type=float, default=5.0, help="Power used for free reservation when a capacity stage is disabled. Default: 5 MW.")
    parser.add_argument("--free-reservation-soc-min-pct", type=float, default=20.0, help="SoC minimum used in free reservation. Default: 20%%.")
    parser.add_argument("--free-reservation-soc-max-pct", type=float, default=80.0, help="SoC maximum used in free reservation. Default: 80%%.")
    parser.add_argument("--horizon", type=int, default=96, help="Daily number of intervals. Default: 96 for 15-min resolution.")
    parser.add_argument("--fcr-module", default="FCR", help="Importable FCR module name inside FCR folder.")
    parser.add_argument("--capacity-module", default="afrr_mfrr", help="Importable aFRR/mFRR module name inside capacity folder.")
    parser.add_argument("--da-module", default="da_mfrr", help="Fallback DA module name inside DA folder.")
    parser.add_argument("--da-energy-module", default="da_mfrr_energy", help="Preferred DA energy module name inside DA folder.")
    parser.add_argument("--fcr-terminal-sync", type=parse_bool_or_none, default=None, help="Override SOC_TERMINAL_SYNC for FCR valuation modules: true/false/none.")
    parser.add_argument("--capacity-terminal-sync", type=parse_bool_or_none, default=None, help="Override SOC_TERMINAL_SYNC for capacity valuation modules: true/false/none.")
    parser.add_argument("--final-da-terminal-sync", type=parse_bool_or_none, default=False, help="Override terminal sync for final DA dispatch. Default: false so end SoC can be chained.")
    parser.add_argument("--soc-target-pct", type=float, default=50.0, help="Operational DA target/reference SoC where supported. Default: 50.")
    parser.add_argument("--show-stage-console", action="store_true", help="Print captured stage stdout/stderr after each stage.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue to next day after a failed day instead of stopping.")
    args = parser.parse_args()
    switches = scenario_switches(args)
    scen_label = scenario_label(args, switches)

    root = Path(args.root_dir).resolve()
    fcr_dir = Path(args.fcr_dir)
    capacity_dir = Path(args.capacity_dir)
    da_dir = Path(args.da_dir)
    fcr_dir = fcr_dir if fcr_dir.is_absolute() else root / fcr_dir
    capacity_dir = capacity_dir if capacity_dir.is_absolute() else root / capacity_dir
    da_dir = da_dir if da_dir.is_absolute() else root / da_dir
    data_dir = Path(args.data_dir)
    data_dir = data_dir if data_dir.is_absolute() else root / data_dir

    output_dir = Path(args.output_dir)
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_file = resolve_input_file(
        root,
        args.forecast_file,
        [
            data_dir / "forecasts_rolling.csv",
            data_dir / "forecast_rolling.csv",
            data_dir / "forecast.csv",
        ],
    )
    actuals_file = resolve_input_file(
        root,
        args.actuals_file,
        [
            data_dir / "actuals.csv",
            data_dir / "actual.csv",
        ],
    )

    fc_all = read_with_timestamp(forecast_file, "forecast")
    ac_all = read_with_timestamp(actuals_file, "actuals")

    common_days = sorted(set(fc_all["timestamp"].dt.floor("D")) & set(ac_all["timestamp"].dt.floor("D")))
    if not common_days:
        raise ValueError("No common calendar days between forecast and actuals files.")

    start_soc_pct = float(args.initial_soc_pct)
    monthly_rows: list[dict[str, Any]] = []
    monthly_fcr_results: list[pd.DataFrame] = []
    monthly_capacity_results: list[pd.DataFrame] = []
    monthly_remaining_reservations: list[pd.DataFrame] = []
    monthly_da_reports: list[pd.DataFrame] = []

    # Quiet mode: startup scenario metadata header suppressed.

    for day in common_days:
        day = pd.Timestamp(day)
        day_label = day.strftime("%Y-%m-%d")
        day_dir = output_dir / day_label
        fcr_out_dir = day_dir / "01_FCR"
        cap_out_dir = day_dir / "02_MFRR_AFRR"
        da_out_dir = day_dir / "03_DA"
        for d in [fcr_out_dir, cap_out_dir, da_out_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"\n========== {day_label} ==========")
        print(f"Start SoC for FCR: {start_soc_pct:.4f}%")

        try:
            fc_day = slice_day(fc_all, day)
            ac_day = slice_day(ac_all, day)
            common_ts = sorted(set(fc_day["timestamp"]) & set(ac_day["timestamp"]))
            if not common_ts:
                print(f"{day_label}: skipped; no common timestamps in this day.")
                continue
            fc_day = fc_day[fc_day["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
            ac_day = ac_day[ac_day["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)

            # Persist daily inputs in output folder for audit.
            fc_day.to_csv(day_dir / "forecasts_rolling.csv", index=False, encoding="utf-8-sig")
            ac_day.to_csv(day_dir / "actuals.csv", index=False, encoding="utf-8-sig")

            # Stage modules use fixed relative file names, so write daily slices
            # into each stage folder. This does NOT touch the original monthly
            # CSVs in data_dir. The stage folders may contain only the current
            # day's working inputs after the run, by design.
            stage_input_dirs = [da_dir]

            if switches["enable_afrr_mfrr_capacity"]:
                stage_input_dirs.append(capacity_dir)

            # Do not write daily inputs into root/FCR unless the separate FCR
            # stage is really enabled. In the current 3-scenario setup, FCR is
            # handled inside S3 reserve competition, so root/FCR is not needed.
            if RUN_SEPARATE_FCR_STAGE and switches["enable_fcr"]:
                stage_input_dirs.append(fcr_dir)

            write_daily_inputs(stage_input_dirs, fc_day, ac_day)

            # -----------------------------------------------------------------
            # 1) FCR, or pass-through free reservation when disabled
            # -----------------------------------------------------------------
            fcr_results = fcr_out_dir / "market_bidding_results_fcr_with_recovery_lock.csv"
            fcr_reservation = fcr_out_dir / "battery_reservations_fcr_symmetric.csv"
            fcr_submitted_wc = fcr_out_dir / "battery_reservations_fcr_symmetric_submitted_worst_case.csv"
            fcr_summary_json = fcr_out_dir / "fcr_day_summary.json"

            if switches["enable_fcr"]:
                fcr_summary = run_child(
                    stage_dir=fcr_dir,
                    code=FCR_CHILD_CODE,
                    payload={
                        "stage_dir": str(fcr_dir),
                        "module_name": args.fcr_module,
                        "config_module": "config",
                        "soc_init_pct": start_soc_pct,
                        "horizon": min(args.horizon, len(common_ts)),
                        "terminal_sync": args.fcr_terminal_sync,
                        "enable_da": switches["enable_da"],
                        "enable_fcr": switches["enable_fcr"],
                        "enable_mfrr_energy": switches["enable_mfrr_energy"],
                        "enable_afrr_mfrr_capacity": switches["enable_afrr_mfrr_capacity"],
                        "results_file": str(fcr_results),
                        "reservation_file": str(fcr_reservation),
                        "submitted_worst_case_file": str(fcr_submitted_wc),
                        "summary_json": str(fcr_summary_json),
                        "print_console": False,
                    },
                    log_path=fcr_out_dir / "fcr_stage.log",
                    script_name="__run_chain_fcr_stage.py",
                    show_console=args.show_stage_console,
                )
                print(f"FCR revenue: {fcr_summary.get('total_fcr_revenue_eur')}")
            else:
                free_res = make_free_reservation(
                    common_ts,
                    p_max_mw=args.free_reservation_p_max_mw,
                    soc_min_pct=args.free_reservation_soc_min_pct,
                    soc_max_pct=args.free_reservation_soc_max_pct,
                )
                free_res.to_csv(fcr_reservation, index=False, encoding="utf-8-sig")
                free_res.to_csv(fcr_submitted_wc, index=False, encoding="utf-8-sig")
                fcr_summary = {
                    "status": "SKIPPED",
                    "soc_init_pct": start_soc_pct,
                    "total_fcr_revenue_eur": 0.0,
                    "fcr_forecast_scheduling_objective_eur": 0.0,
                    "elapsed_seconds": 0.0,
                }
                fcr_summary_json.write_text(json.dumps(fcr_summary, indent=2, ensure_ascii=False), encoding="utf-8")
                # Quiet mode: FCR skipped message suppressed.

            # -----------------------------------------------------------------
            # 2) aFRR/mFRR capacity, or pass-through previous reservation
            # -----------------------------------------------------------------
            cap_results = cap_out_dir / "market_bidding_results_afrr_mfrr_capacity_after_fcr.csv"
            cap_reservation = cap_out_dir / "battery_reservations_after_fcr_afrr_mfrr_capacity.csv"
            cap_submitted_wc = cap_out_dir / "battery_reservations_after_fcr_afrr_mfrr_submitted_worst_case.csv"
            cap_summary_json = cap_out_dir / "afrr_mfrr_day_summary.json"

            if switches["enable_afrr_mfrr_capacity"]:
                # aFRR/mFRR stage expects FCR reservation under this fixed file name.
                copy_required(fcr_reservation, capacity_dir / "battery_reservations_fcr_symmetric.csv", "FCR/free reservation")

                cap_summary = run_child(
                    stage_dir=capacity_dir,
                    code=CAPACITY_CHILD_CODE,
                    payload={
                        "stage_dir": str(capacity_dir),
                        "module_name": args.capacity_module,
                        "market_config_module": "market_config",
                        "config_module": "config_capacity",
                        "soc_init_pct": start_soc_pct,
                        "horizon": min(args.horizon, len(common_ts)),
                        "terminal_sync": args.capacity_terminal_sync,
                        "enable_da": switches["enable_da"],
                        "enable_fcr": switches["enable_fcr"],
                        "enable_mfrr_energy": switches["enable_mfrr_energy"],
                        "enable_afrr_mfrr_capacity": switches["enable_afrr_mfrr_capacity"],
                        "results_file": str(cap_results),
                        "reservation_file": str(cap_reservation),
                        "submitted_worst_case_file": str(cap_submitted_wc),
                        "summary_json": str(cap_summary_json),
                        "use_actuals_for_backtest_clearing": True,
                        "print_console": False,
                    },
                    log_path=cap_out_dir / "afrr_mfrr_stage.log",
                    script_name="__run_chain_afrr_mfrr_stage.py",
                    show_console=args.show_stage_console,
                )
                print(f"aFRR/mFRR capacity revenue: {cap_summary.get('total_afrr_mfrr_capacity_revenue_eur')}")
            else:
                # No capacity stacking. Pass FCR/free reservation directly to DA.
                copy_required(fcr_reservation, cap_reservation, "FCR/free reservation")
                copy_required(fcr_reservation, cap_submitted_wc, "FCR/free reservation")
                cap_summary = {
                    "status": "SKIPPED",
                    "soc_init_pct": start_soc_pct,
                    "total_afrr_mfrr_capacity_revenue_eur": 0.0,
                    "dp_forecast_scheduling_objective_eur": 0.0,
                    "elapsed_seconds": 0.0,
                }
                cap_summary_json.write_text(json.dumps(cap_summary, indent=2, ensure_ascii=False), encoding="utf-8")
                # Quiet mode: capacity skipped message suppressed.

            # Audit copy of remaining battery capacity after enabled capacity stages.
            remaining_copy = day_dir / "capacity_remaining_after_enabled_capacity_markets.csv"
            copy_required(cap_reservation, remaining_copy, "remaining capacity reservation")
            # Backward-compatible name used by previous full-stack runner.
            copy_required(cap_reservation, day_dir / "capacity_remaining_after_fcr_afrr_mfrr.csv", "remaining capacity reservation")

            # -----------------------------------------------------------------
            # 3) DA / mFRR-energy dispatch to end of day, or no-DA pass-through
            # -----------------------------------------------------------------
            da_report = da_out_dir / "da_mfrr_after_capacity_report.csv"
            da_summary_json = da_out_dir / "da_day_summary.json"
            da_summary_csv = da_out_dir / "da_mfrr_after_capacity_summary.csv"

            if switches["enable_da"]:
                # DA stage expects final capacity reservation under this fixed file name.
                copy_required(cap_reservation, da_dir / "battery_reservations_after_fcr_afrr_mfrr_capacity.csv", "final capacity reservation")

                da_summary = run_child(
                    stage_dir=da_dir,
                    code=DA_CHILD_CODE,
                    payload={
                        "stage_dir": str(da_dir),
                        "market_config_module": "market_config",
                        "da_energy_module": args.da_energy_module,
                        "da_module": args.da_module,
                        "soc_init_pct": start_soc_pct,
                        "soc_target_pct": args.soc_target_pct,
                        "horizon": min(args.horizon, len(common_ts)),
                        "final_da_terminal_sync": args.final_da_terminal_sync,
                        "enable_da": switches["enable_da"],
                        "enable_fcr": switches["enable_fcr"],
                        "enable_mfrr_energy": switches["enable_mfrr_energy"],
                        "enable_afrr_mfrr_capacity": switches["enable_afrr_mfrr_capacity"],
                        "report_file": str(da_report),
                        "summary_csv": str(da_summary_csv),
                        "summary_json": str(da_summary_json),
                        "print_console": False,
                        "print_solver": False,
                    },
                    log_path=da_out_dir / "da_stage.log",
                    script_name="__run_chain_da_stage.py",
                    show_console=args.show_stage_console,
                )

                end_soc_pct = last_soc_from_summary(da_summary)
                da_cash_actual = da_summary.get("da_cashflow_eur_actual")
                mfrr_cash_actual = da_summary.get("mfrr_energy_cashflow_eur_actual")
                imb_actual = da_summary.get("imbalance_penalty_eur_actual")
                degr_actual = da_summary.get("battery_degradation_eur")
                net_actual = da_summary.get("market_net_cashflow_eur_actual")
                net_forecast = da_summary.get("net_cashflow_eur_forecast")

                def _fmt_eur(x):
                    return "n/a" if x is None else f"{float(x):.2f} EUR"

                print(
                    "DA result: "
                    f"DA={_fmt_eur(da_cash_actual)} | "
                    
                    f"imb_pen={_fmt_eur(imb_actual)} | "
                    f"degr={_fmt_eur(degr_actual)} | "
                    
                    
                    f"end SoC={end_soc_pct:.4f}%"
                )
            else:
                da_summary = {
                    "status": "SKIPPED",
                    "engine": "no_da_stage",
                    "soc_init_pct": start_soc_pct,
                    "soc_end_pct": start_soc_pct,
                    "da_cashflow_eur_actual": 0.0,
                    "mfrr_energy_cashflow_eur_actual": 0.0,
                    "imbalance_penalty_eur_actual": 0.0,
                    "battery_degradation_eur": 0.0,
                    "market_net_cashflow_eur_actual": 0.0,
                    "da_cashflow_eur_forecast": 0.0,
                    "mfrr_energy_cashflow_eur_forecast": 0.0,
                    "net_cashflow_eur_forecast": 0.0,
                    "elapsed_seconds": 0.0,
                }
                da_summary_json.write_text(json.dumps(da_summary, indent=2, ensure_ascii=False), encoding="utf-8")
                end_soc_pct = start_soc_pct
                print("DA skipped; end SoC is unchanged.")

            # -----------------------------------------------------------------
            # Monthly aggregation
            # -----------------------------------------------------------------
            row = {
                "date": day_label,
                "scenario": scen_label,
                "enable_da": switches["enable_da"],
                "enable_mfrr_energy": switches["enable_mfrr_energy"],
                "enable_fcr": switches["enable_fcr"],
                "enable_afrr_mfrr_capacity": switches["enable_afrr_mfrr_capacity"],
                "start_soc_pct": round(start_soc_pct, 6),
                "end_soc_pct": round(end_soc_pct, 6),
                "intervals": len(common_ts),
                "fcr_status": fcr_summary.get("status"),
                "afrr_mfrr_status": cap_summary.get("status"),
                "da_status": da_summary.get("status"),
                "da_engine": da_summary.get("engine"),
                "fcr_revenue_eur": fcr_summary.get("total_fcr_revenue_eur"),
                "afrr_mfrr_capacity_revenue_eur": cap_summary.get("total_afrr_mfrr_capacity_revenue_eur"),
                "da_cashflow_eur_actual": da_summary.get("da_cashflow_eur_actual"),
                "mfrr_energy_cashflow_eur_actual": da_summary.get("mfrr_energy_cashflow_eur_actual"),
                "imbalance_penalty_eur_actual": da_summary.get("imbalance_penalty_eur_actual"),
                "battery_degradation_eur": da_summary.get("battery_degradation_eur"),
                "da_net_cashflow_eur_actual": da_summary.get("market_net_cashflow_eur_actual"),
                "da_cashflow_eur_forecast": da_summary.get("da_cashflow_eur_forecast"),
                "mfrr_energy_cashflow_eur_forecast": da_summary.get("mfrr_energy_cashflow_eur_forecast"),
                "da_net_cashflow_eur_forecast": da_summary.get("net_cashflow_eur_forecast"),
                "total_revenue_market_actual_eur": (
                    float(fcr_summary.get("total_fcr_revenue_eur") or 0.0)
                    + float(cap_summary.get("total_afrr_mfrr_capacity_revenue_eur") or 0.0)
                    + float(da_summary.get("market_net_cashflow_eur_actual") or 0.0)
                ),
                "fcr_stage_seconds": fcr_summary.get("elapsed_seconds"),
                "afrr_mfrr_stage_seconds": cap_summary.get("elapsed_seconds"),
                "da_stage_seconds": da_summary.get("elapsed_seconds"),
                "day_dir": str(day_dir),
            }
            monthly_rows.append(row)

            def _fmt_day_eur(value: Any) -> str:
                return "n/a" if value is None else f"{float(value):.2f} EUR"

            print(
                "Total day market: "
                f"FCR={_fmt_day_eur(row['fcr_revenue_eur'])} | "
                f"aFRR/mFRR_capacity={_fmt_day_eur(row['afrr_mfrr_capacity_revenue_eur'])} | "
                f"DA_net={_fmt_day_eur(row['da_net_cashflow_eur_actual'])} | "
                f"TOTAL={_fmt_day_eur(row['total_revenue_market_actual_eur'])}"
            )

            if fcr_results.exists():
                x = pd.read_csv(fcr_results)
                x.insert(0, "date", day_label)
                monthly_fcr_results.append(x)
            if cap_results.exists():
                x = pd.read_csv(cap_results)
                x.insert(0, "date", day_label)
                monthly_capacity_results.append(x)
            if cap_reservation.exists():
                x = pd.read_csv(cap_reservation)
                x.insert(0, "date", day_label)
                monthly_remaining_reservations.append(x)
            if da_report.exists():
                x = pd.read_csv(da_report)
                x.insert(0, "date", day_label)
                monthly_da_reports.append(x)

            # Chain SoC into next day.
            start_soc_pct = end_soc_pct

        except Exception as exc:
            print(f"ERROR on {day_label}: {exc}")
            monthly_rows.append({
                "date": day_label,
                "scenario": scen_label,
                "enable_da": switches["enable_da"],
                "enable_mfrr_energy": switches["enable_mfrr_energy"],
                "enable_fcr": switches["enable_fcr"],
                "enable_afrr_mfrr_capacity": switches["enable_afrr_mfrr_capacity"],
                "start_soc_pct": round(start_soc_pct, 6),
                "status": "FAILED",
                "error": str(exc),
                "day_dir": str(day_dir),
            })
            if not args.continue_on_error:
                raise

    # -------------------------------------------------------------------------
    # Save monthly outputs
    # -------------------------------------------------------------------------
    monthly_summary = pd.DataFrame(monthly_rows)
    monthly_summary.to_csv(output_dir / "monthly_chain_summary.csv", index=False, encoding="utf-8-sig")

    if monthly_fcr_results:
        pd.concat(monthly_fcr_results, ignore_index=True).to_csv(
            output_dir / "monthly_fcr_results.csv", index=False, encoding="utf-8-sig"
        )
    if monthly_capacity_results:
        pd.concat(monthly_capacity_results, ignore_index=True).to_csv(
            output_dir / "monthly_afrr_mfrr_capacity_results.csv", index=False, encoding="utf-8-sig"
        )
    if monthly_remaining_reservations:
        pd.concat(monthly_remaining_reservations, ignore_index=True).to_csv(
            output_dir / "monthly_remaining_capacity_after_fcr_afrr_mfrr.csv", index=False, encoding="utf-8-sig"
        )
    if monthly_da_reports:
        pd.concat(monthly_da_reports, ignore_index=True).to_csv(
            output_dir / "monthly_da_after_capacity_report.csv", index=False, encoding="utf-8-sig"
        )

    def _monthly_sum(col: str) -> float:
        if col not in monthly_summary.columns:
            return 0.0
        return float(pd.to_numeric(monthly_summary[col], errors="coerce").fillna(0.0).sum())

    print("\n========== MONTH TOTAL ==========")
    print(f"FCR revenue: {_monthly_sum('fcr_revenue_eur'):.2f} EUR")
    print(f"aFRR/mFRR capacity revenue: {_monthly_sum('afrr_mfrr_capacity_revenue_eur'):.2f} EUR")
    print(f"DA cashflow actual: {_monthly_sum('da_cashflow_eur_actual'):.2f} EUR")
    print(f"mFRR energy cashflow actual: {_monthly_sum('mfrr_energy_cashflow_eur_actual'):.2f} EUR")
    print(f"Imbalance penalty actual: {_monthly_sum('imbalance_penalty_eur_actual'):.2f} EUR")
    print(f"Battery degradation: {_monthly_sum('battery_degradation_eur'):.2f} EUR")
    print(f"DA market net actual: {_monthly_sum('da_net_cashflow_eur_actual'):.2f} EUR")
    print(f"Total market actual: {_monthly_sum('total_revenue_market_actual_eur'):.2f} EUR")

    print("\nDone.")
    print(f"Scenario: {scen_label}")
    print(f"Monthly summary: {output_dir / 'monthly_chain_summary.csv'}")
    print(f"Final chained start SoC for next period: {start_soc_pct:.4f}%")


# =============================================================================
# Scenario table runner
# =============================================================================


def _bool_to_cli(value: Any) -> str:
    return "true" if bool(value) else "false"


def _print_table_run_totals(output_dir: Path, scenario_name: str) -> None:
    summary_path = output_dir / "monthly_chain_summary.csv"
    if not summary_path.exists():
        print(f"No monthly summary found for {scenario_name}: {summary_path}")
        return
    df = pd.read_csv(summary_path)

    def s(col: str) -> float:
        if col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())

    print(f"\n--- Scenario total: {scenario_name} ---")
    print(f"FCR revenue: {s('fcr_revenue_eur'):.2f} EUR")
    print(f"aFRR/mFRR capacity revenue: {s('afrr_mfrr_capacity_revenue_eur'):.2f} EUR")
    print(f"DA cashflow actual: {s('da_cashflow_eur_actual'):.2f} EUR")
    print(f"mFRR energy cashflow actual: {s('mfrr_energy_cashflow_eur_actual'):.2f} EUR")
    print(f"DA market net actual: {s('da_net_cashflow_eur_actual'):.2f} EUR")
    print(f"Total market actual: {s('total_revenue_market_actual_eur'):.2f} EUR")
    print(f"Summary file: {summary_path}")


def _run_selected_scenarios_from_table() -> None:
    script_path = Path(__file__).resolve()
    root_dir = Path(DEFAULT_ROOT_DIR).resolve()
    selected = [(name, cfg) for name, cfg in SCENARIO_RUN_TABLE.items() if bool(cfg.get("RUN", False))]
    if not selected:
        raise RuntimeError("No scenario has RUN=True in SCENARIO_RUN_TABLE.")

    # Quiet mode: scenario-table startup header suppressed.

    for scenario_name, cfg in selected:
        output_dir = root_dir / str(cfg.get("OUTPUT_DIR", f"results_{scenario_name}"))
        cmd = [
            sys.executable, str(script_path), "--single-run", "--root-dir", str(root_dir), "--scenario", "custom",
            "--enable-da", _bool_to_cli(cfg.get("DA", False)),
            "--enable-mfrr-energy", _bool_to_cli(cfg.get("MFRR_ENERGY", False)),
            "--enable-fcr", _bool_to_cli(cfg.get("FCR", False)),
            "--enable-afrr-mfrr-capacity", _bool_to_cli(cfg.get("AFRR_MFRR_CAPACITY", False)),
            "--initial-soc-pct", str(float(cfg.get("INITIAL_SOC_PCT", DEFAULT_INITIAL_SOC_PCT))),
            "--final-da-terminal-sync", _bool_to_cli(cfg.get("FINAL_DA_TERMINAL_SYNC", DEFAULT_FINAL_DA_TERMINAL_SYNC)),
            "--soc-target-pct", str(float(cfg.get("SOC_TARGET_PCT", DEFAULT_SOC_TARGET_PCT))),
            "--da-energy-module", str(cfg.get("DA_ENERGY_MODULE", "da_mfrr_energy")),
            "--output-dir", str(output_dir),
        ]
        if cfg.get("CONTINUE_ON_ERROR", False):
            cmd.append("--continue-on-error")
        if cfg.get("SHOW_STAGE_CONSOLE", False):
            cmd.append("--show-stage-console")
        # Quiet mode: per-scenario banner suppressed.
        proc = subprocess.run(cmd, cwd=str(root_dir))
        if proc.returncode != 0:
            raise RuntimeError(f"Scenario {scenario_name} failed with return code {proc.returncode}.")
        _print_table_run_totals(output_dir, scenario_name)


def main() -> None:
    # Normal use: run this file with no arguments and edit SCENARIO_RUN_TABLE above.
    # Command-line use remains available; any CLI argument triggers one single run.
    if RUN_SELECTED_SCENARIOS_FROM_TABLE and len(sys.argv) == 1:
        _run_selected_scenarios_from_table()
    else:
        _main_single_run()


if __name__ == "__main__":
    main()
