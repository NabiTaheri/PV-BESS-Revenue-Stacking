
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
