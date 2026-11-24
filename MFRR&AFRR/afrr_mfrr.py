# =========================
# file: afrr_mfrr_capacity_bidding_dp.py
# Purpose:
#   aFRR/mFRR capacity bidding after FCR reservation.
#
# Morning-bidding logic:
#   - Starts from battery reservation left after FCR.
#   - Uses only forecast capacity prices to decide where/what to bid.
#   - Bids either UP or DOWN per interval, never both at the same time.
#   - Uses the real Scenario 2 DA+mFRR-energy engine with forecast-as-actual
#     to calculate opportunity cost and portfolio pre-check.
#   - Does not use actual prices in opportunity-cost or bid optimization.
#
# Backtest / clearing logic:
#   - Actual capacity prices are optional and used only after submitted bids
#     have already been created.
#   - If actuals are unavailable, the code still returns the submitted bid plan
#     and a submitted-worst-case reservation.
#
# Required files/modules:
#   - config_capacity.py
#   - ../DA/da_mfrr_energy.py + ../DA/market_config.py
#       used as the real Scenario 2 engine for forecast-only valuation.
#   - forecasts_rolling.csv
#   - battery_reservations_fcr_symmetric.csv
#       output from FCR stage.
# =========================

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# SCRIPT VERSION MARKER
# ============================================================
SCRIPT_VERSION = "AFRR_MFRR_CAPACITY_S2_FORECAST_STACK_PRECHECK_MUTUALLY_EXCLUSIVE_UP_DOWN_01"


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================
HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

cfg_module = importlib.import_module("config_capacity")

# Scenario-2 valuation is loaded from the real DA engine in ../DA.
# Do not use da_mfrr.py for opportunity-cost valuation.


# ============================================================
# FILE SETTINGS
# ============================================================
FORECASTS_FILE = Path(getattr(cfg_module, "FORECAST_FILE", "forecasts_rolling.csv"))

ACTUALS_FILE = Path(getattr(cfg_module, "ACTUALS_FILE", "actuals.csv"))

FCR_RESERVATIONS_FILE = Path(
    getattr(
        cfg_module,
        "FCR_RESERVATIONS_FILE",
        "battery_reservations_fcr_symmetric.csv",
    )
)

RESULTS_FILE = Path(
    getattr(
        cfg_module,
        "AFRR_MFRR_RESULTS_FILE",
        "market_bidding_results_afrr_mfrr_capacity_after_fcr.csv",
    )
)

RESERVATIONS_OUTPUT_FILE = Path(
    getattr(
        cfg_module,
        "AFRR_MFRR_RESERVATIONS_FILE",
        "battery_reservations_after_fcr_afrr_mfrr_capacity.csv",
    )
)

SUBMITTED_WORST_CASE_RESERVATIONS_FILE = Path(
    getattr(
        cfg_module,
        "AFRR_MFRR_SUBMITTED_WORST_CASE_RESERVATIONS_FILE",
        "battery_reservations_after_fcr_afrr_mfrr_submitted_worst_case.csv",
    )
)


# ============================================================
# NUMERIC SETTINGS
# ============================================================
STEP_MW = float(getattr(cfg_module, "BIDDING_STEPS_MW", {}).get("aFRR", 1.0))
EPS = float(getattr(cfg_module, "EPS", 1e-9))

PRINT_PROGRESS = bool(
    getattr(cfg_module, "PRINT_SETTINGS", {}).get(
        "PRINT_AFRR_MFRR_RISK_STATUS",
        True,
    )
)

ALLOW_NEGATIVE_FORECAST_MARGIN = bool(
    getattr(cfg_module, "ALLOW_NEGATIVE_AFRR_MFRR_FORECAST_MARGIN", False)
)

RECOVERY_START_OFFSET_INTERVALS = int(
    getattr(cfg_module, "AFRR_MFRR_RECOVERY_START_OFFSET_INTERVALS", 1)
)

RECOVERY_MAX_INTERVALS = getattr(
    cfg_module,
    "AFRR_MFRR_RECOVERY_MAX_INTERVALS",
    None,
)


# ============================================================
# BASIC HELPERS
# ============================================================
def progress(msg: str) -> None:
    if PRINT_PROGRESS:
        print(msg, flush=True)


def floor_to_step(x: float, step: float = STEP_MW) -> float:
    return float(math.floor(float(x) / step + EPS) * step)


def fmt_curve(values: list[float]) -> str:
    if not values:
        return "[]"

    return "[" + ",".join(f"{float(v):.2f}" for v in values) + "]"


def read_col(
    df: pd.DataFrame,
    col: str,
    horizon: int,
    default: float = 0.0,
) -> np.ndarray:
    if col in df.columns:
        arr = pd.to_numeric(df[col], errors="coerce").fillna(default).values[:horizon]
    else:
        arr = np.full(min(len(df), horizon), default, dtype=float)

    if len(arr) < horizon:
        arr = np.pad(arr, (0, horizon - len(arr)), constant_values=default)

    return arr.astype(float)


def read_first_available_col(
    df: pd.DataFrame,
    possible_cols: list[str],
    horizon: int,
    default: float = np.nan,
) -> np.ndarray:
    for col in possible_cols:
        if col in df.columns:
            return read_col(df, col, horizon, default)

    return np.full(horizon, default, dtype=float)


def market_enabled(market_name: str, default: bool = True) -> bool:
    return bool(getattr(cfg_module, "MARKETS", {}).get(market_name, default))


def build_cfg() -> dict[str, Any]:
    return {
        "BATTERY": cfg_module.BATTERY,
        "DT": float(getattr(cfg_module, "DT", cfg_module.RESOLUTION_MIN / 60.0)),
        "HORIZON": int(getattr(cfg_module, "HORIZON", 96)),
        "RESERVE_BUFFER_HOURS": getattr(cfg_module, "RESERVE_BUFFER_HOURS", {}),
        "RISK_MANAGEMENT": getattr(cfg_module, "RISK_MANAGEMENT", {}),
        "MARKETS": getattr(cfg_module, "MARKETS", {}),
    }


def reserve_buffer_h(market: str) -> float:
    buffers = getattr(cfg_module, "RESERVE_BUFFER_HOURS", {})

    return float(
        buffers.get(
            market,
            buffers.get(market.lower(), 0.25),
        )
    )


def _resolve_existing_path(path: Path) -> Path:
    if path.exists():
        return path

    matches = sorted(HERE.glob(path.stem + "*.csv"))

    if matches:
        return matches[0]

    return path


def make_free_battery_reservation_profile(
    timestamps: pd.Series,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    b = cfg["BATTERY"]

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "p_charge_max_mw": float(b["p_max_mw"]),
            "p_discharge_max_mw": float(b["p_max_mw"]),
            "soc_min_pct": float(b["soc_min_pct"]),
            "soc_max_pct": float(b["soc_max_pct"]),
        }
    )


# ============================================================
# FORECAST DATA LOADING - NO ACTUALS HERE
# ============================================================
def load_forecast_input_data(horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    fc_path = _resolve_existing_path(FORECASTS_FILE)

    if not fc_path.exists():
        raise FileNotFoundError(f"Missing forecast file: {FORECASTS_FILE}")

    df_fc = pd.read_csv(fc_path)

    if "timestamp" not in df_fc.columns:
        raise ValueError(f"{fc_path} must contain timestamp column.")

    df_fc["timestamp"] = pd.to_datetime(df_fc["timestamp"])
    df_fc = df_fc.iloc[:horizon].copy().reset_index(drop=True)

    if len(df_fc) < horizon:
        raise ValueError(f"{fc_path} has only {len(df_fc)} rows, horizon={horizon}.")

    timestamps = df_fc["timestamp"].reset_index(drop=True)

    return df_fc, timestamps


def get_forecast_capacity_price_data(
    df_fc: pd.DataFrame,
    horizon: int,
) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {}

    afrr_common = read_first_available_col(
        df_fc,
        ["afrr_capacity_price_forecast", "afrr_price_forecast"],
        horizon,
        default=np.nan,
    )

    mfrr_common = read_first_available_col(
        df_fc,
        ["mfrr_capacity_price_forecast", "mfrr_price_forecast"],
        horizon,
        default=np.nan,
    )

    data["fc_afrr_up"] = read_first_available_col(
        df_fc,
        [
            "afrr_capacity_price_up_forecast",
            "afrr_up_capacity_price_forecast",
            "afrr_up_price_forecast",
        ],
        horizon,
        default=np.nan,
    )

    data["fc_afrr_down"] = read_first_available_col(
        df_fc,
        [
            "afrr_capacity_price_down_forecast",
            "afrr_down_capacity_price_forecast",
            "afrr_down_price_forecast",
        ],
        horizon,
        default=np.nan,
    )

    data["fc_mfrr_up"] = read_first_available_col(
        df_fc,
        [
            "mfrr_capacity_price_up_forecast",
            "mfrr_up_capacity_price_forecast",
            "mfrr_up_price_forecast",
        ],
        horizon,
        default=np.nan,
    )

    data["fc_mfrr_down"] = read_first_available_col(
        df_fc,
        [
            "mfrr_capacity_price_down_forecast",
            "mfrr_down_capacity_price_forecast",
            "mfrr_down_price_forecast",
        ],
        horizon,
        default=np.nan,
    )

    data["fc_afrr_up"] = np.where(np.isnan(data["fc_afrr_up"]), afrr_common, data["fc_afrr_up"])
    data["fc_afrr_down"] = np.where(np.isnan(data["fc_afrr_down"]), afrr_common, data["fc_afrr_down"])
    data["fc_mfrr_up"] = np.where(np.isnan(data["fc_mfrr_up"]), mfrr_common, data["fc_mfrr_up"])
    data["fc_mfrr_down"] = np.where(np.isnan(data["fc_mfrr_down"]), mfrr_common, data["fc_mfrr_down"])

    return data


# ============================================================
# OPTIONAL ACTUAL DATA - CLEARING/BACKTEST ONLY
# ============================================================
def load_actual_capacity_price_data_if_available(
    timestamps: pd.Series,
    horizon: int,
) -> dict[str, np.ndarray] | None:
    """Load actual capacity prices only for ex-post clearing/backtest.

    This function must never be called before bid optimization.
    """
    ac_path = _resolve_existing_path(ACTUALS_FILE)

    if not ac_path.exists():
        return None

    df_act = pd.read_csv(ac_path)

    if "timestamp" in df_act.columns:
        df_act["timestamp"] = pd.to_datetime(df_act["timestamp"])

        df_act = (
            pd.DataFrame({"timestamp": pd.to_datetime(timestamps)})
            .merge(df_act, on="timestamp", how="left")
            .reset_index(drop=True)
        )
    else:
        df_act = df_act.iloc[:horizon].copy().reset_index(drop=True)

    data: dict[str, np.ndarray] = {}

    data["act_afrr_up"] = read_first_available_col(
        df_act,
        [
            "afrr_capacity_price_up",
            "afrr_up_capacity_price",
            "afrr_up_price",
        ],
        horizon,
        default=np.nan,
    )

    data["act_afrr_down"] = read_first_available_col(
        df_act,
        [
            "afrr_capacity_price_down",
            "afrr_down_capacity_price",
            "afrr_down_price",
        ],
        horizon,
        default=np.nan,
    )

    data["act_mfrr_up"] = read_first_available_col(
        df_act,
        [
            "mfrr_capacity_price_up",
            "mfrr_up_capacity_price",
            "mfrr_up_price",
        ],
        horizon,
        default=np.nan,
    )

    data["act_mfrr_down"] = read_first_available_col(
        df_act,
        [
            "mfrr_capacity_price_down",
            "mfrr_down_capacity_price",
            "mfrr_down_price",
        ],
        horizon,
        default=np.nan,
    )

    return data


# ============================================================
# FCR RESERVATION LOADING
# ============================================================
def normalize_reservation_columns(
    res: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    out = res.copy().reset_index(drop=True)
    b = cfg["BATTERY"]

    required = ["timestamp", "p_charge_max_mw", "p_discharge_max_mw"]
    missing = [c for c in required if c not in out.columns]

    if missing:
        raise ValueError(f"Reservation data missing columns: {missing}")

    if "soc_min_pct" not in out.columns:
        out["soc_min_pct"] = float(b["soc_min_pct"])

    if "soc_max_pct" not in out.columns:
        out["soc_max_pct"] = float(b["soc_max_pct"])

    out["timestamp"] = pd.to_datetime(out["timestamp"])

    out["p_charge_max_mw"] = (
        pd.to_numeric(out["p_charge_max_mw"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, float(b["p_max_mw"]))
        .astype(float)
    )

    out["p_discharge_max_mw"] = (
        pd.to_numeric(out["p_discharge_max_mw"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, float(b["p_max_mw"]))
        .astype(float)
    )

    out["soc_min_pct"] = (
        pd.to_numeric(out["soc_min_pct"], errors="coerce")
        .fillna(float(b["soc_min_pct"]))
        .astype(float)
    )

    out["soc_max_pct"] = (
        pd.to_numeric(out["soc_max_pct"], errors="coerce")
        .fillna(float(b["soc_max_pct"]))
        .astype(float)
    )

    bad = out["soc_min_pct"] > out["soc_max_pct"] + EPS

    if bad.any():
        raise ValueError(
            "Reservation file has infeasible SOC band in some intervals:\n"
            + str(out.loc[bad, ["timestamp", "soc_min_pct", "soc_max_pct"]])
        )

    return out


def load_fcr_reservations(
    timestamps: pd.Series,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    fcr_path = _resolve_existing_path(FCR_RESERVATIONS_FILE)

    if not fcr_path.exists():
        raise FileNotFoundError(
            f"Missing FCR reservation file: {FCR_RESERVATIONS_FILE}. "
            "aFRR/mFRR capacity bidding must start from capacity left after FCR."
        )

    res = pd.read_csv(fcr_path)
    res["timestamp"] = pd.to_datetime(res["timestamp"])

    tmp = pd.DataFrame({"timestamp": pd.to_datetime(timestamps)})
    joined = tmp.merge(res, on="timestamp", how="left")

    if joined[["p_charge_max_mw", "p_discharge_max_mw"]].isna().any().any():
        if len(res) != len(timestamps):
            raise ValueError(
                "FCR reservation timestamps do not match forecast timestamps and row count differs."
            )

        joined = res.iloc[: len(timestamps)].copy().reset_index(drop=True)
        joined["timestamp"] = pd.to_datetime(timestamps)

    # progress(f"Using FCR reservation file: {fcr_path}")

    return normalize_reservation_columns(joined, cfg)


# ============================================================
# CAPACITY MARKET SELECTION
# ============================================================
def best_capacity_market_forecast(
    price_data: dict[str, np.ndarray],
    t: int,
    direction: str,
) -> tuple[str, float]:
    candidates: list[tuple[str, float]] = []

    if direction == "up":
        if market_enabled("aFRR_Capacity", True):
            p = float(price_data["fc_afrr_up"][t])
            if not np.isnan(p):
                candidates.append(("aFRR", p))

        if market_enabled("mFRR_Capacity", True):
            p = float(price_data["fc_mfrr_up"][t])
            if not np.isnan(p):
                candidates.append(("mFRR", p))

    else:
        if market_enabled("aFRR_Capacity", True):
            p = float(price_data["fc_afrr_down"][t])
            if not np.isnan(p):
                candidates.append(("aFRR", p))

        if market_enabled("mFRR_Capacity", True):
            p = float(price_data["fc_mfrr_down"][t])
            if not np.isnan(p):
                candidates.append(("mFRR", p))

    if not candidates:
        return "None", 0.0

    return max(candidates, key=lambda x: x[1])


def actual_clearing_price_for_market(
    actual_price_data: dict[str, np.ndarray] | None,
    t: int,
    market: str,
    direction: str,
) -> float:
    if actual_price_data is None:
        return float("nan")

    if market == "aFRR":
        return float(actual_price_data[f"act_afrr_{direction}"][t])

    if market == "mFRR":
        return float(actual_price_data[f"act_mfrr_{direction}"][t])

    return float("nan")


# ============================================================
# RESERVATION FREEZE / AVAILABILITY
# ============================================================
def available_capacity_mw(
    res: pd.DataFrame,
    t: int,
    direction: str,
    cfg: dict[str, Any],
) -> float:
    """Available capacity after FCR reservation.

    UP uses discharge headroom.
    DOWN uses charge headroom.
    The energy limit uses the currently available SOC band in the reservation.
    """
    b = cfg["BATTERY"]

    e_max = float(b["e_max_mwh"])
    eta = float(b["efficiency"])
    buffer_h = reserve_buffer_h("aFRR")

    soc_min = float(res.loc[t, "soc_min_pct"])
    soc_max = float(res.loc[t, "soc_max_pct"])

    available_band_mwh = max(0.0, (soc_max - soc_min) / 100.0 * e_max)

    if direction == "up":
        power_limit = max(0.0, float(res.loc[t, "p_discharge_max_mw"]))
        energy_limit = available_band_mwh * eta / buffer_h if buffer_h > EPS else power_limit
    else:
        power_limit = max(0.0, float(res.loc[t, "p_charge_max_mw"]))
        energy_limit = available_band_mwh / max(buffer_h * eta, EPS) if buffer_h > EPS else power_limit

    return floor_to_step(min(power_limit, energy_limit), STEP_MW)


def make_mw_steps(available_mw: float) -> list[float]:
    available_mw = floor_to_step(available_mw, STEP_MW)

    if available_mw < STEP_MW - EPS:
        return []

    n = int(math.floor(available_mw / STEP_MW + EPS))

    return [float(i * STEP_MW) for i in range(1, n + 1)]


def apply_capacity_freeze(
    base_res: pd.DataFrame,
    interval: int,
    direction: str,
    q_mw: float,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    out = base_res.copy()

    b = cfg["BATTERY"]
    e_max = float(b["e_max_mwh"])
    eta = float(b["efficiency"])
    buffer_h = reserve_buffer_h("aFRR")

    q = float(q_mw)

    if q <= EPS:
        return out

    if out["p_charge_max_mw"].dtype != np.dtype("float64"):
        out["p_charge_max_mw"] = out["p_charge_max_mw"].astype(float)
    if out["p_discharge_max_mw"].dtype != np.dtype("float64"):
        out["p_discharge_max_mw"] = out["p_discharge_max_mw"].astype(float)
    if out["soc_min_pct"].dtype != np.dtype("float64"):
        out["soc_min_pct"] = out["soc_min_pct"].astype(float)
    if out["soc_max_pct"].dtype != np.dtype("float64"):
        out["soc_max_pct"] = out["soc_max_pct"].astype(float)

    if direction == "up":
        old = float(out.at[interval, "p_discharge_max_mw"])

        if q > old + EPS:
            raise ValueError(
                f"Not enough discharge capacity at interval={interval}: "
                f"available={old}, needed={q}"
            )

        out.at[interval, "p_discharge_max_mw"] = max(0.0, old - q)

        out.at[interval, "soc_min_pct"] = min(
            100.0,
            float(out.at[interval, "soc_min_pct"])
            + 100.0 * q * buffer_h / max(eta, EPS) / e_max,
        )

    elif direction == "down":
        old = float(out.at[interval, "p_charge_max_mw"])

        if q > old + EPS:
            raise ValueError(
                f"Not enough charge capacity at interval={interval}: "
                f"available={old}, needed={q}"
            )

        out.at[interval, "p_charge_max_mw"] = max(0.0, old - q)

        out.at[interval, "soc_max_pct"] = max(
            0.0,
            float(out.at[interval, "soc_max_pct"])
            - 100.0 * q * buffer_h * eta / e_max,
        )

    else:
        raise ValueError("direction must be 'up' or 'down'.")

    if float(out.at[interval, "soc_min_pct"]) > float(out.at[interval, "soc_max_pct"]) + EPS:
        raise ValueError(
            f"Infeasible capacity freeze at interval={interval}, "
            f"direction={direction}, q={q}"
        )

    return out


def joint_capacity_freeze_feasible(
    base_res: pd.DataFrame,
    interval: int,
    up_mw: float,
    down_mw: float,
    cfg: dict[str, Any],
) -> bool:
    try:
        test = base_res

        if up_mw > EPS:
            test = apply_capacity_freeze(test, interval, "up", up_mw, cfg)

        if down_mw > EPS:
            test = apply_capacity_freeze(test, interval, "down", down_mw, cfg)

        return True
    except Exception:
        return False


# ============================================================
# OPPORTUNITY COST CURVES USING DA+mFRR FORECAST ENGINE
# ============================================================
def _find_da_dir() -> Path:
    candidates = [
        HERE.parent / "DA",
        HERE.parent.parent / "DA",
        HERE,
    ]

    for candidate in candidates:
        if (candidate / "da_mfrr_energy.py").exists() and (candidate / "market_config.py").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find Scenario-2 DA engine. Expected da_mfrr_energy.py "
        "and market_config.py in ../DA. This module must not fall back to da_mfrr.py."
    )


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


_DA_ENGINE_CACHE: dict[str, Any] = {}


def _load_s2_engine() -> tuple[Any, Any, Path]:
    cached = _DA_ENGINE_CACHE.get("engine")

    if cached is not None:
        return cached

    da_dir = _find_da_dir()

    original_sys_path = list(sys.path)
    original_market_config = sys.modules.get("market_config")

    try:
        if str(da_dir) in sys.path:
            sys.path.remove(str(da_dir))
        sys.path.insert(0, str(da_dir))

        if "market_config" in sys.modules:
            del sys.modules["market_config"]

        market_config = _load_module_from_path("market_config", da_dir / "market_config.py")
        da_engine = _load_module_from_path("da_mfrr_energy", da_dir / "da_mfrr_energy.py")

        if not hasattr(market_config, "MarketConfig"):
            raise AttributeError(f"{da_dir / 'market_config.py'} has no MarketConfig dataclass.")

        if not hasattr(da_engine, "run"):
            raise AttributeError(f"{da_dir / 'da_mfrr_energy.py'} has no run(cfg) function.")

        cached = (market_config, da_engine, da_dir)
        _DA_ENGINE_CACHE["engine"] = cached

        return cached
    finally:
        sys.path[:] = original_sys_path

        if original_market_config is not None:
            sys.modules["market_config"] = original_market_config


def _forecast_as_actuals_file(df_fc: pd.DataFrame, path: Path) -> Path:
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df_fc["timestamp"])

    mappings = {
        "pv_actual_mw": ["pv_forecast_mw"],
        "da_clearing_price": ["da_price_forecast"],
        "balancing_price": ["balancing_price_forecast"],
        "balancing_flag": ["balancing_flag_forecast"],
        "balancing_flag_actual": ["balancing_flag_forecast"],
        "imbalance_price_actual": ["imbalance_price_forecast"],
        "fcr_clearing_price": ["fcr_price_forecast", "fcr_capacity_price_forecast", "fcr_clearing_price_forecast"],
        "afrr_capacity_price_up": ["afrr_capacity_price_up_forecast", "afrr_up_capacity_price_forecast", "afrr_capacity_price_forecast"],
        "afrr_capacity_price_down": ["afrr_capacity_price_down_forecast", "afrr_down_capacity_price_forecast", "afrr_capacity_price_forecast"],
        "mfrr_capacity_price_up": ["mfrr_capacity_price_up_forecast", "mfrr_up_capacity_price_forecast", "mfrr_capacity_price_forecast"],
        "mfrr_capacity_price_down": ["mfrr_capacity_price_down_forecast", "mfrr_down_capacity_price_forecast", "mfrr_capacity_price_forecast"],
    }

    for target, sources in mappings.items():
        for source in sources:
            if source in df_fc.columns:
                out[target] = pd.to_numeric(df_fc[source], errors="coerce").fillna(0.0)
                break

        if target not in out.columns:
            out[target] = 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")

    return path


def _sum_first_existing_column(df: pd.DataFrame, columns: list[str]) -> float:
    for column in columns:
        if column in df.columns:
            return float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).sum())

    raise ValueError(f"Scenario-2 report has none of these value columns: {columns}")


_EXACT_SPARSE_EVALUATOR = None
_EXACT_SPARSE_SIGNATURE = None


def run_s2_forecast_value_sparse(reservation_df: pd.DataFrame) -> float:
    """Same S2 forecast objective/constraints, sparse cached MILP backend.

    Each reserved capacity MW re-optimizes the original
    MILP. Static coefficient matrix and forecast parsing are reused.
    """
    global _EXACT_SPARSE_EVALUATOR, _EXACT_SPARSE_SIGNATURE
    p = _resolve_existing_path(FORECASTS_FILE).resolve()
    s = p.stat()
    signature = (str(p), s.st_mtime_ns, s.st_size,
                 os.environ.get('MFRR_ACTIVATION_FRACTION'),
                 float(getattr(cfg_module,'SOC_INIT_PCT',50.0)),
                 float(getattr(cfg_module,'SOC_TARGET_PCT',50.0)))
    if _EXACT_SPARSE_EVALUATOR is None or _EXACT_SPARSE_SIGNATURE != signature:
        from fast_exact_oc import ExactSparseS2
        _EXACT_SPARSE_EVALUATOR = ExactSparseS2(sys.modules[__name__])
        _EXACT_SPARSE_SIGNATURE = signature
    return _EXACT_SPARSE_EVALUATOR.value(reservation_df)


def run_s2_forecast_value(reservation_df: pd.DataFrame) -> float:
    """Run the real Scenario 2 engine with forecast prices treated as actuals.

    This is the only valid opportunity-cost valuation for S3 capacity bids:
        S2 = DA + mFRR energy

    No actual market price is used here. A temporary actuals file is built from
    forecast columns only so the real rolling engine can clear and settle its
    own DA/mFRR logic in forecast space.
    """
    if os.environ.get('BALTIC_OC_METHOD','exact').lower() == 'exact_sparse':
        return run_s2_forecast_value_sparse(reservation_df)
    market_config, da_engine, _ = _load_s2_engine()
    df_fc, _timestamps = load_forecast_input_data(int(build_cfg()["HORIZON"]))

    with tempfile.TemporaryDirectory(prefix="s2_forecast_value_") as tmp_name:
        tmp = Path(tmp_name)

        forecast_file = tmp / "forecasts_rolling.csv"
        actuals_file = tmp / "actuals_forecast_as_actuals.csv"
        reservation_file = tmp / "reservation.csv"
        report_file = tmp / "s2_forecast_report.csv"
        summary_file = tmp / "s2_forecast_summary.csv"

        df_fc.to_csv(forecast_file, index=False, encoding="utf-8-sig")
        _forecast_as_actuals_file(df_fc, actuals_file)

        res = reservation_df.copy()
        res["timestamp"] = pd.to_datetime(res["timestamp"])
        res.to_csv(reservation_file, index=False, encoding="utf-8-sig")

        base_cfg = market_config.MarketConfig()
        supported = {f.name for f in fields(base_cfg)} if is_dataclass(base_cfg) else set()

        kwargs = {
            "forecast_file": forecast_file,
            "actuals_file": actuals_file,
            "reservation_file": reservation_file,
            "output_report": report_file,
            "output_summary": summary_file,
            "soc_init_pct": float(getattr(cfg_module, "SOC_INIT_PCT", 50.0)),
            "soc_target_pct": float(getattr(cfg_module, "SOC_TARGET_PCT", 50.0)),
            "rolling_soc_reference_pct": float(getattr(cfg_module, "SOC_TARGET_PCT", 50.0)),
            "rolling_recovery_target_soc_pct": float(getattr(cfg_module, "SOC_TARGET_PCT", 50.0)),
            "soc_terminal_sync": bool(getattr(cfg_module, "SOC_TERMINAL_SYNC", True)),
            "rolling_terminal_soc_sync": bool(getattr(cfg_module, "SOC_TERMINAL_SYNC", True)),
            "enable_mfrr_energy_market": True,
            "include_hypothetical_mfrr_in_da_stage": True,
            "reservation_file_required": True,
            "allow_reservation_row_order_fallback": False,
            "print_interval_table": False,
        }

        cfg = replace(base_cfg, **{k: v for k, v in kwargs.items() if k in supported})
        report = da_engine.run(cfg)

        if report is None:
            report = pd.read_csv(report_file)

        return _sum_first_existing_column(
            report,
            [
                "market_net_cashflow_eur_actual",
                "net_cashflow_eur_actual",
                "market_net_eur_actual",
            ],
        )


def run_da_mfrr_forecast_value(reservation_df: pd.DataFrame) -> float:
    """Compatibility wrapper: now runs full Scenario 2 forecast valuation."""
    return run_s2_forecast_value(reservation_df)


def calc_oc_curve_for_interval_direction(
    cfg: dict[str, Any],
    base_res: pd.DataFrame,
    base_s2_value: float,
    interval: int,
    direction: str,
    mw_steps: list[float],
) -> dict[str, Any]:
    dt = float(cfg["DT"])
    prev_loss = 0.0

    raw_bid_curve: list[float] = []
    marginal_oc_eur: list[float] = []
    total_loss_curve: list[float] = []
    test_value_curve: list[float] = []

    progress(
        f"  OC interval {interval + 1}/{int(cfg['HORIZON'])} "
        f"{direction.upper()}: {len(mw_steps)} step(s)"
    )

    for q in mw_steps:
        progress(f"    testing {direction} freeze q={q:.1f} MW ...")

        test_res = apply_capacity_freeze(
            base_res=base_res,
            interval=interval,
            direction=direction,
            q_mw=q,
            cfg=cfg,
        )

        test_value = run_da_mfrr_forecast_value(test_res)

        loss = max(0.0, base_s2_value - test_value)
        marginal = max(0.0, loss - prev_loss)
        prev_loss = loss

        bid_price = marginal / (STEP_MW * dt) if dt > EPS else 99999.0

        raw_bid_curve.append(float(bid_price))
        marginal_oc_eur.append(float(marginal))
        total_loss_curve.append(float(loss))
        test_value_curve.append(float(test_value))

    bid_curve: list[float] = []
    last = -1e30

    for p in raw_bid_curve:
        p2 = float(p) if last <= -1e20 else max(float(p), last + 0.01)
        bid_curve.append(round(p2, 6))
        last = p2

    return {
        "bid_curve": bid_curve,
        "raw_bid_curve": [round(x, 6) for x in raw_bid_curve],
        "marginal_oc_eur": [round(x, 6) for x in marginal_oc_eur],
        "total_loss_s2_curve_eur": [round(x, 6) for x in total_loss_curve],
        "test_s2_value_curve_eur": [round(x, 6) for x in test_value_curve],
    }


def build_all_interval_curves(
    price_data: dict[str, np.ndarray],
    base_res: pd.DataFrame,
    cfg: dict[str, Any],
    base_s2_value: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for t in range(int(cfg["HORIZON"])):
        progress(f"\n[{t + 1}/{int(cfg['HORIZON'])}] Building aFRR/mFRR capacity OC curves")

        up_available = available_capacity_mw(base_res, t, "up", cfg)
        down_available = available_capacity_mw(base_res, t, "down", cfg)

        up_steps = make_mw_steps(up_available)
        down_steps = make_mw_steps(down_available)

        up_curve = calc_oc_curve_for_interval_direction(
            cfg=cfg,
            base_res=base_res,
            base_s2_value=base_s2_value,
            interval=t,
            direction="up",
            mw_steps=up_steps,
        )

        down_curve = calc_oc_curve_for_interval_direction(
            cfg=cfg,
            base_res=base_res,
            base_s2_value=base_s2_value,
            interval=t,
            direction="down",
            mw_steps=down_steps,
        )

        up_market, up_fc_price = best_capacity_market_forecast(price_data, t, "up")
        down_market, down_fc_price = best_capacity_market_forecast(price_data, t, "down")

        rows.append(
            {
                "Interval": t,
                "Available_UP_MW": up_available,
                "Available_DOWN_MW": down_available,
                "UP_MW_Steps": up_steps,
                "DOWN_MW_Steps": down_steps,
                "UP_Best_Market_Forecast": up_market,
                "DOWN_Best_Market_Forecast": down_market,
                "UP_Forecast_Price_EUR_per_MW_h": up_fc_price,
                "DOWN_Forecast_Price_EUR_per_MW_h": down_fc_price,
                "UP_Bid_Curve": up_curve["bid_curve"],
                "DOWN_Bid_Curve": down_curve["bid_curve"],
                "UP_Raw_Bid_Curve": up_curve["raw_bid_curve"],
                "DOWN_Raw_Bid_Curve": down_curve["raw_bid_curve"],
                "UP_Marginal_OC_EUR": up_curve["marginal_oc_eur"],
                "DOWN_Marginal_OC_EUR": down_curve["marginal_oc_eur"],
                "UP_Total_Loss_S2_Curve_EUR": up_curve["total_loss_s2_curve_eur"],
                "DOWN_Total_Loss_S2_Curve_EUR": down_curve["total_loss_s2_curve_eur"],
                "UP_Test_S2_Value_Curve_EUR": up_curve["test_s2_value_curve_eur"],
                "DOWN_Test_S2_Value_Curve_EUR": down_curve["test_s2_value_curve_eur"],
            }
        )

    return rows


# ============================================================
# DP BID SCHEDULER - FORECAST ONLY
# ============================================================
def build_risk_params(cfg: dict[str, Any]) -> dict[str, Any]:
    b = cfg["BATTERY"]
    risk = cfg.get("RISK_MANAGEMENT", {})

    eta = float(b["efficiency"])
    p_max = float(b["p_max_mw"])
    dt = float(cfg["DT"])
    buffer_h = reserve_buffer_h("aFRR")

    max_up = float(
        risk.get(
            "AFRR_MFRR_UP_MAX_CUMULATIVE_EXPOSURE_MWH",
            risk.get(
                "UP_SOC_HEADROOM_MWH",
                (float(b["soc_init_pct"]) - float(b["soc_min_pct"]))
                / 100.0
                * float(b["e_max_mwh"]),
            ),
        )
    )

    max_down = float(
        risk.get(
            "AFRR_MFRR_DOWN_MAX_CUMULATIVE_EXPOSURE_MWH",
            risk.get(
                "DOWN_SOC_HEADROOM_MWH",
                (float(b["soc_max_pct"]) - float(b["soc_init_pct"]))
                / 100.0
                * float(b["e_max_mwh"]),
            ),
        )
    )

    return {
        "enable_risk": bool(
            risk.get(
                "ENABLE_AFRR_MFRR_RISK_LIMIT",
                risk.get("ENABLE_AFRR_MFRR_CAPACITY_RISK_LIMIT", True),
            )
        ),
        "max_up_exposure_mwh": max(0.0, max_up),
        "max_down_exposure_mwh": max(0.0, max_down),
        "up_exposure_per_mw_interval_mwh": buffer_h / max(eta, EPS),
        "down_exposure_per_mw_interval_mwh": buffer_h * eta,
        "charge_recovery_per_interval_mwh": p_max * dt * eta,
        "discharge_recovery_per_interval_mwh": p_max * dt / max(eta, EPS),
    }


def cumulative_step_value(
    bid_curve: list[float],
    fc_price: float,
    steps: int,
    dt: float,
) -> float:
    value = 0.0

    for k in range(steps):
        value += (float(fc_price) - float(bid_curve[k])) * STEP_MW * dt

    return float(value)


def build_interval_dp_actions(
    row: dict[str, Any],
    base_res: pd.DataFrame,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build mutually-exclusive UP/DOWN capacity actions for one interval.

    aFRR/mFRR capacity bids are allowed in only one direction per interval:
        - no bid
        - UP only
        - DOWN only

    The selected market inside each direction is still the better forecast-price
    market between aFRR and mFRR.
    """
    t = int(row["Interval"])
    dt = float(cfg["DT"])

    max_ku = len(row["UP_Bid_Curve"])
    max_kd = len(row["DOWN_Bid_Curve"])

    actions: list[dict[str, Any]] = [
        {
            "ku": 0,
            "kd": 0,
            "up_mw": 0.0,
            "down_mw": 0.0,
            "value_eur": 0.0,
        }
    ]

    for ku in range(1, max_ku + 1):
        up_mw = ku * STEP_MW

        if not joint_capacity_freeze_feasible(base_res, t, up_mw, 0.0, cfg):
            continue

        value = cumulative_step_value(
            row["UP_Bid_Curve"],
            row["UP_Forecast_Price_EUR_per_MW_h"],
            ku,
            dt,
        )

        if ALLOW_NEGATIVE_FORECAST_MARGIN or value > EPS:
            actions.append(
                {
                    "ku": ku,
                    "kd": 0,
                    "up_mw": float(up_mw),
                    "down_mw": 0.0,
                    "value_eur": float(value),
                }
            )

    for kd in range(1, max_kd + 1):
        down_mw = kd * STEP_MW

        if not joint_capacity_freeze_feasible(base_res, t, 0.0, down_mw, cfg):
            continue

        value = cumulative_step_value(
            row["DOWN_Bid_Curve"],
            row["DOWN_Forecast_Price_EUR_per_MW_h"],
            kd,
            dt,
        )

        if ALLOW_NEGATIVE_FORECAST_MARGIN or value > EPS:
            actions.append(
                {
                    "ku": 0,
                    "kd": kd,
                    "up_mw": 0.0,
                    "down_mw": float(down_mw),
                    "value_eur": float(value),
                }
            )

    return actions


def optimize_bid_profile_by_dp(
    interval_rows: list[dict[str, Any]],
    base_res: pd.DataFrame,
    risk_params: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Forecast-only DP for submitted aFRR/mFRR capacity bids.

    Uses only:
        - forecast capacity prices
        - DA+mFRR opportunity-cost bid curves
        - risk/recovery limits

    No actual clearing prices are used here.
    """
    horizon = int(cfg["HORIZON"])

    interval_actions = [
        build_interval_dp_actions(row, base_res, cfg)
        for row in interval_rows
    ]

    if not bool(risk_params["enable_risk"]):
        up_bid = np.zeros(horizon)
        down_bid = np.zeros(horizon)
        objective = 0.0

        for t, actions in enumerate(interval_actions):
            best = max(actions, key=lambda a: float(a["value_eur"]))
            up_bid[t] = float(best["up_mw"])
            down_bid[t] = float(best["down_mw"])
            objective += float(best["value_eur"])

        return {
            "status": "Optimal",
            "bid_up_mw_profile": up_bid,
            "bid_down_mw_profile": down_bid,
            "up_exposure_profile_mwh": np.zeros(horizon),
            "down_exposure_profile_mwh": np.zeros(horizon),
            "charge_recovery_profile_mwh": np.zeros(horizon),
            "discharge_recovery_profile_mwh": np.zeros(horizon),
            "objective_value_eur": float(objective),
            "interval_actions": interval_actions,
        }

    scale = int(getattr(cfg_module, "AFRR_MFRR_DP_RISK_SCALE_PER_MWH", 1000))

    max_up_e = int(round(float(risk_params["max_up_exposure_mwh"]) * scale))
    max_down_e = int(round(float(risk_params["max_down_exposure_mwh"]) * scale))

    up_exp_unit = int(
        round(float(risk_params["up_exposure_per_mw_interval_mwh"]) * scale * STEP_MW)
    )

    down_exp_unit = int(
        round(float(risk_params["down_exposure_per_mw_interval_mwh"]) * scale * STEP_MW)
    )

    charge_rec_unit = int(
        round(float(risk_params["charge_recovery_per_interval_mwh"]) * scale)
    )

    discharge_rec_unit = int(
        round(float(risk_params["discharge_recovery_per_interval_mwh"]) * scale)
    )

    if max_up_e <= 0 or max_down_e <= 0:
        raise ValueError("Invalid aFRR/mFRR exposure limits.")

    if up_exp_unit <= 0 or down_exp_unit <= 0:
        raise ValueError("Invalid aFRR/mFRR exposure unit.")

    # Sparse forward DP:
    # dp[(eu, ed)] = best value after processed intervals so far.
    dp: dict[tuple[int, int], float] = {(0, 0): 0.0}
    back: list[dict[tuple[int, int], tuple[tuple[int, int], int]]] = []

    for t in range(horizon):
        actions = interval_actions[t]
        next_dp: dict[tuple[int, int], float] = {}
        next_back: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}

        for state, base_value in dp.items():
            eu, ed = state

            for idx, action in enumerate(actions):
                ku = int(action["ku"])
                kd = int(action["kd"])

                if ku == 0 and kd == 0:
                    eu_next = max(0, eu - charge_rec_unit)
                    ed_next = max(0, ed - discharge_rec_unit)
                else:
                    eu_next = eu + ku * up_exp_unit
                    ed_next = ed + kd * down_exp_unit

                if eu_next > max_up_e or ed_next > max_down_e:
                    continue

                state_next = (eu_next, ed_next)
                value_next = float(base_value) + float(action["value_eur"])

                if state_next not in next_dp or value_next > next_dp[state_next] + EPS:
                    next_dp[state_next] = value_next
                    next_back[state_next] = (state, idx)

        if not next_dp:
            raise RuntimeError(
                f"aFRR/mFRR bid DP infeasible at interval {t}. "
                "Check FCR reservation, exposure limits, recovery settings, and bid steps."
            )

        dp = next_dp
        back.append(next_back)

    terminal_state, best_value = max(dp.items(), key=lambda kv: kv[1])

    chosen_action_indices = [0] * horizon
    state = terminal_state

    for t in range(horizon - 1, -1, -1):
        prev_state, idx = back[t][state]
        chosen_action_indices[t] = idx
        state = prev_state

    up_bid = np.zeros(horizon)
    down_bid = np.zeros(horizon)
    up_exposure = np.zeros(horizon)
    down_exposure = np.zeros(horizon)
    charge_recovery = np.zeros(horizon)
    discharge_recovery = np.zeros(horizon)

    eu = 0
    ed = 0

    for t, idx in enumerate(chosen_action_indices):
        action = interval_actions[t][idx]

        ku = int(action["ku"])
        kd = int(action["kd"])

        up_bid[t] = ku * STEP_MW
        down_bid[t] = kd * STEP_MW

        if ku == 0 and kd == 0:
            old_eu = eu
            old_ed = ed

            eu = max(0, eu - charge_rec_unit)
            ed = max(0, ed - discharge_rec_unit)

            charge_recovery[t] = float(old_eu - eu) / scale
            discharge_recovery[t] = float(old_ed - ed) / scale
        else:
            eu += ku * up_exp_unit
            ed += kd * down_exp_unit

        up_exposure[t] = float(eu) / scale
        down_exposure[t] = float(ed) / scale

    return {
        "status": "Optimal",
        "bid_up_mw_profile": up_bid,
        "bid_down_mw_profile": down_bid,
        "up_exposure_profile_mwh": up_exposure,
        "down_exposure_profile_mwh": down_exposure,
        "charge_recovery_profile_mwh": charge_recovery,
        "discharge_recovery_profile_mwh": discharge_recovery,
        "objective_value_eur": float(best_value),
        "interval_actions": interval_actions,
    }

def recalculate_exposure_recovery_from_profiles(
    bid_up_profile: np.ndarray,
    bid_down_profile: np.ndarray,
    risk_params: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Recompute exposure/recovery profiles from FINAL (post-repair) bids.

    Mirrors the DP transition exactly so reported numbers match
    the profile that is actually submitted.
    """
    horizon = int(cfg["HORIZON"])
    scale = int(getattr(cfg_module, "AFRR_MFRR_DP_RISK_SCALE_PER_MWH", 1000))

    up_exp_unit = int(round(float(risk_params["up_exposure_per_mw_interval_mwh"]) * scale * STEP_MW))
    down_exp_unit = int(round(float(risk_params["down_exposure_per_mw_interval_mwh"]) * scale * STEP_MW))
    charge_rec_unit = int(round(float(risk_params["charge_recovery_per_interval_mwh"]) * scale))
    discharge_rec_unit = int(round(float(risk_params["discharge_recovery_per_interval_mwh"]) * scale))

    up_exposure = np.zeros(horizon)
    down_exposure = np.zeros(horizon)
    charge_recovery = np.zeros(horizon)
    discharge_recovery = np.zeros(horizon)

    eu = 0
    ed = 0

    for t in range(horizon):
        ku = int(math.floor(float(bid_up_profile[t]) / STEP_MW + EPS))
        kd = int(math.floor(float(bid_down_profile[t]) / STEP_MW + EPS))

        if ku == 0 and kd == 0:
            old_eu, old_ed = eu, ed
            eu = max(0, eu - charge_rec_unit)
            ed = max(0, ed - discharge_rec_unit)
            charge_recovery[t] = float(old_eu - eu) / scale
            discharge_recovery[t] = float(old_ed - ed) / scale
        else:
            eu += ku * up_exp_unit
            ed += kd * down_exp_unit

        up_exposure[t] = float(eu) / scale
        down_exposure[t] = float(ed) / scale

    return {
        "up_exposure_profile_mwh": up_exposure,
        "down_exposure_profile_mwh": down_exposure,
        "charge_recovery_profile_mwh": charge_recovery,
        "discharge_recovery_profile_mwh": discharge_recovery,
    }

# ============================================================
# CLEARING AND FINAL RESERVATION LOCKS
# ============================================================
def cleared_steps_by_price(
    bid_curve: list[float],
    clearing_price: float,
) -> int:
    if np.isnan(float(clearing_price)):
        return 0

    accepted = 0

    for bid in bid_curve:
        if float(bid) <= float(clearing_price) + EPS:
            accepted += 1
        else:
            break

    return accepted


def init_audit_columns(out: pd.DataFrame) -> pd.DataFrame:
    # S3_CLEARING_FAST_SAFE_V1: skip redundant float64 conversions
    defaults = {
        "afrr_up_accepted_mw": 0.0,
        "afrr_down_accepted_mw": 0.0,
        "mfrr_up_accepted_mw": 0.0,
        "mfrr_down_accepted_mw": 0.0,
        "afrr_mfrr_delivery_up_locked_mw": 0.0,
        "afrr_mfrr_delivery_down_locked_mw": 0.0,
        "afrr_mfrr_recovery_charge_locked_mw": 0.0,
        "afrr_mfrr_recovery_discharge_locked_mw": 0.0,
        "afrr_mfrr_uncovered_charge_recovery_mwh": 0.0,
        "afrr_mfrr_uncovered_discharge_recovery_mwh": 0.0,
    }

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

        if out[col].dtype != np.dtype("float64") or out[col].hasnans:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default).astype(float)

    for col in ["p_charge_max_mw", "p_discharge_max_mw", "soc_min_pct", "soc_max_pct"]:
        if out[col].dtype != np.dtype("float64") or out[col].hasnans:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)

    return out


def apply_delivery_lock_with_audit(
    base_res: pd.DataFrame,
    interval: int,
    market: str,
    direction: str,
    q_mw: float,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    out = apply_capacity_freeze(
        base_res=base_res,
        interval=interval,
        direction=direction,
        q_mw=q_mw,
        cfg=cfg,
    )

    out = init_audit_columns(out)

    q = float(q_mw)

    if direction == "up":
        out.at[interval, "afrr_mfrr_delivery_up_locked_mw"] += q

        if market == "aFRR":
            out.at[interval, "afrr_up_accepted_mw"] += q
        elif market == "mFRR":
            out.at[interval, "mfrr_up_accepted_mw"] += q

    else:
        out.at[interval, "afrr_mfrr_delivery_down_locked_mw"] += q

        if market == "aFRR":
            out.at[interval, "afrr_down_accepted_mw"] += q
        elif market == "mFRR":
            out.at[interval, "mfrr_down_accepted_mw"] += q

    return out


def apply_recovery_lock_after_acceptance(
    base_res: pd.DataFrame,
    source_interval: int,
    accepted_up_mw: float,
    accepted_down_mw: float,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Lock future recovery capacity after accepted capacity.

    UP activation discharges battery:
        future charge capacity is reserved.

    DOWN activation charges battery:
        future discharge capacity is reserved.
    """
    out = init_audit_columns(base_res.copy())

    eta = float(cfg["BATTERY"]["efficiency"])
    dt = float(cfg["DT"])
    buffer_h = reserve_buffer_h("aFRR")

    remaining_charge_recovery_mwh = max(
        0.0,
        float(accepted_up_mw) * buffer_h / max(eta, EPS),
    )

    remaining_discharge_recovery_mwh = max(
        0.0,
        float(accepted_down_mw) * buffer_h * eta,
    )

    if remaining_charge_recovery_mwh <= EPS and remaining_discharge_recovery_mwh <= EPS:
        return out

    max_intervals = RECOVERY_MAX_INTERVALS

    if max_intervals is None:
        max_intervals = len(out) - source_interval - RECOVERY_START_OFFSET_INTERVALS

    max_intervals = max(0, int(max_intervals))

    first = source_interval + RECOVERY_START_OFFSET_INTERVALS
    last = min(len(out), first + max_intervals)

    for j in range(first, last):
        if remaining_charge_recovery_mwh <= EPS and remaining_discharge_recovery_mwh <= EPS:
            break

        if remaining_charge_recovery_mwh > EPS:
            available_charge = max(0.0, float(out.at[j, "p_charge_max_mw"]))

            lock_charge_mw = min(
                available_charge,
                remaining_charge_recovery_mwh / max(dt * eta, EPS),
            )

            if lock_charge_mw > EPS:
                out.at[j, "p_charge_max_mw"] = max(0.0, available_charge - lock_charge_mw)
                out.at[j, "afrr_mfrr_recovery_charge_locked_mw"] += lock_charge_mw

                remaining_charge_recovery_mwh = max(
                    0.0,
                    remaining_charge_recovery_mwh - lock_charge_mw * dt * eta,
                )

        if remaining_discharge_recovery_mwh > EPS:
            available_discharge = max(0.0, float(out.at[j, "p_discharge_max_mw"]))

            lock_discharge_mw = min(
                available_discharge,
                remaining_discharge_recovery_mwh * eta / max(dt, EPS),
            )

            if lock_discharge_mw > EPS:
                out.at[j, "p_discharge_max_mw"] = max(0.0, available_discharge - lock_discharge_mw)
                out.at[j, "afrr_mfrr_recovery_discharge_locked_mw"] += lock_discharge_mw

                remaining_discharge_recovery_mwh = max(
                    0.0,
                    remaining_discharge_recovery_mwh - lock_discharge_mw * dt / max(eta, EPS),
                )

    out.at[source_interval, "afrr_mfrr_uncovered_charge_recovery_mwh"] += (
        remaining_charge_recovery_mwh
    )

    out.at[source_interval, "afrr_mfrr_uncovered_discharge_recovery_mwh"] += (
        remaining_discharge_recovery_mwh
    )

    return out


def build_final_reservation_profile(
    base_res: pd.DataFrame,
    accepted: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    out = init_audit_columns(base_res.copy())

    for _, r in accepted.iterrows():
        interval = int(r["Interval"])

        if float(r["Accepted_UP_MW"]) > EPS:
            out = apply_delivery_lock_with_audit(
                base_res=out,
                interval=interval,
                market=str(r["UP_Market"]),
                direction="up",
                q_mw=float(r["Accepted_UP_MW"]),
                cfg=cfg,
            )

        if float(r["Accepted_DOWN_MW"]) > EPS:
            out = apply_delivery_lock_with_audit(
                base_res=out,
                interval=interval,
                market=str(r["DOWN_Market"]),
                direction="down",
                q_mw=float(r["Accepted_DOWN_MW"]),
                cfg=cfg,
            )

    for _, r in accepted.iterrows():
        interval = int(r["Interval"])

        up_q = float(r["Accepted_UP_MW"])
        down_q = float(r["Accepted_DOWN_MW"])

        if up_q > EPS or down_q > EPS:
            out = apply_recovery_lock_after_acceptance(
                base_res=out,
                source_interval=interval,
                accepted_up_mw=up_q,
                accepted_down_mw=down_q,
                cfg=cfg,
            )

    return out


# ============================================================
# PRE-BID WORST-CASE RECOVERY REPAIR
# ============================================================
def make_assumed_accepted_frame_from_profiles(
    interval_rows: list[dict[str, Any]],
    timestamps: pd.Series,
    bid_up_profile: np.ndarray,
    bid_down_profile: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row in interval_rows:
        t = int(row["Interval"])

        rows.append(
            {
                "Interval": t,
                "timestamp": timestamps.iloc[t],
                "UP_Market": str(row["UP_Best_Market_Forecast"]),
                "DOWN_Market": str(row["DOWN_Best_Market_Forecast"]),
                "Accepted_UP_MW": round(float(bid_up_profile[t]), 6),
                "Accepted_DOWN_MW": round(float(bid_down_profile[t]), 6),
            }
        )

    return pd.DataFrame(rows)


def submitted_worst_case_uncovered_recovery(
    base_res: pd.DataFrame,
    interval_rows: list[dict[str, Any]],
    timestamps: pd.Series,
    bid_up_profile: np.ndarray,
    bid_down_profile: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[float, float, pd.DataFrame]:
    assumed = make_assumed_accepted_frame_from_profiles(
        interval_rows=interval_rows,
        timestamps=timestamps,
        bid_up_profile=bid_up_profile,
        bid_down_profile=bid_down_profile,
    )

    test_res = build_final_reservation_profile(
        base_res=base_res,
        accepted=assumed,
        cfg=cfg,
    )

    uncovered_charge = float(test_res["afrr_mfrr_uncovered_charge_recovery_mwh"].sum())
    uncovered_discharge = float(test_res["afrr_mfrr_uncovered_discharge_recovery_mwh"].sum())

    return uncovered_charge, uncovered_discharge, test_res


def last_step_margin_eur(
    row: dict[str, Any],
    direction: str,
    current_steps: int,
    dt: float,
) -> float:
    if current_steps <= 0:
        return 1e30

    if direction == "up":
        curve = list(row["UP_Bid_Curve"])
        fc = float(row["UP_Forecast_Price_EUR_per_MW_h"])
    else:
        curve = list(row["DOWN_Bid_Curve"])
        fc = float(row["DOWN_Forecast_Price_EUR_per_MW_h"])

    if current_steps > len(curve):
        return 1e30

    return float((fc - float(curve[current_steps - 1])) * STEP_MW * dt)


def repair_bid_profile_for_submitted_worst_case_recovery(
    base_res: pd.DataFrame,
    interval_rows: list[dict[str, Any]],
    timestamps: pd.Series,
    bid_up_profile: np.ndarray,
    bid_down_profile: np.ndarray,
    cfg: dict[str, Any],
    print_console: bool = True,
) -> dict[str, Any]:
    """Trim submitted bids until worst-case submitted clearing has full recovery.

    Assumption before bidding:
        all submitted MW may clear
        all cleared MW may be activated
        delivery + recovery must be fully reservable
    """
    dt = float(cfg["DT"])

    up = np.array(bid_up_profile, dtype=float).copy()
    down = np.array(bid_down_profile, dtype=float).copy()

    removed_steps: list[dict[str, Any]] = []

    max_iter = int(round((np.sum(up) + np.sum(down)) / STEP_MW + 10))

    for iteration in range(max_iter + 1):
        uc, ud, test_res = submitted_worst_case_uncovered_recovery(
            base_res=base_res,
            interval_rows=interval_rows,
            timestamps=timestamps,
            bid_up_profile=up,
            bid_down_profile=down,
            cfg=cfg,
        )

        if uc <= EPS and ud <= EPS:
            # if print_console:
            #     print(
            #         "Submitted worst-case recovery check: OK "
            #         f"(removed {len(removed_steps)} MW-step(s) before bidding)",
            #         flush=True,
            #     )

            return {
                "bid_up_mw_profile": up,
                "bid_down_mw_profile": down,
                "submitted_worst_case_reservation": test_res,
                "submitted_worst_case_uncovered_charge_mwh": uc,
                "submitted_worst_case_uncovered_discharge_mwh": ud,
                "removed_steps": removed_steps,
            }

        candidates: list[tuple[float, int, int, str]] = []

        for row in interval_rows:
            t = int(row["Interval"])

            ku = int(math.floor(float(up[t]) / STEP_MW + EPS))
            kd = int(math.floor(float(down[t]) / STEP_MW + EPS))

            if ku > 0:
                margin = last_step_margin_eur(row, "up", ku, dt)
                candidates.append((margin, -t, t, "up"))

            if kd > 0:
                margin = last_step_margin_eur(row, "down", kd, dt)
                candidates.append((margin, -t, t, "down"))

        if not candidates:
            raise RuntimeError(
                "No submitted bids remain, but recovery is still uncovered: "
                f"charge={uc:.6f} MWh, discharge={ud:.6f} MWh"
            )

        candidates.sort(key=lambda x: (x[0], x[1]))

        margin, _, t_remove, direction = candidates[0]

        if direction == "up":
            up[t_remove] = max(0.0, float(up[t_remove]) - STEP_MW)
        else:
            down[t_remove] = max(0.0, float(down[t_remove]) - STEP_MW)

        removed_steps.append(
            {
                "iteration": iteration + 1,
                "interval": int(t_remove),
                "direction": direction,
                "removed_mw": float(STEP_MW),
                "forecast_margin_eur": float(margin),
                "uncovered_charge_before_mwh": float(uc),
                "uncovered_discharge_before_mwh": float(ud),
            }
        )

        if print_console:
            print(
                "Pre-bid recovery repair: "
                f"removed {STEP_MW:.1f} MW {direction.upper()} at interval {t_remove}; "
                f"uncovered before removal: charge={uc:.4f} MWh, discharge={ud:.4f} MWh",
                flush=True,
            )

    raise RuntimeError("Submitted worst-case recovery repair did not converge.")


# ============================================================
# RESULT BUILDING
# ============================================================
def build_results_table(
    interval_rows: list[dict[str, Any]],
    timestamps: pd.Series,
    bid_up_profile: np.ndarray,
    bid_down_profile: np.ndarray,
    actual_price_data: dict[str, np.ndarray] | None,
    cfg: dict[str, Any],
    base_s2_value: float,
    opt: dict[str, Any],
    repair: dict[str, Any],
    print_console: bool = True,
) -> tuple[pd.DataFrame, float]:
    dt = float(cfg["DT"])
    result_rows: list[dict[str, Any]] = []
    total_capacity_revenue = 0.0

    if print_console:
        print("\n" + "=" * 180)
        print(
            "T | UP Bid/Acc/Mkt | DOWN Bid/Acc/Mkt | "
            "UP fc/clear | DOWN fc/clear | UP OC Curve | DOWN OC Curve"
        )
        print("-" * 180)

    for row in interval_rows:
        t = int(row["Interval"])

        up_market = str(row["UP_Best_Market_Forecast"])
        down_market = str(row["DOWN_Best_Market_Forecast"])

        submitted_up_mw = float(bid_up_profile[t])
        submitted_down_mw = float(bid_down_profile[t])

        up_steps = int(math.floor(submitted_up_mw / STEP_MW + EPS))
        down_steps = int(math.floor(submitted_down_mw / STEP_MW + EPS))

        up_curve_submitted = list(row["UP_Bid_Curve"][:up_steps])
        down_curve_submitted = list(row["DOWN_Bid_Curve"][:down_steps])

        up_clear = actual_clearing_price_for_market(
            actual_price_data,
            t,
            up_market,
            "up",
        )

        down_clear = actual_clearing_price_for_market(
            actual_price_data,
            t,
            down_market,
            "down",
        )

        accepted_up_steps = min(cleared_steps_by_price(up_curve_submitted, up_clear), up_steps)
        accepted_down_steps = min(cleared_steps_by_price(down_curve_submitted, down_clear), down_steps)

        accepted_up_mw = accepted_up_steps * STEP_MW
        accepted_down_mw = accepted_down_steps * STEP_MW

        up_revenue = 0.0 if np.isnan(up_clear) else accepted_up_mw * up_clear * dt
        down_revenue = 0.0 if np.isnan(down_clear) else accepted_down_mw * down_clear * dt

        total_capacity_revenue += up_revenue + down_revenue

        if print_console:
            up_clear_txt = "NA" if np.isnan(up_clear) else f"{up_clear:.2f}"
            down_clear_txt = "NA" if np.isnan(down_clear) else f"{down_clear:.2f}"

            print(
                f"{t:<2} | "
                f"{submitted_up_mw:.1f}/{accepted_up_mw:.1f}/{up_market:<4} | "
                f"{submitted_down_mw:.1f}/{accepted_down_mw:.1f}/{down_market:<4} | "
                f"{float(row['UP_Forecast_Price_EUR_per_MW_h']):.2f}/{up_clear_txt} | "
                f"{float(row['DOWN_Forecast_Price_EUR_per_MW_h']):.2f}/{down_clear_txt} | "
                f"{fmt_curve(up_curve_submitted):<25} | "
                f"{fmt_curve(down_curve_submitted):<25}",
                flush=True,
            )

        result_rows.append(
            {
                "Interval": t,
                "timestamp": timestamps.iloc[t],
                "Available_UP_MW": round(float(row["Available_UP_MW"]), 6),
                "Available_DOWN_MW": round(float(row["Available_DOWN_MW"]), 6),
                "UP_Market": up_market,
                "DOWN_Market": down_market,
                "Bid_UP_MW": round(submitted_up_mw, 6),
                "Bid_DOWN_MW": round(submitted_down_mw, 6),
                "Accepted_UP_MW": round(accepted_up_mw, 6),
                "Accepted_DOWN_MW": round(accepted_down_mw, 6),
                "UP_Forecast_Price_EUR_per_MW_h": round(float(row["UP_Forecast_Price_EUR_per_MW_h"]), 6),
                "DOWN_Forecast_Price_EUR_per_MW_h": round(float(row["DOWN_Forecast_Price_EUR_per_MW_h"]), 6),
                "UP_Actual_Clearing_Price_EUR_per_MW_h": None if np.isnan(up_clear) else round(up_clear, 6),
                "DOWN_Actual_Clearing_Price_EUR_per_MW_h": None if np.isnan(down_clear) else round(down_clear, 6),
                "Submitted_UP_OC_Bid_Curve_EUR_per_MW_h": json.dumps(up_curve_submitted),
                "Submitted_DOWN_OC_Bid_Curve_EUR_per_MW_h": json.dumps(down_curve_submitted),
                "Full_UP_OC_Bid_Curve_EUR_per_MW_h": json.dumps(row["UP_Bid_Curve"]),
                "Full_DOWN_OC_Bid_Curve_EUR_per_MW_h": json.dumps(row["DOWN_Bid_Curve"]),
                "Raw_UP_OC_Bid_Curve_EUR_per_MW_h": json.dumps(row["UP_Raw_Bid_Curve"]),
                "Raw_DOWN_OC_Bid_Curve_EUR_per_MW_h": json.dumps(row["DOWN_Raw_Bid_Curve"]),
                "UP_Marginal_OC_EUR_per_Interval": json.dumps(row["UP_Marginal_OC_EUR"]),
                "DOWN_Marginal_OC_EUR_per_Interval": json.dumps(row["DOWN_Marginal_OC_EUR"]),
                "UP_Total_Loss_S2_Curve_EUR": json.dumps(row["UP_Total_Loss_S2_Curve_EUR"]),
                "DOWN_Total_Loss_S2_Curve_EUR": json.dumps(row["DOWN_Total_Loss_S2_Curve_EUR"]),
                "UP_Test_S2_Value_Curve_EUR": json.dumps(row["UP_Test_S2_Value_Curve_EUR"]),
                "DOWN_Test_S2_Value_Curve_EUR": json.dumps(row["DOWN_Test_S2_Value_Curve_EUR"]),
                "UP_Capacity_Revenue_EUR": round(up_revenue, 6),
                "DOWN_Capacity_Revenue_EUR": round(down_revenue, 6),
                "Capacity_Revenue_EUR": round(up_revenue + down_revenue, 6),
                "DP_UP_Exposure_After_MWh": round(float(opt["up_exposure_profile_mwh"][t]), 6),
                "DP_DOWN_Exposure_After_MWh": round(float(opt["down_exposure_profile_mwh"][t]), 6),
                "DP_Charge_Recovery_MWh": round(float(opt["charge_recovery_profile_mwh"][t]), 6),
                "DP_Discharge_Recovery_MWh": round(float(opt["discharge_recovery_profile_mwh"][t]), 6),
                "Base_S2_Forecast_Value_EUR": round(base_s2_value, 6),
                "Submitted_Worst_Case_Recovery_Repair_Removed_Steps": len(repair["removed_steps"]),
                "Submitted_Worst_Case_Uncovered_Charge_Recovery_MWh": round(
                    float(repair["submitted_worst_case_uncovered_charge_mwh"]),
                    6,
                ),
                "Submitted_Worst_Case_Uncovered_Discharge_Recovery_MWh": round(
                    float(repair["submitted_worst_case_uncovered_discharge_mwh"]),
                    6,
                ),
            }
        )

    return pd.DataFrame(result_rows), float(total_capacity_revenue)



# ============================================================
# S3 FORECAST PORTFOLIO PRE-CHECK AGAINST S2
# ============================================================
def build_forecast_accepted_results_from_bid_profile(
    interval_rows: list[dict[str, Any]],
    timestamps: pd.Series,
    bid_up_profile: np.ndarray,
    bid_down_profile: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row in interval_rows:
        t = int(row["Interval"])

        rows.append(
            {
                "timestamp": pd.to_datetime(timestamps.iloc[t]),
                "Interval": t,
                "UP_Market": str(row["UP_Best_Market_Forecast"]),
                "DOWN_Market": str(row["DOWN_Best_Market_Forecast"]),
                "Accepted_UP_MW": float(bid_up_profile[t]),
                "Accepted_DOWN_MW": float(bid_down_profile[t]),
            }
        )

    return pd.DataFrame(rows)


def forecast_capacity_revenue_from_bid_profile(
    interval_rows: list[dict[str, Any]],
    bid_up_profile: np.ndarray,
    bid_down_profile: np.ndarray,
    cfg: dict[str, Any],
) -> float:
    dt = float(cfg["DT"])
    total = 0.0

    for row in interval_rows:
        t = int(row["Interval"])

        total += (
            float(bid_up_profile[t])
            * max(0.0, float(row["UP_Forecast_Price_EUR_per_MW_h"]))
            * dt
        )

        total += (
            float(bid_down_profile[t])
            * max(0.0, float(row["DOWN_Forecast_Price_EUR_per_MW_h"]))
            * dt
        )

    return float(total)


def run_s3_forecast_portfolio_precheck(
    base_res: pd.DataFrame,
    interval_rows: list[dict[str, Any]],
    timestamps: pd.Series,
    bid_up_profile: np.ndarray,
    bid_down_profile: np.ndarray,
    base_s2_value: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    forecast_capacity_revenue = forecast_capacity_revenue_from_bid_profile(
        interval_rows=interval_rows,
        bid_up_profile=bid_up_profile,
        bid_down_profile=bid_down_profile,
        cfg=cfg,
    )

    forecast_accepted = build_forecast_accepted_results_from_bid_profile(
        interval_rows=interval_rows,
        timestamps=timestamps,
        bid_up_profile=bid_up_profile,
        bid_down_profile=bid_down_profile,
    )

    reservation_after_forecast_capacity = build_final_reservation_profile(
        base_res=base_res,
        accepted=forecast_accepted,
        cfg=cfg,
    )

    s2_after_capacity = run_s2_forecast_value(reservation_after_forecast_capacity)
    portfolio_value = forecast_capacity_revenue + s2_after_capacity
    delta = portfolio_value - float(base_s2_value)
    passed = delta >= -EPS

    return {
        "passed": bool(passed),
        "base_s2_forecast_value_eur": float(base_s2_value),
        "forecast_capacity_revenue_before_precheck_eur": float(forecast_capacity_revenue),
        "s2_forecast_value_after_forecast_capacity_eur": float(s2_after_capacity),
        "s3_forecast_portfolio_value_before_precheck_eur": float(portfolio_value),
        "s3_forecast_portfolio_delta_before_precheck_eur": float(delta),
        "reservation_after_forecast_capacity": reservation_after_forecast_capacity,
    }


def zero_capacity_bid_profiles(horizon: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(horizon, dtype=float), np.zeros(horizon, dtype=float)


# ============================================================
# RESERVE CAPACITY COMPETITION RUNNER
# New logic:
#   One symmetric MW block competes between FCR, aFRR(up+down), mFRR(up+down).
#   FCR clears first. Remaining capacity is offered to aFRR/mFRR packages.
#   aFRR/mFRR clearing is direction-level OR: UP, DOWN, or both can be accepted.
# ============================================================
SCRIPT_VERSION = "RESERVE_CAPACITY_COMPETITION_ALWAYS_BID_OC_FCR_FIRST_02"

RESERVE_COMPETITION_RESULTS_FILE = RESULTS_FILE
RESERVE_COMPETITION_RESERVATIONS_FILE = RESERVATIONS_OUTPUT_FILE
RESERVE_COMPETITION_SUBMITTED_WORST_CASE_FILE = SUBMITTED_WORST_CASE_RESERVATIONS_FILE


def read_fcr_forecast_price(df_fc: pd.DataFrame, horizon: int) -> np.ndarray:
    return read_first_available_col(
        df_fc,
        [
            "fcr_price_forecast",
            "fcr_capacity_price_forecast",
            "fcr_clearing_price_forecast",
            "fcr_forecast_price",
        ],
        horizon,
        default=np.nan,
    )


def read_fcr_actual_price(df_act: pd.DataFrame, horizon: int) -> np.ndarray:
    return read_first_available_col(
        df_act,
        [
            "fcr_clearing_price",
            "fcr_capacity_price",
            "fcr_price",
            "fcr_actual_price",
        ],
        horizon,
        default=np.nan,
    )


def get_forecast_reserve_price_data(df_fc: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    data = get_forecast_capacity_price_data(df_fc, horizon)
    data["fc_fcr"] = read_fcr_forecast_price(df_fc, horizon)
    return data


def load_actual_reserve_price_data_if_available(
    timestamps: pd.Series,
    horizon: int,
) -> dict[str, np.ndarray] | None:
    ac_path = _resolve_existing_path(ACTUALS_FILE)
    if not ac_path.exists():
        return None

    df_act = pd.read_csv(ac_path)
    if "timestamp" in df_act.columns:
        df_act["timestamp"] = pd.to_datetime(df_act["timestamp"])
        df_act = (
            pd.DataFrame({"timestamp": pd.to_datetime(timestamps)})
            .merge(df_act, on="timestamp", how="left")
            .reset_index(drop=True)
        )
    else:
        df_act = df_act.iloc[:horizon].copy().reset_index(drop=True)

    data = load_actual_capacity_price_data_if_available(timestamps, horizon) or {}
    data["act_fcr"] = read_fcr_actual_price(df_act, horizon)
    return data


def init_reserve_competition_audit_columns(out: pd.DataFrame) -> pd.DataFrame:
    out = init_audit_columns(out)
    defaults = {
        "fcr_accepted_mw": 0.0,
        "fcr_delivery_up_locked_mw": 0.0,
        "fcr_delivery_down_locked_mw": 0.0,
        "fcr_recovery_charge_locked_mw": 0.0,
        "fcr_recovery_discharge_locked_mw": 0.0,
        "reserve_competition_symmetric_delivery_up_locked_mw": 0.0,
        "reserve_competition_symmetric_delivery_down_locked_mw": 0.0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        if out[col].dtype != np.dtype("float64") or out[col].hasnans:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default).astype(float)
    return out


def available_symmetric_capacity_mw(
    res: pd.DataFrame,
    t: int,
    cfg: dict[str, Any],
    market_for_buffer: str = "FCR",
) -> float:
    b = cfg["BATTERY"]
    e_max = float(b["e_max_mwh"])
    eta = float(b["efficiency"])
    buffer_h = reserve_buffer_h(market_for_buffer)

    soc_min = float(res.at[t, "soc_min_pct"])
    soc_max = float(res.at[t, "soc_max_pct"])
    available_band_mwh = max(0.0, (soc_max - soc_min) / 100.0 * e_max)

    discharge_power = max(0.0, float(res.at[t, "p_discharge_max_mw"]))
    charge_power = max(0.0, float(res.at[t, "p_charge_max_mw"]))

    if buffer_h <= EPS:
        energy_limit = min(discharge_power, charge_power)
    else:
        up_energy_limit = available_band_mwh * eta / buffer_h
        down_energy_limit = available_band_mwh / max(buffer_h * eta, EPS)
        energy_limit = min(up_energy_limit, down_energy_limit)

    return floor_to_step(min(discharge_power, charge_power, energy_limit), STEP_MW)


def make_symmetric_mw_steps(available_mw: float) -> list[float]:
    available_mw = floor_to_step(available_mw, STEP_MW)
    if available_mw < STEP_MW - EPS:
        return []
    n = int(math.floor(available_mw / STEP_MW + EPS))
    return [float(i * STEP_MW) for i in range(1, n + 1)]


def apply_symmetric_reserve_freeze(
    base_res: pd.DataFrame,
    interval: int,
    q_mw: float,
    cfg: dict[str, Any],
    market_for_buffer: str = "FCR",
) -> pd.DataFrame:
    """Freeze q MW as a symmetric reserve block: UP and DOWN at the same time."""
    out = base_res.copy()
    q = float(q_mw)
    if q <= EPS:
        return init_reserve_competition_audit_columns(out)

    # Use the same physical reserve-duration semantics as existing UP/DOWN freezes.
    # FCR/aFRR/mFRR all use RESERVE_BUFFER_HOURS in the current config.
    out = apply_capacity_freeze(out, interval, "up", q, cfg)
    out = apply_capacity_freeze(out, interval, "down", q, cfg)
    out = init_reserve_competition_audit_columns(out)
    out.at[interval, "reserve_competition_symmetric_delivery_up_locked_mw"] += q
    out.at[interval, "reserve_competition_symmetric_delivery_down_locked_mw"] += q
    return out


def apply_fcr_delivery_lock_with_audit(
    base_res: pd.DataFrame,
    interval: int,
    q_mw: float,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    out = apply_symmetric_reserve_freeze(base_res, interval, q_mw, cfg, "FCR")
    out = init_reserve_competition_audit_columns(out)
    q = float(q_mw)
    out.at[interval, "fcr_accepted_mw"] += q
    out.at[interval, "fcr_delivery_up_locked_mw"] += q
    out.at[interval, "fcr_delivery_down_locked_mw"] += q
    return out


def apply_fcr_recovery_lock_after_acceptance(
    base_res: pd.DataFrame,
    source_interval: int,
    accepted_fcr_mw: float,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    before = init_reserve_competition_audit_columns(base_res.copy())
    out = apply_recovery_lock_after_acceptance(
        base_res=before,
        source_interval=source_interval,
        accepted_up_mw=float(accepted_fcr_mw),
        accepted_down_mw=float(accepted_fcr_mw),
        cfg=cfg,
    )
    out = init_reserve_competition_audit_columns(out)
    # Mirror aggregate recovery locks into FCR-specific audit columns for transparency.
    charge_delta = out["afrr_mfrr_recovery_charge_locked_mw"].astype(float) - before["afrr_mfrr_recovery_charge_locked_mw"].astype(float)
    discharge_delta = out["afrr_mfrr_recovery_discharge_locked_mw"].astype(float) - before["afrr_mfrr_recovery_discharge_locked_mw"].astype(float)
    out["fcr_recovery_charge_locked_mw"] += charge_delta.clip(lower=0.0)
    out["fcr_recovery_discharge_locked_mw"] += discharge_delta.clip(lower=0.0)
    return out


def symmetric_step_worst_case_feasible(
    base_res: pd.DataFrame,
    interval: int,
    q_mw: float,
    cfg: dict[str, Any],
    include_recovery: bool = True,
) -> bool:
    try:
        test = apply_symmetric_reserve_freeze(base_res, interval, q_mw, cfg, "FCR")
        if include_recovery:
            test = apply_recovery_lock_after_acceptance(test, interval, q_mw, q_mw, cfg)
            if float(test["afrr_mfrr_uncovered_charge_recovery_mwh"].sum()) > EPS:
                return False
            if float(test["afrr_mfrr_uncovered_discharge_recovery_mwh"].sum()) > EPS:
                return False
        return True
    except Exception:
        return False


def split_symmetric_floor_price(
    floor_price: float,
    up_forecast: float,
    down_forecast: float,
) -> tuple[float, float]:
    """Split one symmetric block floor between UP and DOWN directional bids."""
    floor = max(0.0, float(floor_price))
    up = max(0.0, 0.0 if np.isnan(float(up_forecast)) else float(up_forecast))
    down = max(0.0, 0.0 if np.isnan(float(down_forecast)) else float(down_forecast))
    denom = up + down
    if denom <= EPS:
        return floor / 2.0, floor / 2.0
    return floor * up / denom, floor * down / denom


def calc_symmetric_oc_curve_for_interval(
    cfg: dict[str, Any],
    base_res: pd.DataFrame,
    base_s2_value: float,
    interval: int,
    mw_steps: list[float],
) -> dict[str, Any]:
    dt = float(cfg["DT"])
    prev_loss = 0.0
    raw_bid_curve: list[float] = []
    marginal_oc_eur: list[float] = []
    total_loss_curve: list[float] = []
    test_value_curve: list[float] = []

    progress(f"  symmetric OC interval {interval + 1}/{int(cfg['HORIZON'])}: {len(mw_steps)} step(s)")

    for q in mw_steps:
        test_res = apply_symmetric_reserve_freeze(base_res, interval, q, cfg, "FCR")
        test_value = run_da_mfrr_forecast_value(test_res)
        loss = max(0.0, float(base_s2_value) - float(test_value))
        marginal = max(0.0, loss - prev_loss)
        prev_loss = loss
        bid_price = marginal / (STEP_MW * dt) if dt > EPS else 99999.0

        raw_bid_curve.append(float(bid_price))
        marginal_oc_eur.append(float(marginal))
        total_loss_curve.append(float(loss))
        test_value_curve.append(float(test_value))

    # enforce monotone supply curve
    bid_curve: list[float] = []
    last = -1e30
    for p in raw_bid_curve:
        p2 = float(p) if last <= -1e20 else max(float(p), last + 0.01)
        bid_curve.append(round(p2, 6))
        last = p2

    return {
        "bid_curve": bid_curve,
        "raw_bid_curve": [round(x, 6) for x in raw_bid_curve],
        "marginal_oc_eur": [round(x, 6) for x in marginal_oc_eur],
        "total_loss_s2_curve_eur": [round(x, 6) for x in total_loss_curve],
        "test_s2_value_curve_eur": [round(x, 6) for x in test_value_curve],
    }


def best_package_market_forecast(price_data: dict[str, np.ndarray], t: int) -> dict[str, float | str]:
    afrr_up = float(price_data["fc_afrr_up"][t])
    afrr_down = float(price_data["fc_afrr_down"][t])
    mfrr_up = float(price_data["fc_mfrr_up"][t])
    mfrr_down = float(price_data["fc_mfrr_down"][t])
    fcr = float(price_data["fc_fcr"][t])

    if np.isnan(afrr_up):
        afrr_up = 0.0
    if np.isnan(afrr_down):
        afrr_down = 0.0
    if np.isnan(mfrr_up):
        mfrr_up = 0.0
    if np.isnan(mfrr_down):
        mfrr_down = 0.0
    if np.isnan(fcr):
        fcr = 0.0

    afrr_package = (afrr_up + afrr_down) if market_enabled("aFRR_Capacity", True) else -1e30
    mfrr_package = (mfrr_up + mfrr_down) if market_enabled("mFRR_Capacity", True) else -1e30
    fcr_package = fcr if market_enabled("FCR", True) else -1e30

    return {
        "fcr": fcr_package,
        "afrr": afrr_package,
        "mfrr": mfrr_package,
        "afrr_up": afrr_up,
        "afrr_down": afrr_down,
        "mfrr_up": mfrr_up,
        "mfrr_down": mfrr_down,
    }


def build_reserve_competition_interval_rows(
    price_data: dict[str, np.ndarray],
    base_res: pd.DataFrame,
    cfg: dict[str, Any],
    base_s2_value: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    horizon = int(cfg["HORIZON"])

    for t in range(horizon):
        progress(f"\n[{t + 1}/{horizon}] Building reserve-competition symmetric OC curve")
        available = available_symmetric_capacity_mw(base_res, t, cfg, "FCR")
        steps = make_symmetric_mw_steps(available)
        curve = calc_symmetric_oc_curve_for_interval(
            cfg=cfg,
            base_res=base_res,
            base_s2_value=base_s2_value,
            interval=t,
            mw_steps=steps,
        )
        p = best_package_market_forecast(price_data, t)
        rows.append({
            "Interval": t,
            "Available_Symmetric_MW": available,
            "MW_Steps": steps,
            "Symmetric_Bid_Curve": curve["bid_curve"],
            "Symmetric_Raw_Bid_Curve": curve["raw_bid_curve"],
            "Symmetric_Marginal_OC_EUR": curve["marginal_oc_eur"],
            "Symmetric_Total_Loss_S2_Curve_EUR": curve["total_loss_s2_curve_eur"],
            "Symmetric_Test_S2_Value_Curve_EUR": curve["test_s2_value_curve_eur"],
            "FCR_Forecast_Price_EUR_per_MW_h": float(p["fcr"]),
            "aFRR_UP_Forecast_Price_EUR_per_MW_h": float(p["afrr_up"]),
            "aFRR_DOWN_Forecast_Price_EUR_per_MW_h": float(p["afrr_down"]),
            "mFRR_UP_Forecast_Price_EUR_per_MW_h": float(p["mfrr_up"]),
            "mFRR_DOWN_Forecast_Price_EUR_per_MW_h": float(p["mfrr_down"]),
            "aFRR_Package_Forecast_Price_EUR_per_MW_h": float(p["afrr"]),
            "mFRR_Package_Forecast_Price_EUR_per_MW_h": float(p["mfrr"]),
        })

    return rows


def _clean_price(x: float) -> float:
    """Return a comparable forecast package price.

    NaN is treated as unavailable/zero for enabled markets. Disabled markets are
    filtered before this helper is called.
    """
    try:
        v = float(x)
    except Exception:
        return 0.0
    if np.isnan(v):
        return 0.0
    return v


def _step_preferred_market(row: dict[str, Any], floor_price: float, allow_fcr: bool = True) -> tuple[str, float]:
    """Choose the forecast-best reserve market for one symmetric MW step.

    IMPORTANT MODELING RULE:
    The bid is submitted at the opportunity-cost floor even if the forecast
    margin is negative. Forecast prices are used only to rank markets.

    FCR is considered first only for MW steps where its forecast package price
    is at least as large as both aFRR(up+down) and mFRR(up+down). Otherwise the
    step is left for the later aFRR/mFRR stage.

    Return value second field is only an audit spread = selected_forecast_price
    - floor_price. It is NOT used as a bid/no-bid filter.
    """
    candidates: list[tuple[str, float]] = []
    if allow_fcr and market_enabled("FCR", True):
        candidates.append(("FCR", _clean_price(row["FCR_Forecast_Price_EUR_per_MW_h"])))
    if market_enabled("aFRR_Capacity", True):
        candidates.append(("aFRR", _clean_price(row["aFRR_Package_Forecast_Price_EUR_per_MW_h"])))
    if market_enabled("mFRR_Capacity", True):
        candidates.append(("mFRR", _clean_price(row["mFRR_Package_Forecast_Price_EUR_per_MW_h"])))

    if not candidates:
        return "None", -float(floor_price)

    market, price = max(candidates, key=lambda x: (x[1], 1 if x[0] == "FCR" else 0))
    return market, float(price - float(floor_price))


def _best_non_fcr_capacity_market(row: dict[str, Any], floor_price: float) -> tuple[str, float]:
    """Choose aFRR or mFRR by forecast package price, always submitting at OC.

    No positive-margin requirement is applied. If a reserve market exists, a bid
    is submitted with the opportunity-cost floor; actual clearing decides later.
    """
    candidates: list[tuple[str, float]] = []
    if market_enabled("aFRR_Capacity", True):
        candidates.append(("aFRR", _clean_price(row["aFRR_Package_Forecast_Price_EUR_per_MW_h"])))
    if market_enabled("mFRR_Capacity", True):
        candidates.append(("mFRR", _clean_price(row["mFRR_Package_Forecast_Price_EUR_per_MW_h"])))

    if not candidates:
        return "None", -float(floor_price)

    market, price = max(candidates, key=lambda x: x[1])
    return market, float(price - float(floor_price))


def actual_price(actual_price_data: dict[str, np.ndarray] | None, key: str, t: int) -> float:
    if actual_price_data is None or key not in actual_price_data:
        return float("nan")
    return float(actual_price_data[key][t])


def _lock_capacity_acceptance(
    res: pd.DataFrame,
    interval: int,
    market: str,
    accepted_up_mw: float,
    accepted_down_mw: float,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    out = res.copy()
    if accepted_up_mw > EPS:
        out = apply_delivery_lock_with_audit(out, interval, market, "up", accepted_up_mw, cfg)
    if accepted_down_mw > EPS:
        out = apply_delivery_lock_with_audit(out, interval, market, "down", accepted_down_mw, cfg)
    if accepted_up_mw > EPS or accepted_down_mw > EPS:
        out = apply_recovery_lock_after_acceptance(out, interval, accepted_up_mw, accepted_down_mw, cfg)
    return out


def _try_lock_capacity_worst_case(
    res: pd.DataFrame,
    interval: int,
    market: str,
    q_mw: float,
    cfg: dict[str, Any],
) -> tuple[bool, pd.DataFrame]:
    try:
        test = _lock_capacity_acceptance(res, interval, market, q_mw, q_mw, cfg)
        if float(test["afrr_mfrr_uncovered_charge_recovery_mwh"].sum()) > EPS:
            return False, res
        if float(test["afrr_mfrr_uncovered_discharge_recovery_mwh"].sum()) > EPS:
            return False, res
        return True, test
    except Exception:
        return False, res


def run_reserve_competition_clearing(
    interval_rows: list[dict[str, Any]],
    timestamps: pd.Series,
    base_res: pd.DataFrame,
    actual_price_data: dict[str, np.ndarray] | None,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Sequential market logic:
    1) Use forecast to decide which symmetric MW blocks should be offered to FCR.
    2) Clear FCR using actual FCR price; lock accepted FCR and recovery.
    3) Remaining symmetric capacity is offered to either aFRR or mFRR package, whichever has higher forecast package margin.
    4) aFRR/mFRR clear direction-level with OR semantics.
    """
    dt = float(cfg["DT"])
    horizon = int(cfg["HORIZON"])
    final_res = init_reserve_competition_audit_columns(base_res.copy())
    submitted_wc_res = init_reserve_competition_audit_columns(base_res.copy())

    rows: list[dict[str, Any]] = []
    total_fcr_revenue = 0.0
    total_afrr_revenue = 0.0
    total_mfrr_revenue = 0.0

    for row in interval_rows:
        t = int(row["Interval"])
        bid_curve = list(row["Symmetric_Bid_Curve"])
        max_steps = len(bid_curve)

        fcr_clear = actual_price(actual_price_data, "act_fcr", t)
        afrr_up_clear = actual_price(actual_price_data, "act_afrr_up", t)
        afrr_down_clear = actual_price(actual_price_data, "act_afrr_down", t)
        mfrr_up_clear = actual_price(actual_price_data, "act_mfrr_up", t)
        mfrr_down_clear = actual_price(actual_price_data, "act_mfrr_down", t)

        submitted_fcr_steps = 0
        accepted_fcr_steps = 0
        fcr_revenue = 0.0

        submitted_afrr_steps = 0
        submitted_mfrr_steps = 0
        accepted_afrr_up_steps = 0
        accepted_afrr_down_steps = 0
        accepted_mfrr_up_steps = 0
        accepted_mfrr_down_steps = 0
        afrr_revenue = 0.0
        mfrr_revenue = 0.0

        step_records: list[dict[str, Any]] = []

        # -------- FCR first --------
        fcr_accepted_step_indices: set[int] = set()
        for k, floor_price in enumerate(bid_curve):
            step_no = k + 1
            preferred, margin = _step_preferred_market(row, floor_price, allow_fcr=True)
            if preferred != "FCR":
                continue
            ok_submit, submitted_wc_test = _try_lock_capacity_worst_case(
                submitted_wc_res, t, "FCR", STEP_MW, cfg
            )
            if not ok_submit:
                continue

            submitted_wc_res = submitted_wc_test
            submitted_fcr_steps += 1
            fcr_submitted = True
            fcr_accepted = (not np.isnan(fcr_clear)) and fcr_clear + EPS >= float(floor_price)

            if fcr_accepted:
                try:
                    final_res = apply_fcr_delivery_lock_with_audit(final_res, t, STEP_MW, cfg)
                    final_res = apply_fcr_recovery_lock_after_acceptance(final_res, t, STEP_MW, cfg)
                    if float(final_res["afrr_mfrr_uncovered_charge_recovery_mwh"].sum()) > EPS:
                        raise RuntimeError("uncovered charge recovery after FCR lock")
                    if float(final_res["afrr_mfrr_uncovered_discharge_recovery_mwh"].sum()) > EPS:
                        raise RuntimeError("uncovered discharge recovery after FCR lock")
                    accepted_fcr_steps += 1
                    fcr_accepted_step_indices.add(k)
                    fcr_revenue += STEP_MW * fcr_clear * dt
                except Exception:
                    # If locking would be unsafe, do not submit/accept this step in the simulation.
                    submitted_fcr_steps -= 1
                    fcr_submitted = False
                    fcr_accepted = False

            step_records.append({
                "step": step_no,
                "floor": float(floor_price),
                "forecast_preferred_market": preferred,
                "forecast_margin_eur_per_mw_h": float(margin),
                "fcr_submitted": bool(fcr_submitted),
                "fcr_accepted": bool(fcr_accepted),
            })

        # -------- Then aFRR/mFRR from remaining capacity --------
        # The capacity market is selected only after the FCR clearing state is known.
        remaining_slots = int(math.floor(available_symmetric_capacity_mw(final_res, t, cfg, "aFRR") / STEP_MW + EPS))
        used_capacity_steps = 0

        for k, floor_price in enumerate(bid_curve):
            if used_capacity_steps >= remaining_slots:
                break
            # A step that already cleared as FCR is physically gone.
            if k in fcr_accepted_step_indices:
                continue

            market, margin = _best_non_fcr_capacity_market(row, floor_price)
            if market == "None":
                continue

            if market == "aFRR":
                up_fc = float(row["aFRR_UP_Forecast_Price_EUR_per_MW_h"])
                down_fc = float(row["aFRR_DOWN_Forecast_Price_EUR_per_MW_h"])
                up_bid, down_bid = split_symmetric_floor_price(floor_price, up_fc, down_fc)
                up_clear = afrr_up_clear
                down_clear = afrr_down_clear
            else:
                up_fc = float(row["mFRR_UP_Forecast_Price_EUR_per_MW_h"])
                down_fc = float(row["mFRR_DOWN_Forecast_Price_EUR_per_MW_h"])
                up_bid, down_bid = split_symmetric_floor_price(floor_price, up_fc, down_fc)
                up_clear = mfrr_up_clear
                down_clear = mfrr_down_clear

            # Before submitting, require feasibility if all already-submitted reserve bids
            # plus this UP/DOWN package clear in the worst case.
            ok_wc, submitted_wc_test = _try_lock_capacity_worst_case(
                submitted_wc_res, t, market, STEP_MW, cfg
            )
            if not ok_wc:
                continue
            submitted_wc_res = submitted_wc_test

            if market == "aFRR":
                submitted_afrr_steps += 1
            else:
                submitted_mfrr_steps += 1

            up_accepted = (not np.isnan(up_clear)) and up_clear + EPS >= up_bid
            down_accepted = (not np.isnan(down_clear)) and down_clear + EPS >= down_bid

            accepted_up_mw = STEP_MW if up_accepted else 0.0
            accepted_down_mw = STEP_MW if down_accepted else 0.0

            if accepted_up_mw > EPS or accepted_down_mw > EPS:
                final_res = _lock_capacity_acceptance(final_res, t, market, accepted_up_mw, accepted_down_mw, cfg)
                used_capacity_steps += 1

                if market == "aFRR":
                    accepted_afrr_up_steps += int(up_accepted)
                    accepted_afrr_down_steps += int(down_accepted)
                    if up_accepted:
                        afrr_revenue += STEP_MW * up_clear * dt
                    if down_accepted:
                        afrr_revenue += STEP_MW * down_clear * dt
                else:
                    accepted_mfrr_up_steps += int(up_accepted)
                    accepted_mfrr_down_steps += int(down_accepted)
                    if up_accepted:
                        mfrr_revenue += STEP_MW * up_clear * dt
                    if down_accepted:
                        mfrr_revenue += STEP_MW * down_clear * dt
            else:
                # Bid submitted but neither direction clears. The capacity remains available for later DA stage.
                used_capacity_steps += 0

            step_records.append({
                "step": k + 1,
                "floor": float(floor_price),
                "forecast_preferred_market_after_fcr": market,
                "forecast_margin_eur_per_mw_h": float(margin),
                "up_bid_floor": float(up_bid),
                "down_bid_floor": float(down_bid),
                "up_accepted": bool(up_accepted),
                "down_accepted": bool(down_accepted),
            })

        total_fcr_revenue += fcr_revenue
        total_afrr_revenue += afrr_revenue
        total_mfrr_revenue += mfrr_revenue

        rows.append({
            "Interval": t,
            "timestamp": timestamps.iloc[t],
            "Available_Symmetric_MW": round(float(row["Available_Symmetric_MW"]), 6),
            "Symmetric_OC_Bid_Curve_EUR_per_MW_h": json.dumps(row["Symmetric_Bid_Curve"]),
            "Symmetric_Raw_OC_Bid_Curve_EUR_per_MW_h": json.dumps(row["Symmetric_Raw_Bid_Curve"]),
            "Symmetric_Marginal_OC_EUR_per_Interval": json.dumps(row["Symmetric_Marginal_OC_EUR"]),
            "FCR_Forecast_Price_EUR_per_MW_h": round(float(row["FCR_Forecast_Price_EUR_per_MW_h"]), 6),
            "aFRR_Package_Forecast_Price_EUR_per_MW_h": round(float(row["aFRR_Package_Forecast_Price_EUR_per_MW_h"]), 6),
            "mFRR_Package_Forecast_Price_EUR_per_MW_h": round(float(row["mFRR_Package_Forecast_Price_EUR_per_MW_h"]), 6),
            "FCR_Actual_Clearing_Price_EUR_per_MW_h": None if np.isnan(fcr_clear) else round(fcr_clear, 6),
            "aFRR_UP_Actual_Clearing_Price_EUR_per_MW_h": None if np.isnan(afrr_up_clear) else round(afrr_up_clear, 6),
            "aFRR_DOWN_Actual_Clearing_Price_EUR_per_MW_h": None if np.isnan(afrr_down_clear) else round(afrr_down_clear, 6),
            "mFRR_UP_Actual_Clearing_Price_EUR_per_MW_h": None if np.isnan(mfrr_up_clear) else round(mfrr_up_clear, 6),
            "mFRR_DOWN_Actual_Clearing_Price_EUR_per_MW_h": None if np.isnan(mfrr_down_clear) else round(mfrr_down_clear, 6),
            "Bid_FCR_MW": round(submitted_fcr_steps * STEP_MW, 6),
            "Accepted_FCR_MW": round(accepted_fcr_steps * STEP_MW, 6),
            "Bid_aFRR_Package_MW": round(submitted_afrr_steps * STEP_MW, 6),
            "Bid_mFRR_Package_MW": round(submitted_mfrr_steps * STEP_MW, 6),
            "Accepted_aFRR_UP_MW": round(accepted_afrr_up_steps * STEP_MW, 6),
            "Accepted_aFRR_DOWN_MW": round(accepted_afrr_down_steps * STEP_MW, 6),
            "Accepted_mFRR_UP_MW": round(accepted_mfrr_up_steps * STEP_MW, 6),
            "Accepted_mFRR_DOWN_MW": round(accepted_mfrr_down_steps * STEP_MW, 6),
            "FCR_Revenue_EUR": round(fcr_revenue, 6),
            "aFRR_Capacity_Revenue_EUR": round(afrr_revenue, 6),
            "mFRR_Capacity_Revenue_EUR": round(mfrr_revenue, 6),
            "Reserve_Capacity_Revenue_EUR": round(fcr_revenue + afrr_revenue + mfrr_revenue, 6),
            "Step_Decisions_JSON": json.dumps(step_records),
        })

    results = pd.DataFrame(rows)
    totals = {
        "total_fcr_revenue_eur": float(total_fcr_revenue),
        "total_afrr_capacity_revenue_eur": float(total_afrr_revenue),
        "total_mfrr_capacity_revenue_eur": float(total_mfrr_revenue),
        "total_reserve_capacity_revenue_eur": float(total_fcr_revenue + total_afrr_revenue + total_mfrr_revenue),
    }
    return results, final_res, submitted_wc_res, totals


# ============================================================
# MAIN RUNNER - rewritten
# ============================================================
def run_afrr_mfrr_capacity_case(
    print_console: bool = True,
    save_csv: bool = True,
    results_file: str | Path = RESULTS_FILE,
    reservation_output_file: str | Path = RESERVATIONS_OUTPUT_FILE,
    submitted_worst_case_reservation_file: str | Path = SUBMITTED_WORST_CASE_RESERVATIONS_FILE,
    use_actuals_for_backtest_clearing: bool = True,
) -> dict[str, Any]:
    """Run the reserve-capacity competition stage.

    This replaces the old after-FCR aFRR/mFRR-only stage.

    Forecast decision:
        For each symmetric MW block, compare FCR vs aFRR(UP+DOWN) vs mFRR(UP+DOWN)
        against a symmetric DA+mFRR-energy opportunity-cost floor.

    Sequential clearing:
        1. FCR clears first and locks symmetric delivery + recovery.
        2. Remaining capacity is offered to either aFRR or mFRR package.
        3. aFRR/mFRR are accepted direction-by-direction with OR semantics.
    """
    start = time.time()
    cfg = build_cfg()
    horizon = int(cfg["HORIZON"])

    df_fc, timestamps = load_forecast_input_data(horizon)
    forecast_price_data = get_forecast_reserve_price_data(df_fc, horizon)

    # In the new design, this stage owns FCR+aFRR+mFRR competition. It starts from
    # free battery reservation unless an explicit input reservation is provided.
    start_from_free = bool(getattr(cfg_module, "RESERVE_COMPETITION_START_FROM_FREE_RESERVATION", True))
    if start_from_free:
        base_res = make_free_battery_reservation_profile(timestamps, cfg)
    else:
        base_res = load_fcr_reservations(timestamps, cfg)

    base_s2_value = run_da_mfrr_forecast_value(base_res)
    progress(f"Base S2 forecast value before reserve competition: {base_s2_value:.2f} EUR")

    interval_rows = build_reserve_competition_interval_rows(
        price_data=forecast_price_data,
        base_res=base_res,
        cfg=cfg,
        base_s2_value=base_s2_value,
    )

    actual_price_data = None
    if use_actuals_for_backtest_clearing:
        actual_price_data = load_actual_reserve_price_data_if_available(timestamps, horizon)
        if actual_price_data is None:
            progress("Actual reserve price file not found. Realized clearing will be zero/NA.")

    results, final_res, submitted_wc_res, totals = run_reserve_competition_clearing(
        interval_rows=interval_rows,
        timestamps=timestamps,
        base_res=base_res,
        actual_price_data=actual_price_data,
        cfg=cfg,
    )

    total_uncovered_charge = float(final_res["afrr_mfrr_uncovered_charge_recovery_mwh"].sum())
    total_uncovered_discharge = float(final_res["afrr_mfrr_uncovered_discharge_recovery_mwh"].sum())
    submitted_uncovered_charge = float(submitted_wc_res["afrr_mfrr_uncovered_charge_recovery_mwh"].sum())
    submitted_uncovered_discharge = float(submitted_wc_res["afrr_mfrr_uncovered_discharge_recovery_mwh"].sum())

    # Attach final remaining reservation/audit columns to the interval table.
    audit_cols = [
        "p_charge_max_mw",
        "p_discharge_max_mw",
        "soc_min_pct",
        "soc_max_pct",
        "fcr_accepted_mw",
        "fcr_delivery_up_locked_mw",
        "fcr_delivery_down_locked_mw",
        "fcr_recovery_charge_locked_mw",
        "fcr_recovery_discharge_locked_mw",
        "afrr_up_accepted_mw",
        "afrr_down_accepted_mw",
        "mfrr_up_accepted_mw",
        "mfrr_down_accepted_mw",
        "afrr_mfrr_delivery_up_locked_mw",
        "afrr_mfrr_delivery_down_locked_mw",
        "afrr_mfrr_recovery_charge_locked_mw",
        "afrr_mfrr_recovery_discharge_locked_mw",
        "afrr_mfrr_uncovered_charge_recovery_mwh",
        "afrr_mfrr_uncovered_discharge_recovery_mwh",
    ]
    for col in audit_cols:
        if col in final_res.columns:
            results[col] = final_res[col].values[: len(results)]

    results["Base_S2_Forecast_Value_EUR"] = round(float(base_s2_value), 6)
    results["Total_FCR_Revenue_EUR"] = round(float(totals["total_fcr_revenue_eur"]), 6)
    results["Total_aFRR_Capacity_Revenue_EUR"] = round(float(totals["total_afrr_capacity_revenue_eur"]), 6)
    results["Total_mFRR_Capacity_Revenue_EUR"] = round(float(totals["total_mfrr_capacity_revenue_eur"]), 6)
    results["Total_Reserve_Capacity_Revenue_EUR"] = round(float(totals["total_reserve_capacity_revenue_eur"]), 6)
    results["Runtime_Seconds"] = round(time.time() - start, 3)

    if save_csv:
        results.to_csv(results_file, index=False, encoding="utf-8-sig")
        final_res.to_csv(reservation_output_file, index=False, encoding="utf-8-sig")
        submitted_wc_res.to_csv(submitted_worst_case_reservation_file, index=False, encoding="utf-8-sig")

    if print_console:
        print("-" * 120)
        print("RESERVE CAPACITY COMPETITION SUMMARY")
        print(f"FCR revenue: {totals['total_fcr_revenue_eur']:.2f} EUR")
        print(f"aFRR capacity revenue: {totals['total_afrr_capacity_revenue_eur']:.2f} EUR")
        print(f"mFRR capacity revenue: {totals['total_mfrr_capacity_revenue_eur']:.2f} EUR")
        print(f"TOTAL reserve capacity revenue: {totals['total_reserve_capacity_revenue_eur']:.2f} EUR")
        print(f"Final uncovered recovery: charge={total_uncovered_charge:.6f} MWh, discharge={total_uncovered_discharge:.6f} MWh")

    return {
        "status": "Optimal",
        "results": results,
        "final_reservation": final_res,
        "submitted_worst_case_reservation": submitted_wc_res,
        "base_s2_forecast_value_eur": float(base_s2_value),
        "base_da_mfrr_forecast_value_eur": float(base_s2_value),
        # Backward-compatible key: non-FCR aFRR+mFRR capacity revenue only.
        # The new FCR revenue is exposed separately so the runner can sum FCR + aFRR/mFRR + DA.
        "total_afrr_mfrr_capacity_revenue_eur": float(
            totals["total_afrr_capacity_revenue_eur"] + totals["total_mfrr_capacity_revenue_eur"]
        ),
        "total_reserve_capacity_revenue_eur": float(totals["total_reserve_capacity_revenue_eur"]),
        "total_fcr_revenue_eur": float(totals["total_fcr_revenue_eur"]),
        "total_afrr_capacity_revenue_eur": float(totals["total_afrr_capacity_revenue_eur"]),
        "total_mfrr_capacity_revenue_eur": float(totals["total_mfrr_capacity_revenue_eur"]),
        "dp_forecast_scheduling_objective_eur": 0.0,
        "total_uncovered_charge_recovery_mwh": total_uncovered_charge,
        "total_uncovered_discharge_recovery_mwh": total_uncovered_discharge,
        "submitted_worst_case_uncovered_charge_recovery_mwh": submitted_uncovered_charge,
        "submitted_worst_case_uncovered_discharge_recovery_mwh": submitted_uncovered_discharge,
        "submitted_worst_case_repair_removed_steps": 0,
        "elapsed_seconds": round(time.time() - start, 3),
    }


def main() -> None:
    run_afrr_mfrr_capacity_case(
        print_console=True,
        save_csv=True,
        results_file=RESULTS_FILE,
        reservation_output_file=RESERVATIONS_OUTPUT_FILE,
        submitted_worst_case_reservation_file=SUBMITTED_WORST_CASE_RESERVATIONS_FILE,
        use_actuals_for_backtest_clearing=True,
    )



if __name__ == "__main__":
    main()
