# market_config.py
# Unified configuration file
# Combines:
#   1) capacity/shared market_config settings
#   2) DA/DA+mFRR-energy MarketConfig settings
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    import pulp
except Exception:  # pragma: no cover
    pulp = None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


# =============================================================================
# FILES
# =============================================================================
FORECAST_FILE = Path("forecasts_rolling.csv")
ACTUALS_FILE = Path("actuals.csv")

# Reservation entering DA + mFRR-energy.
# In the reserve-competition stack this is normally produced by the reserve stage.
RESERVATION_FILE = Path("battery_reservations_after_fcr_afrr_mfrr_capacity.csv")

# Scenario runner usually overrides these paths per day.
OUTPUT_REPORT = Path("da_energy_report.csv")
OUTPUT_SUMMARY = Path("da_energy_summary.csv")


# =============================================================================
# SOLVER
# =============================================================================
SOLVER_NAME = "CBC"
SOLVER_PATH = "glpsol"


def make_solver():
    """Return a PuLP solver if PuLP is available."""
    if pulp is None:
        return None

    if SOLVER_NAME.upper() == "GLPK":
        return pulp.GLPK_CMD(path=str(SOLVER_PATH), msg=False)

    return pulp.PULP_CBC_CMD(msg=False)


# =============================================================================
# TIME
# =============================================================================
RESOLUTION_MIN = 15
DT_H = RESOLUTION_MIN / 60.0
DT = DT_H
HORIZON = 96
EPS = 1e-9


# =============================================================================
# BATTERY / SITE
# =============================================================================
BATTERY_P_MAX_MW = _env_float("BALTIC_BESS_P_MW", 5.0)
BATTERY_E_MWH = _env_float("BALTIC_BESS_E_MWH", 10.0)

EFF = 0.95
INV_EFF = 1.0 / EFF

PHYSICAL_SOC_MIN_PCT = 20.0
PHYSICAL_SOC_MAX_PCT = 80.0
SOC_INIT_PCT = 50.0
SOC_TARGET_PCT = 50.0

# Hard user requirement: every DA day must end at 50%.
SOC_TERMINAL_SYNC = True

# Reserve energy duration used for symmetric reserve and recovery accounting.
RESERVE_ENERGY_DURATION_H = 0.25

PV_MAX_MW = 5.0
GRID_LIMIT_MW = 10.0
GRID_CHARGING_ALLOWED = True
AUTO_CHARGE_EXCESS_PV = False
PV_BID_FACTOR = 1.0

# Dictionary format used by reserve-capacity modules.
BATTERY = {
    "p_max_mw": BATTERY_P_MAX_MW,
    "e_max_mwh": BATTERY_E_MWH,
    "efficiency": EFF,
    "soc_min_pct": PHYSICAL_SOC_MIN_PCT,
    "soc_max_pct": PHYSICAL_SOC_MAX_PCT,
    "soc_init_pct": SOC_INIT_PCT,
}


# =============================================================================
# MARKET SWITCHES
# =============================================================================
ENABLE_BATTERY = True
ENABLE_DA_MARKET = True
ENABLE_MFRR_ENERGY_MARKET = True
ENABLE_BALANCING_MARKET = ENABLE_MFRR_ENERGY_MARKET

# DA-stage virtual value of mFRR/balancing forecast.
DA_STAGE_INCLUDE_VIRTUAL_MFRR_VALUE = ENABLE_MFRR_ENERGY_MARKET

MARKETS = {
    "DA": ENABLE_DA_MARKET,
    "mFRR_Energy": ENABLE_MFRR_ENERGY_MARKET,
    "FCR": True,
    "aFRR_Capacity": True,
    "mFRR_Capacity": True,
}


# =============================================================================
# DA MARKET
# =============================================================================
DA_TRADE_LOT_MW = 0.1
CURVE_VOLUME_ROUND_MODE = "nearest"  # "nearest", "floor", or "ceil"

TECHNICAL_PRICE_MIN = -500.0
TECHNICAL_PRICE_MAX = 4000.0

# Export/sell at negative DA prices is blocked by default.
DA_EXPORT_PRICE_FLOOR_EUR_PER_MWH = 0.0
BLOCK_DA_EXPORT_BELOW_FLOOR = True

# Legacy DA bid-shaping parameters retained for compatibility.
DA_EXTRA_BUY_DISCOUNT_FACTOR = 0.50
DA_EXTRA_SELL_PREMIUM_FACTOR = 0.50


# =============================================================================
# BALANCING / mFRR ENERGY MARKET
# =============================================================================
MFRR_TRADE_LOT_MW = 0.1
BALANCING_BID_LOT_MW = MFRR_TRADE_LOT_MW

BALANCING_UP_MIN_PRICE_EUR_PER_MWH = 0.0
BALANCING_DOWN_MAX_PRICE_EUR_PER_MWH = 0.0
BALANCING_FUTURE_VALUE_LOOKAHEAD_STEPS = 16
BALANCING_KEEP_DA_FUTURE_FEASIBLE = True
MFRR_ACTIVATION_FRACTION = _env_float("MFRR_ACTIVATION_FRACTION", 1.0)

# Capacity valuation comparison inside DA/balancing engine.
CAPACITY_VALUE_STEP_MW = 1.0
DA_MUST_BEAT_BALANCING_MARGIN_EUR_PER_MW = 0.0


# =============================================================================
# ECONOMICS / SAFETY GUARDS
# =============================================================================
# NOTE: The two supplied configs had different values:
#   - capacity/shared config: 5.0
#   - DA/market_config.py: 25.0
# The unified file keeps DA economics as the active default, because this value
# is injected into MarketConfig and therefore directly affects DA + mFRR energy.
CYCLE_COST_EUR_PER_MWH = _env_float("BALTIC_CYCLE_COST_EUR_PER_MWH", 25.0)

# Keep the old capacity value under an explicit name if a module needs it.
CAPACITY_LEGACY_CYCLE_COST_EUR_PER_MWH = 5.0

OPTIONAL_PARTICIPATION_NO_LOSS = True
DA_BACKTEST_NO_LOSS_GUARD = True
DA_BACKTEST_PHYSICAL_DELIVERY_GUARD = True
BALANCING_INCREMENTAL_NO_LOSS_GUARD = True
MFRR_INCREMENTAL_NO_LOSS_GUARD = BALANCING_INCREMENTAL_NO_LOSS_GUARD
NO_TRADE_IF_FORECAST_OBJECTIVE_NONPOSITIVE = True

# False = production-safe. If reservation timestamps do not match forecast/actual
# timestamps, the engine raises instead of silently using row order.
ALLOW_RESERVATION_ROW_ORDER_FALLBACK = False


# =============================================================================
# DA DP DISCRETIZATION
# =============================================================================
# Used by older/direct DA+mFRR solvers if present.
# Smaller values improve accuracy but increase runtime.
SOC_DP_STEP_MWH = 0.25
POWER_DP_STEP_MW = 0.1


# =============================================================================
# CAPACITY-MARKET CONFIG SHARED BY FCR / aFRR / mFRR CAPACITY
# =============================================================================
BIDDING_STEPS_MW = {
    "FCR": 1.0,
    "aFRR": 1.0,
    "mFRR": 1.0,
}

RESERVE_BUFFER_HOURS = {
    "FCR": RESERVE_ENERGY_DURATION_H,
    "aFRR": RESERVE_ENERGY_DURATION_H,
    "mFRR": RESERVE_ENERGY_DURATION_H,
}

RISK_MANAGEMENT = {
    # FCR risk accounting.
    "ENABLE_FCR_RISK_LIMIT": True,
    "MAX_CUMULATIVE_EXPOSURE_MWH": 999999.0,
    # Explicit aliases used by some FCR variants.
    "FCR_MAX_CUMULATIVE_EXPOSURE_MWH": 999999.0,
    "FCR_EXPOSURE_PER_MW_INTERVAL_MWH": RESERVE_ENERGY_DURATION_H / EFF,
    "FCR_RECOVERY_PER_INTERVAL_MWH": BATTERY_P_MAX_MW * DT_H * EFF,

    # aFRR/mFRR capacity risk accounting.
    "ENABLE_AFRR_MFRR_CAPACITY_RISK_LIMIT": True,
    "ENABLE_AFRR_MFRR_RISK_LIMIT": True,
    "AFRR_MFRR_MAX_CUMULATIVE_EXPOSURE_MWH": 999999.0,
    "AFRR_MFRR_UP_MAX_CUMULATIVE_EXPOSURE_MWH": 999999.0,
    "AFRR_MFRR_DOWN_MAX_CUMULATIVE_EXPOSURE_MWH": 999999.0,
    "AFRR_MFRR_RECOVERY_PER_INTERVAL_MWH": BATTERY_P_MAX_MW * DT_H * EFF,

    # Simultaneous UP/DOWN exposure mode:
    # "max" = worst single direction, consistent with mutually exclusive activation.
    # "sum" = conservative stress test.
    "AFRR_MFRR_JOINT_EXPOSURE_MODE": "max",
}

# FCR recovery locking used both in final reservation and OC pricing.
FCR_RECOVERY_START_OFFSET_INTERVALS = 1
FCR_RECOVERY_MAX_INTERVALS = None
FCR_REQUIRE_FULL_RECOVERY_FOR_OC = True

# aFRR/mFRR reserve recovery settings.
AFRR_MFRR_RECOVERY_START_OFFSET_INTERVALS = 1
AFRR_MFRR_RECOVERY_MAX_INTERVALS = None
AFRR_MFRR_REQUIRE_FULL_RECOVERY_FOR_OC = True

# Optional DP scale used by reserve modules.
FCR_DP_RISK_SCALE_PER_MWH = 1000
AFRR_MFRR_DP_RISK_SCALE_PER_MWH = 1000


# =============================================================================
# OUTPUT FILES FOR RESERVE / CAPACITY MODULES
# =============================================================================
FCR_RESULTS_FILE = "market_bidding_results_fcr_with_recovery_lock.csv"
FCR_RESERVATIONS_FILE = "battery_reservations_fcr_symmetric.csv"

AFRR_MFRR_CAPACITY_RESULTS_FILE = "afrr_mfrr_capacity_value_forecast_only.csv"
AFRR_MFRR_RESULTS_FILE = "market_bidding_results_afrr_mfrr_capacity_after_fcr.csv"
AFRR_MFRR_RESERVATIONS_FILE = "battery_reservations_after_fcr_afrr_mfrr_capacity.csv"
AFRR_MFRR_SUBMITTED_WORST_CASE_RESERVATIONS_FILE = (
    "battery_reservations_after_fcr_afrr_mfrr_submitted_worst_case.csv"
)

# Reserve-competition stage aliases.
RESERVE_COMPETITION_RESULTS_FILE = "market_bidding_results_reserve_competition.csv"
RESERVE_COMPETITION_RESERVATIONS_FILE = "battery_reservations_after_reserve_competition.csv"


# =============================================================================
# PRINT SETTINGS
# =============================================================================
PRINT_INTERVAL_TABLE = True

PRINT_SETTINGS = {
    "PRINT_FCR_RISK_STATUS": True,
    "PRINT_AFRR_MFRR_VALUE_PROGRESS": False,
    "PRINT_AFRR_MFRR_RISK_STATUS": True,
    "PRINT_RESERVE_COMPETITION_PROGRESS": False,
}


# =============================================================================
# DATACLASS CONFIG FOR DA / DA+mFRR ENERGY ENGINE
# =============================================================================
@dataclass(frozen=True)
class MarketConfig:
    """Unified config object for DA-only and DA + optional balancing engines.

    Also exposes reserve/capacity settings as dataclass fields so scenario
    runners may pass one config object through a larger stack if needed.
    """

    forecast_file: Path = FORECAST_FILE
    actuals_file: Path = ACTUALS_FILE
    reservation_file: Path = RESERVATION_FILE
    output_report: Path = OUTPUT_REPORT
    output_summary: Path | None = OUTPUT_SUMMARY

    soc_init_pct: float = SOC_INIT_PCT
    soc_target_pct: float = SOC_TARGET_PCT
    rolling_soc_reference_pct: float = SOC_TARGET_PCT
    rolling_recovery_target_soc_pct: float = SOC_TARGET_PCT
    soc_terminal_sync: bool = SOC_TERMINAL_SYNC
    rolling_terminal_soc_sync: bool = SOC_TERMINAL_SYNC

    battery_p_max_mw: float = BATTERY_P_MAX_MW
    battery_e_mwh: float = BATTERY_E_MWH
    eff: float = EFF

    physical_soc_min_pct: float = PHYSICAL_SOC_MIN_PCT
    physical_soc_max_pct: float = PHYSICAL_SOC_MAX_PCT

    pv_max_mw: float = PV_MAX_MW
    grid_limit_mw: float = GRID_LIMIT_MW
    grid_charging_allowed: bool = GRID_CHARGING_ALLOWED

    da_trade_lot_mw: float = DA_TRADE_LOT_MW
    mfrr_trade_lot_mw: float = MFRR_TRADE_LOT_MW
    balancing_bid_lot_mw: float = BALANCING_BID_LOT_MW
    mfrr_activation_fraction: float = MFRR_ACTIVATION_FRACTION
    capacity_value_step_mw: float = CAPACITY_VALUE_STEP_MW
    cycle_cost_eur_per_mwh: float = CYCLE_COST_EUR_PER_MWH

    da_must_beat_balancing_margin_eur_per_mw: float = DA_MUST_BEAT_BALANCING_MARGIN_EUR_PER_MW
    block_da_export_below_floor: bool = BLOCK_DA_EXPORT_BELOW_FLOOR
    da_export_price_floor_eur_per_mwh: float = DA_EXPORT_PRICE_FLOOR_EUR_PER_MWH

    da_backtest_physical_delivery_guard: bool = DA_BACKTEST_PHYSICAL_DELIVERY_GUARD
    da_backtest_no_loss_guard: bool = DA_BACKTEST_NO_LOSS_GUARD
    optional_participation_no_loss: bool = OPTIONAL_PARTICIPATION_NO_LOSS
    balancing_incremental_no_loss_guard: bool = BALANCING_INCREMENTAL_NO_LOSS_GUARD
    mfrr_incremental_no_loss_guard: bool = MFRR_INCREMENTAL_NO_LOSS_GUARD
    no_trade_if_forecast_objective_nonpositive: bool = NO_TRADE_IF_FORECAST_OBJECTIVE_NONPOSITIVE
    auto_charge_excess_pv: bool = AUTO_CHARGE_EXCESS_PV

    enable_battery: bool = ENABLE_BATTERY
    enable_da_market: bool = ENABLE_DA_MARKET
    enable_mfrr_energy_market: bool = ENABLE_MFRR_ENERGY_MARKET
    enable_balancing_market: bool = ENABLE_BALANCING_MARKET
    include_hypothetical_mfrr_in_da_stage: bool = DA_STAGE_INCLUDE_VIRTUAL_MFRR_VALUE

    reservation_file_required: bool = True
    allow_reservation_row_order_fallback: bool = ALLOW_RESERVATION_ROW_ORDER_FALLBACK
    print_interval_table: bool = PRINT_INTERVAL_TABLE

    # Reserve/capacity settings exposed for compatibility.
    reserve_energy_duration_h: float = RESERVE_ENERGY_DURATION_H
    fcr_recovery_start_offset_intervals: int = FCR_RECOVERY_START_OFFSET_INTERVALS
    fcr_recovery_max_intervals: int | None = FCR_RECOVERY_MAX_INTERVALS
    afrr_mfrr_recovery_start_offset_intervals: int = AFRR_MFRR_RECOVERY_START_OFFSET_INTERVALS
    afrr_mfrr_recovery_max_intervals: int | None = AFRR_MFRR_RECOVERY_MAX_INTERVALS
