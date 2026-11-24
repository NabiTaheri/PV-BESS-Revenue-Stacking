# make_forecast.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# تنظیمات ساده خطای پیش‌بینی
# =============================================================================

PV_ERROR_PCT = 0.3
PV_FORECAST_CAP_MW = 5.0
DA_PRICE_ERROR_PCT = 0.3
BALANCING_PRICE_ERROR_PCT = 0.3
CAPACITY_PRICE_ERROR_PCT = 0.3
BALANCING_FLAG_ERROR_RATE = 0.00

# None یعنی هر بار اجرا، ترکیب تصادفی جدید ساخته می‌شود.
# اگر عدد بدهی، خروجی تکرارپذیر می‌شود.
DEFAULT_RANDOM_SEED: int | None = None


COLUMN_MAPPING = {
    "pv_actual_mw": "pv_forecast_mw",
    "da_clearing_price": "da_price_forecast",
    "imbalance_price_actual": "imbalance_price_forecast",
    "balancing_price": "balancing_price_forecast",
    "afrr_capacity_price_up": "afrr_capacity_price_up_forecast",
    "afrr_capacity_price_down": "afrr_capacity_price_down_forecast",
    "mfrr_capacity_price_up": "mfrr_capacity_price_up_forecast",
    "mfrr_capacity_price_down": "mfrr_capacity_price_down_forecast",
    "fcr_clearing_price": "fcr_price_forecast",
}


ERROR_BY_COLUMN = {
    "pv_actual_mw": PV_ERROR_PCT,
    "da_clearing_price": DA_PRICE_ERROR_PCT,
    "imbalance_price_actual": BALANCING_PRICE_ERROR_PCT,
    "balancing_price": BALANCING_PRICE_ERROR_PCT,
    "afrr_capacity_price_up": CAPACITY_PRICE_ERROR_PCT,
    "afrr_capacity_price_down": CAPACITY_PRICE_ERROR_PCT,
    "mfrr_capacity_price_up": CAPACITY_PRICE_ERROR_PCT,
    "mfrr_capacity_price_down": CAPACITY_PRICE_ERROR_PCT,
    "fcr_clearing_price": CAPACITY_PRICE_ERROR_PCT,
}


def add_multiplicative_error(
    values: pd.Series,
    *,
    error_pct: float,
    rng: np.random.Generator,
) -> pd.Series:
    if error_pct < 0:
        raise ValueError("error_pct must be >= 0.")

    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.isna().any():
        raise ValueError("Input contains non-numeric values.")

    factors = rng.uniform(1.0 - error_pct, 1.0 + error_pct, size=len(numeric_values))
    forecast = numeric_values * factors
    forecast.loc[numeric_values == 0] = 0.0
    return forecast


def make_balancing_flag_forecast(
    actual_flags: np.ndarray,
    *,
    error_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("BALANCING_FLAG_ERROR_RATE must be in [0, 1].")

    valid_flags = {-1, 0, 1}
    bad_flags = set(actual_flags) - valid_flags
    if bad_flags:
        raise ValueError(f"Invalid balancing_flag values: {bad_flags}. Expected only -1, 0, 1.")

    forecast_flags = actual_flags.copy()
    n = len(actual_flags)

    target_error_rows = int(round(error_rate * n))
    if target_error_rows % 2 == 1:
        target_error_rows -= 1

    used_indices: set[int] = set()
    error_count = 0
    all_indices = np.arange(n)

    attempts = 0
    max_attempts = 100_000

    while error_count < target_error_rows and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(all_indices, size=2, replace=False)

        if int(i) in used_indices or int(j) in used_indices:
            continue

        if forecast_flags[i] == forecast_flags[j]:
            continue

        forecast_flags[i], forecast_flags[j] = forecast_flags[j], forecast_flags[i]
        used_indices.add(int(i))
        used_indices.add(int(j))
        error_count += 2

    if error_count < target_error_rows:
        print(
            f"WARNING: requested {target_error_rows} flag errors, "
            f"but only created {error_count}."
        )

    return forecast_flags


def make_forecast(
    *,
    input_file: Path,
    output_file: Path,
    seed: int | None = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    # seed=None یعنی هر بار از entropy سیستم استفاده می‌شود و خروجی جدید می‌شود.
    rng = np.random.default_rng(seed)

    df = pd.read_csv(input_file)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing column in {input_file}: timestamp")

    out = pd.DataFrame({"timestamp": df["timestamp"]})

    for actual_col, forecast_col in COLUMN_MAPPING.items():
        if actual_col not in df.columns:
            out[forecast_col] = 0.0
            continue

        try:
            out[forecast_col] = add_multiplicative_error(
                df[actual_col],
                error_pct=ERROR_BY_COLUMN[actual_col],
                rng=rng,
            )
        except ValueError as exc:
            bad_values = df.loc[
                pd.to_numeric(df[actual_col], errors="coerce").isna(),
                ["timestamp", actual_col],
            ].head(20)
            raise ValueError(f"Column {actual_col} has invalid values:\n{bad_values}") from exc

    out["pv_forecast_mw"] = out["pv_forecast_mw"].clip(
        lower=0.0,
        upper=PV_FORECAST_CAP_MW,
    )

    if "balancing_flag" not in df.columns:
        raise ValueError(f"Missing column in {input_file}: balancing_flag")

    actual_flags = pd.to_numeric(df["balancing_flag"], errors="raise").astype(int).to_numpy()
    forecast_flags = make_balancing_flag_forecast(
        actual_flags,
        error_rate=BALANCING_FLAG_ERROR_RATE,
        rng=rng,
    )

    out["balancing_flag_forecast"] = forecast_flags
    out.to_csv(output_file, index=False, encoding="utf-8-sig")

    actual_flag_error_rate = float(np.mean(forecast_flags != actual_flags))
    seed_text = "random/OS entropy" if seed is None else str(seed)

    print(f"Forecast file created: {output_file}")
    print(f"Rows: {len(out)}")
    print(f"Random seed: {seed_text}")
    print(f"PV error pct: {PV_ERROR_PCT:.2%}")
    print(f"DA price error pct: {DA_PRICE_ERROR_PCT:.2%}")
    print(f"Balancing price error pct: {BALANCING_PRICE_ERROR_PCT:.2%}")
    print(f"Capacity price error pct: {CAPACITY_PRICE_ERROR_PCT:.2%}")
    print(f"Balancing flag error rate: {actual_flag_error_rate:.2%}")

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rolling forecast CSV from actuals.")
    parser.add_argument("--input-file", default="actuals.csv")
    parser.add_argument("--output-file", default="forecasts_rolling.csv")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed. If omitted, every run produces a new random forecast. "
            "Use --seed 42 for reproducible output."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_forecast(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
