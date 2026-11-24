#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

ROOT = Path(__file__).resolve().parent
MC_ROOT = Path(os.environ.get("BALTIC_MC_ROOT", str(ROOT / "monte_carlo_results2"))).resolve()

ACT_DIR = MC_ROOT / "random_activation_sensitivity"
TABLE_DIR = MC_ROOT / "paper_table_outputs_current_runs"

SCENARIO_FILE = TABLE_DIR / "scenario_level_inputs_with_s1_s2_s3.csv"
SAMPLES_FILE = ACT_DIR / "all_random_activation_samples.csv"

OUT_DIR = MC_ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG1_PATH = OUT_DIR / "revenue_distributions.png"
FIG2_PATH = OUT_DIR / "s3_pooled_hist.png"


# =============================================================================
# HELPERS
# =============================================================================

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def euro_axis(ax) -> None:
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )


def euro_xaxis(ax) -> None:
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )


# =============================================================================
# FIGURE 1: BETWEEN-SCENARIO REVENUE DISTRIBUTIONS
# =============================================================================

def plot_revenue_distributions(scenario: pd.DataFrame) -> None:
    require_columns(
        scenario,
        [
            "s1_total_eur",
            "s2_total_used_eur",
            "s3_gross_no_activation_eur",
            "s3_net_scenario_mean_eur",
        ],
        "scenario_level_inputs_with_s1_s2_s3.csv",
    )

    data = [
        pd.to_numeric(scenario["s1_total_eur"], errors="coerce").dropna(),
        pd.to_numeric(scenario["s2_total_used_eur"], errors="coerce").dropna(),
        pd.to_numeric(scenario["s3_gross_no_activation_eur"], errors="coerce").dropna(),
        pd.to_numeric(scenario["s3_net_scenario_mean_eur"], errors="coerce").dropna(),
    ]

    labels = [
        "S1: DA only",
        "S2: DA + mFRR energy",
        "S3: gross full stack",
        "S3: net",
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=False,
    )

    ax.set_ylabel("Monthly revenue (€)")
    ax.set_title("Between-scenario monthly revenue distributions")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    euro_axis(ax)

    fig.tight_layout()
    fig.savefig(FIG1_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FIGURE 2: POOLED S3-NET HISTOGRAM
# =============================================================================

def plot_s3_pooled_histogram(samples: pd.DataFrame) -> None:
    require_columns(
        samples,
        [
            "s3_activation_adjusted_net_eur",
            "s2_total_eur",
            "s3_minus_s2_eur",
        ],
        "all_random_activation_samples.csv",
    )

    s3_net = pd.to_numeric(
        samples["s3_activation_adjusted_net_eur"],
        errors="coerce",
    ).dropna()

    s2_benchmark = pd.to_numeric(
        samples["s2_total_eur"],
        errors="coerce",
    ).dropna()

    delta = pd.to_numeric(
        samples["s3_minus_s2_eur"],
        errors="coerce",
    ).dropna()

    mean_s2 = float(s2_benchmark.mean())
    mean_s3 = float(s3_net.mean())
    prob_s3_ge_s2 = float((delta >= 0).mean() * 100.0)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.hist(
        s3_net,
        bins=40,
        edgecolor="black",
        alpha=0.75,
    )

    ax.axvline(
        mean_s2,
        linestyle="--",
        linewidth=2,
        label=f"Mean S2 benchmark = €{mean_s2:,.0f}",
    )

    ax.axvline(
        mean_s3,
        linestyle="-",
        linewidth=2,
        label=f"Mean S3-net = €{mean_s3:,.0f}",
    )

    ax.axvspan(
        mean_s2,
        s3_net.max(),
        alpha=0.10,
        label=f"S3 ≥ S2 region; probability = {prob_s3_ge_s2:.1f}%",
    )

    ax.set_xlabel("Activation-adjusted S3-net monthly revenue (€)")
    ax.set_ylabel("Frequency")
    ax.set_title("Pooled activation-adjusted S3-net distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    euro_xaxis(ax)

    fig.tight_layout()
    fig.savefig(FIG2_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    scenario = read_csv_required(SCENARIO_FILE)
    samples = read_csv_required(SAMPLES_FILE)

    plot_revenue_distributions(scenario)
    plot_s3_pooled_histogram(samples)

    print("Figures created successfully:")
    print(f"1) {FIG1_PATH}")
    print(f"2) {FIG2_PATH}")


if __name__ == "__main__":
    main()