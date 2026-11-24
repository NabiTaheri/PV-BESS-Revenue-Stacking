from __future__ import annotations

from pathlib import Path
import json
import os
import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

ROOT = Path(__file__).resolve().parent
MC_ROOT = Path(os.environ.get("BALTIC_MC_ROOT", str(ROOT / "monte_carlo_results2"))).resolve()
ACT_DIR = MC_ROOT / "random_activation_sensitivity"

S1_DIR_NAME = "results_S1_DA_only"
S2_DIR_NAME = "results_S2_DA_MFRR_energy"
S3_DIR_NAME = "results_S3_reserve_competition_full_stack"

DAILY_SUMMARY_NAME = "monthly_chain_summary.csv"
TOTAL_REVENUE_COL = "total_revenue_market_actual_eur"

OUT_DIR = MC_ROOT / "paper_table_outputs_current_runs"


# =============================================================================
# HELPERS
# =============================================================================

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def sum_revenue_from_monthly_chain(run_name: str, result_dir_name: str) -> float:
    path = MC_ROOT / run_name / "results" / result_dir_name / DAILY_SUMMARY_NAME
    df = read_csv_required(path)

    if TOTAL_REVENUE_COL not in df.columns:
        raise KeyError(
            f"Column '{TOTAL_REVENUE_COL}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    return float(
        pd.to_numeric(df[TOTAL_REVENUE_COL], errors="coerce")
        .fillna(0.0)
        .sum()
    )


def stats(x: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return {
            "Mean (€)": np.nan,
            "P05 (€)": np.nan,
            "P50 (€)": np.nan,
            "P95 (€)": np.nan,
            "Std (€)": np.nan,
        }

    return {
        "Mean (€)": float(x.mean()),
        "P05 (€)": float(x.quantile(0.05)),
        "P50 (€)": float(x.quantile(0.50)),
        "P95 (€)": float(x.quantile(0.95)),
        "Std (€)": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
    }


def euro(v: float) -> str:
    if pd.isna(v):
        return "--"
    return f"{v:,.0f}"


def make_stats_table(
    items: list[tuple[str, pd.Series]],
    first_col_name: str,
) -> pd.DataFrame:
    rows = []

    for name, s in items:
        row = {first_col_name: name}
        row.update(stats(s))
        rows.append(row)

    return pd.DataFrame(rows)


def latex_table_from_df(
    df: pd.DataFrame,
    caption: str,
    label: str,
    first_col_width: str,
    first_col_name: str,
) -> str:
    numeric_cols = [c for c in df.columns if c != first_col_name]
    align = first_col_width + " " + " ".join(["r"] * len(numeric_cols))

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\renewcommand{\arraystretch}{1.20}")
    lines.append(r"\small")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")

    header = [rf"\textbf{{{first_col_name}}}"] + [
        rf"\textbf{{{c}}}" for c in numeric_cols
    ]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        vals = [str(row[first_col_name])]
        vals += [euro(row[c]) for c in numeric_cols]
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{df_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_base_path = ACT_DIR / "summary_by_base_scenario.csv"
    samples_path = ACT_DIR / "all_random_activation_samples.csv"
    config_path = ACT_DIR / "random_activation_config.json"

    by_base = read_csv_required(by_base_path)
    samples = read_csv_required(samples_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    require_columns(
        by_base,
        [
            "base_run",
            "s3_gross_no_activation_eur",
            "s3_net_mean",
        ],
        "summary_by_base_scenario.csv",
    )

    require_columns(
        samples,
        [
            "base_run",
            "s3_activation_adjusted_net_eur",
            "s3_energy_net_eur",
            "s3_reserve_capacity_gross_eur",
            "net_recovery_cost_eur",
            "degradation_eur",
            "net_activation_adjustment_eur",
        ],
        "all_random_activation_samples.csv",
    )

    # -------------------------------------------------------------------------
    # Read S1 and S2 monthly totals from raw run folders.
    # -------------------------------------------------------------------------
    s1_vals = []
    s2_vals = []

    for run_name in by_base["base_run"].astype(str):
        s1_vals.append(sum_revenue_from_monthly_chain(run_name, S1_DIR_NAME))

        try:
            s2_vals.append(sum_revenue_from_monthly_chain(run_name, S2_DIR_NAME))
        except FileNotFoundError:
            if "s2_total_eur" not in by_base.columns:
                raise

            s2_existing = by_base.loc[
                by_base["base_run"].astype(str) == run_name,
                "s2_total_eur",
            ]

            if len(s2_existing) == 0:
                raise

            s2_vals.append(float(s2_existing.iloc[0]))

    by_base["s1_total_eur"] = s1_vals
    by_base["s2_total_eur_checked"] = s2_vals
    by_base["s2_total_used_eur"] = by_base["s2_total_eur_checked"]

    by_base["s3_net_scenario_mean_eur"] = pd.to_numeric(
        by_base["s3_net_mean"],
        errors="coerce",
    )
    by_base["s3_cost_only_scenario_mean_eur"] = pd.to_numeric(
        by_base["s3_cost_only_mean"], errors="coerce"
    )

    by_base["s3_minus_s2_scenario_mean_eur"] = (
        by_base["s3_net_scenario_mean_eur"]
        - by_base["s2_total_used_eur"]
    )

    # -------------------------------------------------------------------------
    # Table 1: between-scenario distribution.
    # -------------------------------------------------------------------------
    between_table = make_stats_table(
        [
            ("S1: DA only", by_base["s1_total_eur"]),
            ("S2: DA + mFRR energy", by_base["s2_total_used_eur"]),
            ("S3: gross full stack", by_base["s3_gross_no_activation_eur"]),
            ("S3: cost-only (original)", by_base["s3_cost_only_scenario_mean_eur"]),
            (
                "S3: activation-adjusted net (scenario mean)",
                by_base["s3_net_scenario_mean_eur"],
            ),
            (
                r"$\Delta R^{S3-S2}$ (scenario mean)",
                by_base["s3_minus_s2_scenario_mean_eur"],
            ),
        ],
        first_col_name="Case",
    )

    # -------------------------------------------------------------------------
    # Table 2: pooled S3-net and paired increment.
    # -------------------------------------------------------------------------
    if "s3_minus_s2_eur" not in samples.columns:
        s2_map = by_base.set_index("base_run")["s2_total_used_eur"].to_dict()
        samples["s2_total_used_eur"] = samples["base_run"].map(s2_map)
        samples["s3_minus_s2_eur"] = (
            pd.to_numeric(samples["s3_activation_adjusted_net_eur"], errors="coerce")
            - pd.to_numeric(samples["s2_total_used_eur"], errors="coerce")
        )

    pooled_table = make_stats_table(
        [
            ("S3: cost-only (original)", samples["s3_cost_only_net_eur"]),
            ("Original cost-only paired gap", samples["s3_cost_only_minus_s2_eur"]),
            (
                "S3: activation-adjusted net",
                samples["s3_activation_adjusted_net_eur"],
            ),
            (
                r"$\Delta R^{S3-S2}_{i,j}$",
                samples["s3_minus_s2_eur"],
            ),
        ],
        first_col_name="Quantity",
    )

    prob_s3_ge_s2 = float(
        (
            pd.to_numeric(samples["s3_minus_s2_eur"], errors="coerce") >= 0
        ).mean()
        * 100.0
    )

    prob_cost_only_ge_s2 = float(
        (pd.to_numeric(samples["s3_cost_only_minus_s2_eur"], errors="coerce") >= 0).mean() * 100.0
    )

    # -------------------------------------------------------------------------
    # Table 3: S3 decomposition.
    #
    # Settlement-corrected logic:
    #   R_S3_net = R_S3_gross + R_aFRR_activation + R_mFRR_activation
    #              - C_recovery - C_degradation
    #
    # Therefore:
    #   net_recovery_cost_eur is reported as a positive cost.
    #   recovery_cost_contribution_eur = -net_recovery_cost_eur
    #   degradation_cost_contribution_eur = -degradation_eur
    # -------------------------------------------------------------------------
    samples["recovery_cost_contribution_eur"] = -pd.to_numeric(
        samples["net_recovery_cost_eur"],
        errors="coerce",
    )

    samples["degradation_cost_contribution_eur"] = -pd.to_numeric(
        samples["degradation_eur"],
        errors="coerce",
    )

    samples["activation_cashflow_contribution_eur"] = pd.to_numeric(
        samples["activation_energy_cashflow_eur"], errors="coerce",
    )
    samples["net_recovery_and_degradation_adjustment_eur"] = (
        samples["recovery_cost_contribution_eur"]
        + samples["degradation_cost_contribution_eur"]
    )
    # Check algebraic consistency before creating manuscript tables.
    target = samples["net_activation_adjustment_eur"].astype(float)
    reconstructed = (samples["activation_cashflow_contribution_eur"]
                     + samples["net_recovery_and_degradation_adjustment_eur"])
    if not np.allclose(target, reconstructed, rtol=1e-8, atol=1e-6):
        raise ValueError("Decomposition does not reconcile with S3 net adjustment")

    decomp_table = make_stats_table(
        [
            ("Energy-market net revenue in S3", samples["s3_energy_net_eur"]),
            (
                "Reserve-capacity gross revenue",
                samples["s3_reserve_capacity_gross_eur"],
            ),
            ("aFRR activation energy cashflow", samples["afrr_activation_energy_cashflow_eur"]),
            ("mFRR capacity-linked activation energy cashflow", samples["mfrr_activation_energy_cashflow_eur"]),
            (
                "Recovery cost contribution",
                samples["recovery_cost_contribution_eur"],
            ),
            (
                "Degradation cost contribution",
                samples["degradation_cost_contribution_eur"],
            ),
            (
                "Net recovery and degradation adjustment",
                samples["net_recovery_and_degradation_adjustment_eur"],
            ),
            ("S3 cost-only net (original calculation)", samples["s3_cost_only_net_eur"]),
            (
                r"\textbf{S3 activation-adjusted net revenue}",
                samples["s3_activation_adjusted_net_eur"],
            ),
        ],
        first_col_name="Component",
    )

    # -------------------------------------------------------------------------
    # Save CSV outputs.
    # -------------------------------------------------------------------------
    by_base.to_csv(
        OUT_DIR / "scenario_level_inputs_with_s1_s2_s3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    between_table.to_csv(
        OUT_DIR / "table_case_results_between.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_table.to_csv(
        OUT_DIR / "table_case_results_pooled.csv",
        index=False,
        encoding="utf-8-sig",
    )

    decomp_table.to_csv(
        OUT_DIR / "table_s3_decomposition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = {
        "n_base_runs_used": int(by_base["base_run"].nunique()),
        "n_activation_samples_total": int(len(samples)),
        "n_activation_samples_per_base_scenario_from_config": config.get(
            "n_activation_samples_per_base_scenario"
        ),
        "prob_s3_ge_s2_percent": prob_s3_ge_s2,
        "prob_s3_cost_only_ge_s2_percent": prob_cost_only_ge_s2,
        "mean_s3_minus_s2_eur": float(samples["s3_minus_s2_eur"].mean()),
        "mean_original_cost_only_s3_minus_s2_eur": float(samples["s3_cost_only_minus_s2_eur"].mean()),
        "activation_config": {
            "distribution": config.get("activation_fraction_distribution"),
            "fcr_theoretical_mean": config.get("fcr_activation_theoretical_mean"),
            "afrr_theoretical_mean": config.get("afrr_activation_theoretical_mean"),
            "mfrr_theoretical_mean": config.get("mfrr_activation_theoretical_mean"),
            "degradation_eur_per_mwh": config.get("degradation_eur_per_mwh"),
            "activation_energy_payment_included": config.get(
                "activation_energy_payment_included",
                False,
            ),
            "s3_net_formula": config.get(
                "s3_net_formula",
                "s3_net = s3_gross + activation_energy_cashflow - net_recovery_cost - degradation",
            ),
        },
    }

    with open(OUT_DIR / "paper_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # -------------------------------------------------------------------------
    # LaTeX output.
    # -------------------------------------------------------------------------
    n_s = int(by_base["base_run"].nunique())
    n_a = int(
        config.get(
            "n_activation_samples_per_base_scenario",
            len(samples) // max(n_s, 1),
        )
    )
    n_total = int(len(samples))

    caption_between = (
        rf"Monthly revenue statistics across the currently completed "
        rf"$N_s={n_s}$ forecast scenarios. For S3-net, each forecast scenario "
        rf"is first summarised by the mean of its $N_a={n_a}$ reserve-activation "
        rf"samples, and the reported statistics are then calculated across the "
        rf"resulting scenario-level values."
    )

    caption_pooled = (
        rf"Pooled distribution of activation-adjusted S3 revenue and paired "
        rf"stacking increments across the currently available "
        rf"$N_s \times N_a = {n_s} \times {n_a} = {n_total:,}$ realisations. "
        rf"The stacking increment compares each $R^{{S3,net}}_{{i,j}}$ with the "
        rf"corresponding S2 revenue from the same forecast scenario $i$."
    )

    caption_decomp = (
        rf"Decomposition of the S3 activation-adjusted result across the currently "
        rf"available $N_s \times N_a = {n_total:,}$ paired realisations. Components "
        rf"are reported as signed revenue contributions. Recovery and degradation "
        rf"are shown as negative contributions; aFRR/mFRR energy settlements are "
        rf"included using direction-specific clearing prices wherever published."
    )

    latex_between = latex_table_from_df(
        between_table,
        caption=caption_between,
        label="tab:case_results_between",
        first_col_width="p{4.6cm}",
        first_col_name="Case",
    )

    latex_pooled = latex_table_from_df(
        pooled_table,
        caption=caption_pooled,
        label="tab:case_results_pooled",
        first_col_width="p{4.2cm}",
        first_col_name="Quantity",
    )

    latex_decomp = latex_table_from_df(
        decomp_table,
        caption=caption_decomp,
        label="tab:s3_decomposition_template",
        first_col_width="p{5.0cm}",
        first_col_name="Component",
    )

    prob_text = (
        "\n\n"
        r"\noindent" "\n"
        "The probability that full stacking outperforms the day-ahead plus "
        "mFRR-energy case is reported as a paired profitability metric:\n"
        r"\begin{equation*}" "\n"
        rf"\Pr(S3 \geq S2) = {prob_s3_ge_s2:.1f}\,\%." "\n"
        r"\end{equation*}" "\n"
        rf"The metric is computed over all currently available "
        rf"$N_s \times N_a = {n_total:,}$ paired realisations by comparing each "
        rf"$R^{{S3,net}}_{{i,j}}$ with the corresponding S2 revenue from forecast "
        rf"scenario $i$." "\n\n"
        rf"The S3 outcome is further decomposed into its main revenue and cost "
        rf"components in Table~\ref{{tab:s3_decomposition_template}}. This "
        rf"decomposition distinguishes gross reserve-capacity income from the "
        rf"activation energy settlements and recovery/degradation costs that determine net "
        rf"profitability." "\n"
    )

    latex_all = "\n\n".join(
        [
            latex_between,
            latex_pooled,
            prob_text,
            latex_decomp,
        ]
    )

    (OUT_DIR / "paper_tables_latex.tex").write_text(
        latex_all,
        encoding="utf-8",
    )

    # Console output.
    print("\nDone.")
    print(f"Completed base scenarios used: {n_s}")
    print(f"Activation samples used: {n_total}")
    print(f"P(S3 >= S2): {prob_s3_ge_s2:.2f}%")
    print(f"Outputs saved to: {OUT_DIR}")

    print("\nBetween-scenario table:")
    print(between_table.to_string(index=False))

    print("\nPooled table:")
    print(pooled_table.to_string(index=False))

    print("\nS3 decomposition table:")
    print(decomp_table.to_string(index=False))


if __name__ == "__main__":
    main()