# May 2026 - PV-BESS Revenue Stacking

Reproducibility package for:

**Activation-Adjusted Revenue Stacking for Renewable-Plus-Storage Assets:
A Two-Level Monte Carlo Case Study in the Estonian Bidding Zone**

## 1. Case Study

- Bidding zone: Estonia
- Study period: May 2026
- Temporal resolution: 15 minutes
- PV capacity: 5 MW
- BESS capacity: 5 MW / 10 MWh
- Selected base scenarios: 120
- Conditional activation samples per base scenario: 1,000
- Activation regimes: Reference and Stress
- Conditional samples per regime: 120,000
- Random seeds: Results/selected_scenarios.csv

## 2. Software Environment

Python 3.11.7

Install the recorded package versions:

    python -m pip install -r requirements.txt

The optimization implementation uses PuLP/CBC and SciPy/HiGHS.

## 3. Repository Structure

- DA/: Day-ahead and mFRR energy optimization.
- MFRR&AFRR/: Reserve capacity optimization.
- Data/: Processed inputs, forecasts and settlement prices.
- Analysis/: Supplementary scientific analysis scripts.
- Results/: Archived results and Monte Carlo samples.
- final_results/: Intermediate outputs for the selected base scenarios.
- generated/: Reproduced analysis outputs.
- reproduce_may_selected.py: Selected-scenario verification and execution wrapper.

## 4. Verify Archived Results

Run the following commands from the May_2026 directory:

    python run_may_exact.py --check
    python reproduce_may_selected.py

Both checks have been successfully executed.

## 5. Reproduce Supplementary Analyses

All six analyses below have been successfully executed using
the inputs contained in this preparation package.

### Paired Joint Distribution

    python Analysis/paired_revenue_joint_distribution_may2026.py --root . --stage2 Results/Stage2 --selected-ids-file Results/selected_scenarios.csv --out generated/paired_joint --samples 1000

### Battery Degradation Sensitivity

    python Analysis/degradation_sensitivity_expost_v2_10rates.py --root . --stage2 Results/Stage2 --selected-ids-file Results/selected_scenarios.csv --out generated/degradation --samples 1000

### FCR Direction and Intensity Sensitivity

    python Analysis/fcr_direction_intensity_sensitivity_may2026.py --root . --stage2 Results/Stage2 --selected-ids-file Results/selected_scenarios.csv --out generated/fcr_direction --samples 1000

### Activation Sensitivity

    python Analysis/rebuild_may120_activation_sweep_v2.py --check
    python Analysis/rebuild_may120_activation_sweep_v2.py --run

### Convergence Evidence

    python Analysis/reference_convergence_step10.py --input May_2026=Results/Stage2 --output generated/convergence

The archived convergence assessment confirms 120 selected base
scenarios, with three consecutive successful checkpoints at
100, 110 and 120 scenarios.

The assessment uses the Reference regime to determine the
selected count and summarizes Stress on the same matched scenarios.

### SOC Recovery and Reachability

    python Analysis/soc_recovery_headroom_sensitivity_v5.py --root . --selected-ids-file Results/selected_scenarios.csv --out generated/soc_recovery --samples 1000

The reachability analysis evaluates conditional recovery feasibility
under the specified recovery-power and activation assumptions.

## 6. Verified Revenue Results

| Metric | Reference | Stress |
|---|---:|---:|
| S1 (EUR) | 25340.07 | 25340.07 |
| S2 (EUR) | 69212.67 | 69212.67 |
| S3 gross (EUR) | 128571.58 | 128571.58 |
| S3 net (EUR) | 124215.86 | 40578.41 |
| Paired revenue difference (EUR) | 55003.19 | -28634.26 |
| Empirical probability of S3-net >= S2 (%) | 100.00 | 5.4967 |

## 7. Random Seed Reproducibility

The exact selected scenario identifiers and base seeds are
recorded in Results/selected_scenarios.csv.

Conditional activation seed formula:

    20260622 + base_seed * 1009

## 8. Full Optimization Re-execution

An optional selected-scenario execution wrapper is provided:

    python reproduce_may_selected.py --run --workers 2

This computationally intensive execution route remains to be
validated end to end.

It targets the exact selected scenario set.

## 9. Validation Status

- Archived revenue results: verified.
- Model input and activation algebra checks: passed.
- Supplementary analysis syntax checks: passed.
- All six supplementary analyses: successfully re-executed.
- Full optimization re-execution: pending validation.
- Comprehensive archived-output comparison: pending.

## 10. Data Availability

The preparation package preserves the processed inputs,
selected scenario identifiers, intermediate optimization
outputs, Monte Carlo samples and supplementary results.

Public redistribution is subject to the applicable data
and software permissions.

Generated outputs and Python cache files are excluded
from version control.




## 11. Archive and Reproduction Scope

This package documents the May 2026 case study.

The Stage2 archive contains the source evaluation records.
The exact 120 scenarios used for the reported May results
are identified in Results/selected_scenarios.csv.

The convergence scripts reproduce the archived evidence
supporting the selected scenario count.

All six supplementary analyses have been successfully
re-executed using the archived inputs provided in
this repository.

The original optimization implementation is preserved.
Legacy exploratory runners are retained where needed for
source-code provenance and internal dependencies.

Use reproduce_may_selected.py for the selected-scenario
verification and optional optimization re-execution route.

The generated directory contains reproducible local outputs
and is excluded from version control.

Public redistribution remains subject to applicable code
and data permissions.
