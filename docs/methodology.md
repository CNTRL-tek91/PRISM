# PRISM Methodology

## Objective

Build a decision-ready retention analytics system that estimates:
1. churn risk over time,
2. causal intervention effect (uplift),
3. expected customer value,
4. strategy-level financial outcomes.

## Data generation and validation

- Synthetic subscription data generated with realistic churn dynamics:
  - engagement decay prior to churn,
  - support ticket escalation,
  - payment failures,
  - hidden persuadability for uplift evaluation.
- Validation checks include:
  - schema and null checks,
  - referential integrity,
  - realistic distribution constraints,
  - treatment ratio sanity checks.

## Feature engineering

Features are computed from a fixed **snapshot date** to avoid leakage.

Feature groups:
- recent engagement windows (3/7/14/30/60 days),
- trend features (engagement ratio, duration ratio),
- recency and consistency,
- subscription/billing behaviors,
- support and cancellation signal,
- derived interactions (e.g., sessions per dollar, support escalation).

## Survival modeling stage

Purpose: estimate `P(active at t | X)` and time-to-churn risk tiers.

Models:
- Cox Proportional Hazards (interpretability),
- Gradient Boosted Survival Analysis (performance),
- Random Survival Forest (non-linear ensemble).

Outputs:
- per-user survival probabilities at 7/14/30/45/60 days,
- risk score and risk tiers.

## Uplift modeling stage

Purpose: estimate who changes behavior under intervention, not just who is high risk.

Models:
- T-Learner,
- S-Learner,
- Uplift Random Forest,
- X-Learner,
- Causal Forest.

Primary evaluation:
- Qini coefficient,
- AUUC,
- uplift-by-decile and calibration curves,
- quadrant segmentation (Persuadable / Sure Thing / Lost Cause / Sleeping Dog).

## LTV estimation

LTV is computed via discounted expected revenue:

`LTV_h = sum_{m=1..h} [Survival(m) * MonthlyPrice * DiscountFactor(m)]`

Where:
- survival is interpolated to monthly steps using model outputs,
- discounting uses annual rate transformed to monthly.

## Campaign optimization

Three strategies are compared under budget constraints:
- Random targeting,
- Churn-targeted targeting,
- Uplift x LTV targeting (PRISM strategy).

For each targeted user:
- expected value = `(expected_uplift * ltv_12m) - intervention_cost`

Outputs:
- ranked target recommendations,
- strategy-level ROI table,
- budget sensitivity curves.

## Assumptions and caveats

- Data is simulated for portfolio realism, not production truth.
- ROI magnitude is sensitive to intervention cost and uplift calibration.
- Negative ROI at current assumptions does not invalidate the model; it indicates economic threshold misalignment.

## Reproducibility

All phases can be rerun through notebook scripts in sequence:

`02 -> 03 -> 05 -> 06 -> 07 -> 08 -> 09`

Core result artifacts are exported to:
- `data/results/` (model outputs),
- `data/powerbi/` (dashboard-ready tables),
- `visualizations/` (final figures).

