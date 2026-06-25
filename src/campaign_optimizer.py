"""
PRISM Campaign Optimization Module
==================================

Builds budget-constrained targeting recommendations and compares ROI for:
1) random targeting
2) churn-targeted targeting
3) uplift * LTV targeting
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pick_best_uplift_column(df: pd.DataFrame) -> str:
    uplift_cols = [c for c in df.columns if c.startswith("uplift_")]
    if not uplift_cols:
        raise ValueError("No uplift_* columns found in uplift predictions data.")
    return max(uplift_cols, key=lambda c: float(df[c].mean()))


def optimize_campaign_targets(
    ltv_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    survival_df: pd.DataFrame,
    budget: float = 250_000.0,
    intervention_cost: float = 15.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (target_recommendations_df, roi_comparison_df).
    """
    if intervention_cost <= 0:
        raise ValueError("intervention_cost must be > 0")
    n_target = int(budget // intervention_cost)
    if n_target <= 0:
        raise ValueError("Budget is too small to target any users.")

    base = (
        ltv_df[["user_id", "ltv_12m"]]
        .merge(uplift_df, on="user_id", how="inner")
        .merge(survival_df[["user_id", "risk_score", "risk_tier"]], on="user_id", how="left")
    )
    best_uplift_col = _pick_best_uplift_column(base)
    base["best_uplift"] = base[best_uplift_col].astype(float)
    base["risk_score"] = base["risk_score"].fillna(base["risk_score"].median())
    base["risk_tier"] = base["risk_tier"].fillna("Unknown")

    treated = base[base["treatment"] == 1]["outcome"]
    control = base[base["treatment"] == 0]["outcome"]
    baseline_uplift = float(control.mean() - treated.mean()) if len(treated) and len(control) else 0.0

    rng = np.random.default_rng(seed)

    random_sel = base.sample(n=min(n_target, len(base)), random_state=seed).copy()
    churn_sel = base.nlargest(n_target, "risk_score").copy()
    uplift_sel = base.nlargest(n_target, "best_uplift").copy()

    random_sel["expected_uplift"] = baseline_uplift
    # Risk-only strategy assumes uniform uplift scaled by churn risk proxy.
    # This keeps the baseline realistic and different from random.
    risk_scale = (churn_sel["risk_score"] / max(base["risk_score"].mean(), 1e-9)).clip(0.5, 2.0)
    churn_sel["expected_uplift"] = baseline_uplift * risk_scale
    uplift_sel["expected_uplift"] = uplift_sel["best_uplift"].clip(lower=0.0)

    random_sel["strategy"] = "Random Targeting"
    churn_sel["strategy"] = "Churn-Targeted"
    uplift_sel["strategy"] = "Uplift x LTV Targeted"

    recommendations = pd.concat([random_sel, churn_sel, uplift_sel], axis=0, ignore_index=True)
    recommendations["expected_value_per_user"] = (
        recommendations["expected_uplift"] * recommendations["ltv_12m"] - intervention_cost
    )
    recommendations["target_rank"] = recommendations.groupby("strategy")[
        "expected_value_per_user"
    ].rank(method="first", ascending=False).astype(int)

    roi_rows = []
    for strategy, sub in recommendations.groupby("strategy"):
        users_targeted = int(len(sub))
        revenue_saved = float((sub["expected_uplift"] * sub["ltv_12m"]).sum())
        campaign_cost = users_targeted * intervention_cost
        net_roi = revenue_saved - campaign_cost
        roi_rows.append(
            {
                "strategy": strategy,
                "users_targeted": users_targeted,
                "avg_expected_uplift": float(sub["expected_uplift"].mean()),
                "revenue_saved_usd": round(revenue_saved, 2),
                "campaign_cost_usd": round(campaign_cost, 2),
                "net_roi_usd": round(net_roi, 2),
                "roi_ratio": round(revenue_saved / campaign_cost, 4) if campaign_cost > 0 else np.nan,
            }
        )

    roi_df = pd.DataFrame(roi_rows).sort_values("net_roi_usd", ascending=False).reset_index(drop=True)
    recommendations = recommendations.sort_values(["strategy", "target_rank"]).reset_index(drop=True)
    recommendations = recommendations[
        [
            "strategy",
            "target_rank",
            "user_id",
            "risk_tier",
            "risk_score",
            "ltv_12m",
            "best_uplift",
            "expected_uplift",
            "expected_value_per_user",
            "quadrant",
            "campaign_type",
            "treatment",
            "outcome",
        ]
    ]
    return recommendations, roi_df


def roi_sensitivity_by_budget(
    ltv_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    survival_df: pd.DataFrame,
    budgets: list[float],
    intervention_cost: float = 15.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute strategy ROI at multiple budget levels."""
    rows = []
    for b in budgets:
        _, roi = optimize_campaign_targets(
            ltv_df=ltv_df,
            uplift_df=uplift_df,
            survival_df=survival_df,
            budget=float(b),
            intervention_cost=intervention_cost,
            seed=seed,
        )
        roi = roi.copy()
        roi["budget_usd"] = float(b)
        rows.append(roi)
    return pd.concat(rows, ignore_index=True)

