"""
PRISM LTV Estimation Module
===========================

Estimates expected customer lifetime value (LTV) by combining:
1) user monthly revenue
2) model-predicted survival probabilities
3) discounted cash flow assumptions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable


def _interpolate_monthly_survival(row: pd.Series, max_months: int = 24) -> np.ndarray:
    """Interpolate survival probabilities to monthly checkpoints."""
    known_days = np.array([0, 7, 14, 30, 45, 60], dtype=float)
    known_survival = np.array(
        [
            1.0,
            row["survival_7d"],
            row["survival_14d"],
            row["survival_30d"],
            row["survival_45d"],
            row["survival_60d"],
        ],
        dtype=float,
    )
    known_survival = np.clip(known_survival, 0.0, 1.0)

    monthly_days = np.arange(1, max_months + 1) * 30
    interp = np.interp(monthly_days, known_days, known_survival)

    # Tail extrapolation after day 60 using exponential decay implied by S(45) -> S(60).
    s45 = max(row["survival_45d"], 1e-6)
    s60 = max(row["survival_60d"], 1e-6)
    daily_ratio = (s60 / s45) ** (1 / 15)

    after_60 = monthly_days > 60
    if np.any(after_60):
        tail_days = monthly_days[after_60] - 60
        interp[after_60] = np.clip(row["survival_60d"] * (daily_ratio**tail_days), 0.0, 1.0)

    return interp


def estimate_ltv(
    survival_df: pd.DataFrame,
    users_df: pd.DataFrame,
    discount_rate_annual: float = 0.10,
    horizons_months: Iterable[int] = (6, 12, 24),
) -> pd.DataFrame:
    """
    Estimate discounted LTV at multiple horizons.

    Parameters
    ----------
    survival_df : DataFrame
        Must include user_id and survival probabilities at 7/14/30/45/60 days.
    users_df : DataFrame
        Must include user_id and monthly_price.
    discount_rate_annual : float
        Annual discount rate for DCF.
    horizons_months : iterable[int]
        Horizons to compute LTV for (months).
    """
    required_surv = {
        "user_id",
        "survival_7d",
        "survival_14d",
        "survival_30d",
        "survival_45d",
        "survival_60d",
    }
    required_users = {"user_id", "monthly_price"}
    missing_surv = required_surv - set(survival_df.columns)
    missing_users = required_users - set(users_df.columns)
    if missing_surv:
        raise ValueError(f"survival_df missing columns: {sorted(missing_surv)}")
    if missing_users:
        raise ValueError(f"users_df missing columns: {sorted(missing_users)}")

    df = survival_df.merge(users_df[["user_id", "monthly_price"]], on="user_id", how="left")
    df["monthly_price"] = df["monthly_price"].fillna(0.0).astype(float)

    horizons = sorted(set(int(h) for h in horizons_months))
    max_h = max(horizons)
    monthly_discount = (1 + discount_rate_annual) ** (1 / 12) - 1
    discount_factors = 1.0 / ((1.0 + monthly_discount) ** np.arange(1, max_h + 1))

    ltv_vals = {h: np.zeros(len(df), dtype=float) for h in horizons}
    expected_active_months = np.zeros(len(df), dtype=float)

    for i, row in enumerate(df.itertuples(index=False)):
        s_month = _interpolate_monthly_survival(pd.Series(row._asdict()), max_months=max_h)
        cashflow = s_month * row.monthly_price * discount_factors
        expected_active_months[i] = s_month[:12].sum()
        for h in horizons:
            ltv_vals[h][i] = cashflow[:h].sum()

    out = df[["user_id"]].copy()
    for h in horizons:
        out[f"ltv_{h}m"] = ltv_vals[h]
    out["expected_active_months_12m"] = expected_active_months
    out["monthly_price"] = df["monthly_price"].values
    out["risk_score"] = df.get("risk_score", pd.Series(np.nan, index=df.index)).values
    out["risk_tier"] = df.get("risk_tier", pd.Series("Unknown", index=df.index)).values

    # Value segmentation for dashboards.
    out["ltv_tier_12m"] = pd.qcut(
        out["ltv_12m"].rank(method="first"),
        q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )
    return out.sort_values("ltv_12m", ascending=False).reset_index(drop=True)

