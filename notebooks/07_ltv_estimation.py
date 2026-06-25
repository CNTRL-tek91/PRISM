"""
PRISM Notebook 07: LTV Estimation
=================================

Computes customer lifetime value using:
- survival probabilities from Notebook 05
- user monthly revenue from raw user table
- discounted cash flow at 6m/12m/24m horizons

Output:
- data/results/ltv_estimates.parquet
- data/results/ltv_estimates.csv
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.ltv_estimator import estimate_ltv


RESULTS_DIR = project_root / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = project_root / "data" / "raw"

print("=" * 60)
print("  PRISM LTV Estimation Pipeline")
print("=" * 60)

print("\n[1/4] Loading inputs...")
survival = pd.read_csv(RESULTS_DIR / "survival_curves.csv")
users = pd.read_parquet(RAW_DIR / "users.parquet", columns=["user_id", "monthly_price", "plan"])
print(f"  Survival rows: {len(survival):,}")
print(f"  Users rows:    {len(users):,}")

print("\n[2/4] Estimating discounted LTV...")
ltv_df = estimate_ltv(
    survival_df=survival,
    users_df=users,
    discount_rate_annual=0.10,
    horizons_months=(6, 12, 24),
)
print(f"  Output shape: {ltv_df.shape}")

print("\n[3/4] LTV summary...")
summary_cols = ["ltv_6m", "ltv_12m", "ltv_24m", "expected_active_months_12m"]
print(ltv_df[summary_cols].describe().round(2).to_string())
print("\n  Average LTV by tier (12m):")
print(
    ltv_df.groupby("ltv_tier_12m", observed=False)["ltv_12m"]
    .mean()
    .round(2)
    .to_string()
)

print("\n[4/4] Saving outputs...")
parquet_path = RESULTS_DIR / "ltv_estimates.parquet"
csv_path = RESULTS_DIR / "ltv_estimates.csv"
ltv_df.to_parquet(parquet_path, engine="pyarrow", index=False)
ltv_df.to_csv(csv_path, index=False)
print(f"  Saved: {parquet_path}")
print(f"  Saved: {csv_path}")

print(f"\n{'=' * 60}")
print("  LTV ESTIMATION COMPLETE")
print("  Next: notebooks/08_campaign_optimization.py")
print(f"{'=' * 60}")

