"""
PRISM Notebook 08: Campaign Optimization
========================================

Builds budget-constrained retention strategy and compares ROI:
1) Random targeting
2) Churn-targeted targeting
3) Uplift x LTV targeted targeting (PRISM strategy)

Output:
- data/results/campaign_recommendations.parquet
- data/results/campaign_recommendations.csv
- data/results/roi_strategy_comparison.csv
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.campaign_optimizer import optimize_campaign_targets, roi_sensitivity_by_budget


RESULTS_DIR = project_root / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR = project_root / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

BUDGET_USD = 250_000
COST_PER_INTERVENTION = 15
BUDGET_GRID = [25_000, 50_000, 100_000, 150_000, 250_000, 350_000, 500_000]

print("=" * 60)
print("  PRISM Campaign Optimization Pipeline")
print("=" * 60)

print("\n[1/5] Loading inputs...")
ltv = pd.read_csv(RESULTS_DIR / "ltv_estimates.csv")
uplift = pd.read_csv(RESULTS_DIR / "uplift_predictions.csv")
survival = pd.read_csv(RESULTS_DIR / "survival_curves.csv")
print(f"  LTV rows:      {len(ltv):,}")
print(f"  Uplift rows:   {len(uplift):,}")
print(f"  Survival rows: {len(survival):,}")

print("\n[2/5] Optimizing campaign targets...")
recs, roi = optimize_campaign_targets(
    ltv_df=ltv,
    uplift_df=uplift,
    survival_df=survival,
    budget=BUDGET_USD,
    intervention_cost=COST_PER_INTERVENTION,
    seed=42,
)

print("\n[3/5] ROI comparison...")
print(roi.to_string(index=False))

print("\n[4/5] Saving outputs...")
recs_parquet = RESULTS_DIR / "campaign_recommendations.parquet"
recs_csv = RESULTS_DIR / "campaign_recommendations.csv"
roi_csv = RESULTS_DIR / "roi_strategy_comparison.csv"

recs.to_parquet(recs_parquet, engine="pyarrow", index=False)
recs.to_csv(recs_csv, index=False)
roi.to_csv(roi_csv, index=False)
print(f"  Saved: {recs_parquet}")
print(f"  Saved: {recs_csv}")
print(f"  Saved: {roi_csv}")

print("\n[5/5] Generating ROI visualization...")
plot_df = roi.sort_values("net_roi_usd", ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(plot_df["strategy"], plot_df["net_roi_usd"], color=["#26A69A", "#5C6BC0", "#B0BEC5"])
ax.set_ylabel("Net ROI (USD)")
ax.set_title("Retention Strategy Net ROI Comparison")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", rotation=12)
for bar, val in zip(bars, plot_df["net_roi_usd"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val, f"${val:,.0f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
fig_path = VIZ_DIR / "roi_strategy_comparison.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

# Budget sensitivity curve (required for business recommendation)
sensitivity = roi_sensitivity_by_budget(
    ltv_df=ltv,
    uplift_df=uplift,
    survival_df=survival,
    budgets=BUDGET_GRID,
    intervention_cost=COST_PER_INTERVENTION,
    seed=42,
)
sensitivity_csv = RESULTS_DIR / "roi_budget_sensitivity.csv"
sensitivity.to_csv(sensitivity_csv, index=False)
print(f"  Saved: {sensitivity_csv}")

fig, ax = plt.subplots(figsize=(9, 5))
for strategy, sub in sensitivity.groupby("strategy"):
    sub = sub.sort_values("budget_usd")
    ax.plot(sub["budget_usd"], sub["net_roi_usd"], marker="o", linewidth=2, label=strategy)
ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
ax.set_title("Budget vs Net ROI by Targeting Strategy")
ax.set_xlabel("Campaign Budget (USD)")
ax.set_ylabel("Net ROI (USD)")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
sens_fig = VIZ_DIR / "roi_budget_sensitivity.png"
fig.savefig(sens_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {sens_fig}")

print(f"\n{'=' * 60}")
print("  CAMPAIGN OPTIMIZATION COMPLETE")
print("  Phase 5 delivered: LTV + Optimization + ROI comparison")
print(f"{'=' * 60}")

