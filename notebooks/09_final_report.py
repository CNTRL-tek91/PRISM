"""
PRISM Notebook 09: Final Report + Power BI Exports
===================================================

Builds final project artifacts:
1) Executive-ready summary tables
2) Final visualizations for portfolio/reports
3) Power BI-ready flat files in data/powerbi/
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = project_root / "data" / "results"
POWERBI_DIR = project_root / "data" / "powerbi"
VIZ_DIR = project_root / "visualizations"

POWERBI_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 68)
print("  PRISM Final Report + Power BI Export Pipeline")
print("=" * 68)

print("\n[1/6] Loading model outputs...")
survival = pd.read_csv(RESULTS_DIR / "survival_curves.csv")
uplift = pd.read_csv(RESULTS_DIR / "uplift_predictions.csv")
ltv = pd.read_csv(RESULTS_DIR / "ltv_estimates.csv")
roi = pd.read_csv(RESULTS_DIR / "roi_strategy_comparison.csv")
sensitivity = pd.read_csv(RESULTS_DIR / "roi_budget_sensitivity.csv")
recommendations = pd.read_csv(RESULTS_DIR / "campaign_recommendations.csv")

print(f"  survival rows:        {len(survival):,}")
print(f"  uplift rows:          {len(uplift):,}")
print(f"  ltv rows:             {len(ltv):,}")
print(f"  recommendations rows: {len(recommendations):,}")


print("\n[2/6] Building Power BI-ready flat tables...")
uplift_cols = [c for c in uplift.columns if c.startswith("uplift_")]
best_uplift_col = max(uplift_cols, key=lambda c: float(uplift[c].mean()))

user_scores = (
    survival[["user_id", "risk_score", "risk_tier", "survival_30d", "survival_60d"]]
    .merge(ltv[["user_id", "ltv_12m", "ltv_24m", "ltv_tier_12m"]], on="user_id", how="left")
    .merge(
        uplift[["user_id", "quadrant", "campaign_type", best_uplift_col]].rename(
            columns={best_uplift_col: "best_uplift_score"}
        ),
        on="user_id",
        how="left",
    )
)
user_scores["is_targeted_prism"] = user_scores["user_id"].isin(
    recommendations[recommendations["strategy"] == "Uplift x LTV Targeted"]["user_id"]
).astype(int)

exec_kpis = pd.DataFrame(
    [
        {
            "metric": "Users scored (survival stage)",
            "value": int(len(survival)),
        },
        {
            "metric": "Users scored (uplift stage)",
            "value": int(len(uplift)),
        },
        {
            "metric": "Avg 12m LTV",
            "value": float(ltv["ltv_12m"].mean()),
        },
        {
            "metric": "Best strategy",
            "value": roi.sort_values("net_roi_usd", ascending=False).iloc[0]["strategy"],
        },
        {
            "metric": "Best strategy net ROI (USD)",
            "value": float(roi.sort_values("net_roi_usd", ascending=False).iloc[0]["net_roi_usd"]),
        },
    ]
)

user_scores.to_csv(POWERBI_DIR / "user_scores.csv", index=False)
roi.to_csv(POWERBI_DIR / "roi_strategy_comparison.csv", index=False)
sensitivity.to_csv(POWERBI_DIR / "roi_budget_sensitivity.csv", index=False)
recommendations.to_csv(POWERBI_DIR / "campaign_recommendations.csv", index=False)
exec_kpis.to_csv(POWERBI_DIR / "executive_kpis.csv", index=False)

print(f"  Saved Power BI tables to: {POWERBI_DIR}")


print("\n[3/6] Creating final executive visuals...")

# 3a. Risk distribution
risk_dist = survival["risk_tier"].value_counts().reindex(
    ["Low", "Moderate", "Elevated", "High", "Critical"], fill_value=0
)
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.bar(risk_dist.index, risk_dist.values, color=["#66BB6A", "#9CCC65", "#FDD835", "#FB8C00", "#E53935"])
ax.set_title("User Risk Tier Distribution")
ax.set_ylabel("Users")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(VIZ_DIR / "final_risk_tier_distribution.png", dpi=160, bbox_inches="tight")
plt.close()

# 3b. Quadrant distribution
quad_dist = uplift["quadrant"].value_counts().reindex(
    ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"], fill_value=0
)
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.bar(quad_dist.index, quad_dist.values, color=["#43A047", "#1E88E5", "#9E9E9E", "#E53935"])
ax.set_title("Uplift Quadrant Distribution")
ax.set_ylabel("Users")
ax.tick_params(axis="x", rotation=15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(VIZ_DIR / "final_uplift_quadrant_distribution.png", dpi=160, bbox_inches="tight")
plt.close()

# 3c. Strategy ROI
plot_roi = roi.sort_values("net_roi_usd", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.bar(plot_roi["strategy"], plot_roi["net_roi_usd"], color=["#26A69A", "#5C6BC0", "#B0BEC5"])
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("Strategy Net ROI Comparison")
ax.set_ylabel("Net ROI (USD)")
ax.tick_params(axis="x", rotation=12)
for bar, val in zip(bars, plot_roi["net_roi_usd"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val, f"${val:,.0f}", ha="center", va="bottom", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(VIZ_DIR / "final_strategy_roi.png", dpi=160, bbox_inches="tight")
plt.close()

# 3d. Budget sensitivity
fig, ax = plt.subplots(figsize=(8.5, 5))
for strategy, grp in sensitivity.groupby("strategy"):
    grp = grp.sort_values("budget_usd")
    ax.plot(grp["budget_usd"], grp["net_roi_usd"], marker="o", linewidth=2, label=strategy)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("Budget Sensitivity: Net ROI by Strategy")
ax.set_xlabel("Budget (USD)")
ax.set_ylabel("Net ROI (USD)")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(VIZ_DIR / "final_budget_sensitivity.png", dpi=160, bbox_inches="tight")
plt.close()

print("  Saved final visualization set to visualizations/")


print("\n[4/6] Creating summary markdown table...")
best = roi.sort_values("net_roi_usd", ascending=False).iloc[0]
summary_md = (
    f"# PRISM Final Summary\n\n"
    f"- Best strategy: **{best['strategy']}**\n"
    f"- Users targeted: **{int(best['users_targeted']):,}**\n"
    f"- Net ROI: **${best['net_roi_usd']:,.0f}**\n"
    f"- Average expected uplift: **{best['avg_expected_uplift']:.2%}**\n"
    f"- Best uplift score column used: **{best_uplift_col}**\n"
)
(project_root / "docs").mkdir(parents=True, exist_ok=True)
(project_root / "docs" / "results_summary.md").write_text(summary_md, encoding="utf-8")
print("  Saved: docs/results_summary.md")


print("\n[5/6] Console summary...")
print(roi.to_string(index=False))

print("\n[6/6] Complete.")
print("=" * 68)
print("  Phase 6 artifacts generated successfully.")
print("=" * 68)

