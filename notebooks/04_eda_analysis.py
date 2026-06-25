"""
PRISM Notebook 04: EDA Analysis
===============================

Performs exploratory analysis on engineered churn features and saves
portfolio-ready visuals for business storytelling.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


FEATURES_PATH = project_root / "data" / "features" / "churn_features.parquet"
VIZ_DIR = project_root / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PRISM EDA Pipeline")
print("=" * 60)

print("\n[1/5] Loading features...")
df = pd.read_parquet(FEATURES_PATH)
print(f"  Shape: {df.shape}")
print(f"  Churn rate: {df['churned_target'].mean():.2%}")

print("\n[2/5] Churn by segment/plan/platform...")
for col in ["segment", "plan", "platform"]:
    if col in df.columns:
        summary = df.groupby(col)["churned_target"].mean().sort_values(ascending=False)
        print(f"\n  Churn rate by {col}:")
        print(summary.round(4).to_string())

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(summary.index.astype(str), summary.values * 100, color="#42A5F5")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title(f"Churn Rate by {col.title()}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(VIZ_DIR / f"eda_churn_by_{col}.png", dpi=150, bbox_inches="tight")
        plt.close()

print("\n[3/5] Feature distributions (churned vs retained)...")
candidate_features = [
    "days_since_last_activity",
    "total_sessions_30d",
    "engagement_trend_ratio",
    "tickets_last_30d",
    "n_payment_failures",
    "tenure_at_snapshot",
]
features = [c for c in candidate_features if c in df.columns]

for feat in features:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.kdeplot(data=df[df["churned_target"] == 0], x=feat, label="Retained", fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(data=df[df["churned_target"] == 1], x=feat, label="Churned", fill=True, alpha=0.3, ax=ax)
    ax.set_title(f"{feat} distribution by churn status")
    ax.legend()
    plt.tight_layout()
    fig.savefig(VIZ_DIR / f"eda_dist_{feat}.png", dpi=150, bbox_inches="tight")
    plt.close()

print("\n[4/5] Correlation heatmap...")
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
focus_cols = [c for c in numeric_cols if c != "user_id"][:20]
if "churned_target" not in focus_cols:
    focus_cols.append("churned_target")
corr = df[focus_cols].corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2, ax=ax)
ax.set_title("Feature Correlation Heatmap (Top Numeric Columns)")
plt.tight_layout()
fig.savefig(VIZ_DIR / "eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n[5/5] Saved EDA outputs to visualizations/")
print(f"{'=' * 60}")
print("  EDA complete!")
print("  Next: notebooks/05_survival_modeling.py")
print(f"{'=' * 60}")

