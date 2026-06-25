"""
PRISM Notebook 05: Survival Modeling
======================================

Trains three survival models to predict TIME-TO-CHURN:
1. Cox Proportional Hazards (interpretable)
2. Gradient Boosted Survival Analysis (performance)
3. Random Survival Forest (ensemble)

All experiments are logged to MLflow for tracking and comparison.
Generates publication-quality visualizations.

Output:
- MLflow experiment runs (local tracking)
- data/results/survival_curves.parquet
- visualizations/survival_*.png
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.survival_models import prepare_survival_data, SurvivalTrainer


# =================================================================
# CONFIGURATION
# =================================================================

SNAPSHOT_DAY = 300
OBSERVATION_DAYS = 365
SEED = 42
VIZ_DIR = project_root / 'visualizations'
VIZ_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = project_root / 'data' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# MLflow setup (local tracking)
MLFLOW_DIR = project_root / 'mlruns'
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.resolve()}")
mlflow.set_experiment("PRISM-Survival-Models")

# Subsample for faster iteration (set to None for full data)
MAX_TRAIN_SAMPLES = 30_000

print("=" * 60)
print("  PRISM Survival Modeling Pipeline")
print(f"  MLflow tracking: local | Max train samples: {MAX_TRAIN_SAMPLES}")
print("=" * 60)


# =================================================================
# 1. LOAD DATA
# =================================================================

print("\n[1/6] Loading data...")
data_dir = project_root / 'data'

features = pd.read_parquet(data_dir / 'features' / 'churn_features.parquet')
users = pd.read_parquet(data_dir / 'raw' / 'users.parquet')

print(f"  Features: {features.shape}")
print(f"  Users: {len(users):,}")


# =================================================================
# 2. PREPARE SURVIVAL DATA
# =================================================================

print("\n[2/6] Preparing survival data...")
surv_data = prepare_survival_data(
    features_df=features,
    users_df=users,
    snapshot_day=SNAPSHOT_DAY,
    observation_days=OBSERVATION_DAYS,
    test_size=0.2,
    seed=SEED,
)

# Subsample for faster iteration
if MAX_TRAIN_SAMPLES and len(surv_data['X_train']) > MAX_TRAIN_SAMPLES:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(surv_data['X_train']), MAX_TRAIN_SAMPLES, replace=False)
    surv_data['X_train'] = surv_data['X_train'][idx]
    surv_data['y_train'] = surv_data['y_train'][idx]
    surv_data['train_df'] = surv_data['train_df'].iloc[idx].reset_index(drop=True)
    print(f"  Subsampled training data to {MAX_TRAIN_SAMPLES:,} rows")

print(f"  Train: {len(surv_data['X_train']):,} | Test: {len(surv_data['X_test']):,}")
print(f"  Features: {len(surv_data['feature_names'])}")


# =================================================================
# 3. TRAIN ALL MODELS (with MLflow tracking)
# =================================================================

print("\n[3/6] Training survival models...")
trainer = SurvivalTrainer(surv_data)

# Train each model inside an MLflow run
for model_name, train_fn, params in [
    ('cox_ph', trainer.train_cox_ph, {
        'penalizer': 0.01, 'l1_ratio': 0.1,
    }),
    ('gbsa', trainer.train_gbsa, {
        'n_estimators': 50, 'max_depth': 3,
        'learning_rate': 0.1, 'subsample': 0.8,
    }),
    ('rsf', trainer.train_rsf, {
        'n_estimators': 50, 'max_depth': 4,
    }),
]:
    with mlflow.start_run(run_name=f"survival_{model_name}"):
        result = train_fn(**params)

        # Log to MLflow
        mlflow.log_params(result['params'])
        mlflow.log_metric("concordance_index", result['concordance_index'])
        mlflow.log_metric("train_time_seconds", result['train_time'])
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("snapshot_day", SNAPSHOT_DAY)

        # Log model (scikit-survival models only)
        if model_name != 'cox_ph':
            mlflow.sklearn.log_model(result['model'], model_name)


# =================================================================
# 4. GENERATE VISUALIZATIONS
# =================================================================

print("\n[4/6] Generating visualizations...")

# ─── 4a. Model Comparison Bar Chart ──────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
model_names = list(trainer.results.keys())
c_indices = [trainer.results[m]['concordance_index'] for m in model_names]
colors = ['#2196F3', '#4CAF50', '#FF9800']

bars = ax.barh(model_names, c_indices, color=colors, height=0.5, edgecolor='white')
for bar, ci in zip(bars, c_indices):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{ci:.4f}', va='center', fontsize=12, fontweight='bold')

ax.set_xlabel('Concordance Index (Test Set)', fontsize=12)
ax.set_title('Survival Model Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0.5, max(c_indices) + 0.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(VIZ_DIR / 'survival_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: survival_model_comparison.png")

# ─── 4b. Cox PH Hazard Ratios ────────────────────────────
if 'cox_ph' in trainer.results:
    top_feat = trainer.results['cox_ph']['top_features'].head(12)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_hr = ['#e53935' if hr > 1 else '#43a047'
                 for hr in top_feat['hazard_ratio']]

    ax.barh(range(len(top_feat)), top_feat['hazard_ratio'].values,
            color=colors_hr, height=0.6, edgecolor='white')
    ax.set_yticks(range(len(top_feat)))
    ax.set_yticklabels(top_feat.index, fontsize=10)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Hazard Ratio (exp(coef))', fontsize=12)
    ax.set_title('Cox PH: Top Feature Hazard Ratios', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'survival_cox_hazard_ratios.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: survival_cox_hazard_ratios.png")

# ─── 4c. GBSA Feature Importance ─────────────────────────
if 'gbsa' in trainer.results:
    importances = trainer.results['gbsa']['feature_importances'].head(15)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(range(len(importances)), importances.values,
            color='#1976D2', height=0.6, edgecolor='white')
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index, fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('GBSA: Top 15 Feature Importances', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'survival_gbsa_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: survival_gbsa_importance.png")

# ─── 4d. Sample Survival Curves ──────────────────────────
# Pick the best scikit-survival model for survival curves
best_sk = 'gbsa' if trainer.results.get('gbsa', {}).get('concordance_index', 0) >= \
                     trainer.results.get('rsf', {}).get('concordance_index', 0) else 'rsf'

times = np.array([1, 3, 7, 14, 21, 30, 45, 60, 65])
X_test = surv_data['X_test']
y_test = surv_data['y_test']

# Predict survival curves for test users
times_pred, surv_probs = trainer.predict_survival_curves(best_sk, X_test[:500], times)

# Separate by actual outcome
events = y_test[:500]['event']
surv_probs_churned = surv_probs[events]
surv_probs_retained = surv_probs[~events]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot mean survival curve for each group
mean_churned = surv_probs_churned.mean(axis=0)
mean_retained = surv_probs_retained.mean(axis=0)
std_churned = surv_probs_churned.std(axis=0)
std_retained = surv_probs_retained.std(axis=0)

ax.plot(times, mean_retained, color='#43a047', linewidth=2.5, label='Retained users')
ax.fill_between(times, mean_retained - std_retained, mean_retained + std_retained,
                alpha=0.15, color='#43a047')

ax.plot(times, mean_churned, color='#e53935', linewidth=2.5, label='Churned users')
ax.fill_between(times, mean_churned - std_churned, mean_churned + std_churned,
                alpha=0.15, color='#e53935')

ax.set_xlabel('Days from Snapshot', fontsize=12)
ax.set_ylabel('Survival Probability S(t)', fontsize=12)
ax.set_title(f'Predicted Survival Curves ({best_sk.upper()})',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower left')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(VIZ_DIR / 'survival_curves_by_outcome.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: survival_curves_by_outcome.png")

# ─── 4e. Risk Tier Distribution ──────────────────────────
tiers = trainer.create_risk_tiers(best_sk, X_test)
tier_order = ['Low', 'Moderate', 'Elevated', 'High', 'Critical']
tier_colors = ['#4CAF50', '#8BC34A', '#FFEB3B', '#FF9800', '#f44336']

# Actual churn rate by predicted risk tier
tier_df = pd.DataFrame({
    'risk_tier': tiers,
    'event': y_test['event'],
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: tier distribution
tier_counts = tier_df['risk_tier'].value_counts().reindex(tier_order)
axes[0].bar(range(len(tier_order)), tier_counts.values,
            color=tier_colors, edgecolor='white')
axes[0].set_xticks(range(len(tier_order)))
axes[0].set_xticklabels(tier_order, fontsize=10)
axes[0].set_ylabel('Number of Users', fontsize=11)
axes[0].set_title('Risk Tier Distribution', fontsize=13, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Right: actual churn rate by tier
churn_by_tier = tier_df.groupby('risk_tier')['event'].mean().reindex(tier_order)
axes[1].bar(range(len(tier_order)), churn_by_tier.values * 100,
            color=tier_colors, edgecolor='white')
axes[1].set_xticks(range(len(tier_order)))
axes[1].set_xticklabels(tier_order, fontsize=10)
axes[1].set_ylabel('Actual Churn Rate (%)', fontsize=11)
axes[1].set_title('Churn Rate by Predicted Risk Tier', fontsize=13, fontweight='bold')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

for i, rate in enumerate(churn_by_tier.values):
    axes[1].text(i, rate * 100 + 0.5, f'{rate*100:.1f}%',
                ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(VIZ_DIR / 'survival_risk_tiers.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: survival_risk_tiers.png")


# =================================================================
# 5. SAVE SURVIVAL PREDICTIONS
# =================================================================

print("\n[5/6] Saving survival predictions...")

# Predict survival curves for ALL test users
times_full = np.array([7, 14, 30, 45, 60])
_, all_probs = trainer.predict_survival_curves(best_sk, X_test, times_full)
all_tiers = trainer.create_risk_tiers(best_sk, X_test)
all_risk_scores = trainer.get_risk_scores(best_sk, X_test)

# Build results DataFrame
test_user_ids = pd.read_parquet(
    data_dir / 'features' / 'churn_features.parquet',
    columns=['user_id']
).iloc[surv_data['test_idx']]['user_id'].values

survival_results = pd.DataFrame({
    'user_id': test_user_ids,
    'risk_score': all_risk_scores,
    'risk_tier': all_tiers,
    'survival_7d': all_probs[:, 0],
    'survival_14d': all_probs[:, 1],
    'survival_30d': all_probs[:, 2],
    'survival_45d': all_probs[:, 3],
    'survival_60d': all_probs[:, 4],
    'actual_event': y_test['event'],
    'actual_time': y_test['time'],
})

output_path = RESULTS_DIR / 'survival_curves.parquet'
survival_results.to_parquet(output_path, engine='pyarrow', index=False)
print(f"  Saved: {output_path} ({len(survival_results):,} rows)")

# Also save CSV for Power BI
csv_path = RESULTS_DIR / 'survival_curves.csv'
survival_results.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")


# =================================================================
# 6. FINAL SUMMARY
# =================================================================

print(f"\n{'='*60}")
print(f"  SURVIVAL MODELING COMPLETE")
print(f"{'='*60}")
print(f"\n  Models trained: {len(trainer.results)}")
for name, result in trainer.results.items():
    print(f"    {name:25s} C-index: {result['concordance_index']:.4f}")

best_name = max(trainer.results, key=lambda k: trainer.results[k]['concordance_index'])
best_ci = trainer.results[best_name]['concordance_index']
print(f"\n  Best model: {best_name} (C-index: {best_ci:.4f})")
print(f"\n  Visualizations saved to: {VIZ_DIR}")
print(f"  Predictions saved to: {RESULTS_DIR}")
print(f"\n  To view MLflow experiments, run:")
print(f"    mlflow ui --backend-store-uri file:///{MLFLOW_DIR.resolve()}")
print(f"\n  Next: notebooks/06_uplift_modeling.py")
print(f"{'='*60}")
