"""
PRISM Notebook 06: Uplift Modeling
====================================

Trains causal uplift models to answer the critical business question:
  "WHO should we target with retention campaigns?"

This goes beyond "who will churn" to "whose behavior CHANGES because
of our intervention" — the key differentiator of this project.

Models: T-Learner, S-Learner, Causal Forest (econml)
Evaluation: Qini curves, uplift by decile, quadrant classification
Business: ROI estimation of targeted vs. blanket campaigns

Output:
- data/results/uplift_predictions.parquet
- data/results/campaign_roi.csv
- visualizations/uplift_*.png
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import warnings
warnings.filterwarnings('ignore')

from src.feature_engineering import build_churn_features
from src.uplift_models import prepare_uplift_data, UpliftTrainer

# =================================================================
# CONFIGURATION
# =================================================================

CAMPAIGN_SNAPSHOT_DAY = 175  # Features computed before campaign (~day 182)
SEED = 42
VIZ_DIR = project_root / 'visualizations'
RESULTS_DIR = project_root / 'data' / 'results'
MLFLOW_DIR = project_root / 'mlruns'

mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.resolve()}")
mlflow.set_experiment("PRISM-Uplift-Models")

print("=" * 60)
print("  PRISM Uplift Modeling Pipeline")
print("  'Who should we actually target?'")
print("=" * 60)


# =================================================================
# 1. LOAD RAW DATA
# =================================================================

print("\n[1/7] Loading raw data...")
data_dir = project_root / 'data' / 'raw'

users = pd.read_parquet(data_dir / 'users.parquet')
daily_activity = pd.read_parquet(data_dir / 'daily_activity.parquet')
subscriptions = pd.read_parquet(data_dir / 'subscriptions.parquet')
support_tickets = pd.read_parquet(data_dir / 'support_tickets.parquet')
campaigns = pd.read_parquet(data_dir / 'campaigns.parquet')
ground_truth = pd.read_parquet(data_dir / 'ground_truth.parquet')

print(f"  Campaign users: {len(campaigns):,}")
print(f"  Treatment rate: {campaigns['treatment'].mean():.1%}")


# =================================================================
# 2. COMPUTE FEATURES AT CAMPAIGN TIME
# =================================================================

print("\n[2/7] Computing features at campaign snapshot (day {})...".format(
    CAMPAIGN_SNAPSHOT_DAY))

features_campaign = build_churn_features(
    users_df=users,
    activity_df=daily_activity,
    subscriptions_df=subscriptions,
    tickets_df=support_tickets,
    snapshot_day=CAMPAIGN_SNAPSHOT_DAY,
    target_window_days=30,
    start_date=pd.Timestamp('2024-01-01'),
)

print(f"  Feature table: {features_campaign.shape}")


# =================================================================
# 3. PREPARE UPLIFT DATA
# =================================================================

print("\n[3/7] Preparing uplift data...")
uplift_data = prepare_uplift_data(
    features_df=features_campaign,
    campaigns_df=campaigns,
    ground_truth_df=ground_truth,
    test_size=0.3,
    seed=SEED,
)


# =================================================================
# 4. TRAIN UPLIFT MODELS
# =================================================================

print("\n[4/7] Training uplift models...")
trainer = UpliftTrainer(uplift_data)

# Train each model inside MLflow
for model_name, train_fn in [
    ('t_learner', trainer.train_t_learner),
    ('s_learner', trainer.train_s_learner),
    ('uplift_random_forest', trainer.train_uplift_random_forest),
    ('x_learner', trainer.train_x_learner),
    ('causal_forest', trainer.train_causal_forest),
]:
    try:
        with mlflow.start_run(run_name=f"uplift_{model_name}"):
            result = train_fn()

            mlflow.log_metric("qini_coefficient", result['metrics']['qini_coefficient'])
            mlflow.log_metric("auuc", result['metrics']['auuc'])
            mlflow.log_metric("mean_uplift", float(result['uplift_scores'].mean()))
            mlflow.log_metric("train_time_seconds", result['train_time'])
            mlflow.set_tag("model_type", model_name)
    except Exception as exc:
        print(f"  Skipped {model_name}: {exc}")


# =================================================================
# 5. GENERATE VISUALIZATIONS
# =================================================================

print("\n[5/7] Generating visualizations...")

# ─── 5a. Uplift by Decile (all models) ────────────────────
model_names = list(trainer.results.keys())
pretty_names = {
    't_learner': 'T-Learner',
    's_learner': 'S-Learner',
    'uplift_random_forest': 'Uplift RF',
    'x_learner': 'X-Learner',
    'causal_forest': 'Causal Forest',
}
model_labels = [pretty_names.get(name, name) for name in model_names]
model_colors = ['#2196F3', '#4CAF50', '#8E24AA', '#00ACC1', '#FF9800']
fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 5))
if len(model_names) == 1:
    axes = [axes]

for ax, name, label, color in zip(axes, model_names, model_labels, model_colors):
    deciles = trainer.results[name]['metrics']['uplift_by_decile']
    x = range(len(deciles))

    bars = ax.bar(x, deciles['actual_uplift'].values * 100,
                  color=color, alpha=0.8, edgecolor='white')

    # Color negative bars red
    for bar, val in zip(bars, deciles['actual_uplift'].values):
        if val < 0:
            bar.set_color('#e53935')

    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Uplift Score Decile', fontsize=11)
    ax.set_ylabel('Actual Uplift (pp)', fontsize=11)
    ax.set_title(f'{label}', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Actual Uplift by Predicted Score Decile',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(VIZ_DIR / 'uplift_by_decile.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: uplift_by_decile.png")


# ─── 5b. Qini Comparison ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
qini_values = [trainer.results[m]['metrics']['qini_coefficient'] for m in model_names]

bars = ax.barh(model_labels, qini_values, color=model_colors,
               height=0.5, edgecolor='white')
for bar, q in zip(bars, qini_values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{q:.4f}', va='center', fontsize=12, fontweight='bold')

ax.set_xlabel('Qini Coefficient', fontsize=12)
ax.set_title('Uplift Model Comparison (Qini)', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(VIZ_DIR / 'uplift_qini_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: uplift_qini_comparison.png")


# ─── 5c. AUUC Comparison ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
auuc_values = [trainer.results[m]['metrics']['auuc'] for m in model_names]
bars = ax.barh(model_labels, auuc_values, color=model_colors[:len(model_names)],
               height=0.5, edgecolor='white')
for bar, a in zip(bars, auuc_values):
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{a:.4f}', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('AUUC', fontsize=12)
ax.set_title('Uplift Model Comparison (AUUC)', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(VIZ_DIR / 'uplift_auuc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: uplift_auuc_comparison.png")


# ─── 5d. Quadrant Analysis (T-Learner) ───────────────────
quadrants = trainer.classify_quadrants('t_learner')
quad_order = ['Persuadable', 'Sure Thing', 'Lost Cause', 'Sleeping Dog']
quad_colors = ['#4CAF50', '#2196F3', '#9E9E9E', '#f44336']
quad_icons = ['Target these!', "Don't waste $", 'Cannot save', 'Avoid!']

quad_counts = pd.Series(quadrants).value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Quadrant distribution
counts = [quad_counts.get(q, 0) for q in quad_order]
bars = axes[0].bar(range(len(quad_order)), counts,
                   color=quad_colors, edgecolor='white')
axes[0].set_xticks(range(len(quad_order)))
axes[0].set_xticklabels(quad_order, fontsize=10, rotation=15)
axes[0].set_ylabel('Number of Users', fontsize=11)
axes[0].set_title('User Quadrant Distribution', fontsize=13, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

for i, (count, icon) in enumerate(zip(counts, quad_icons)):
    axes[0].text(i, count + max(counts)*0.02, icon,
                ha='center', fontsize=9, fontstyle='italic', color='#555')

# Right: Churn rate by quadrant (treated vs control)
quad_df = pd.DataFrame({
    'quadrant': quadrants,
    'treatment': uplift_data['T_test'],
    'churned': uplift_data['Y_test'],
})

quad_churn = quad_df.groupby(['quadrant', 'treatment'])['churned'].mean().unstack()
quad_churn = quad_churn.reindex(quad_order)

x = range(len(quad_order))
w = 0.35
if 0 in quad_churn.columns:
    axes[1].bar([i - w/2 for i in x], quad_churn[0].values * 100,
                width=w, color='#78909C', label='Control', edgecolor='white')
if 1 in quad_churn.columns:
    axes[1].bar([i + w/2 for i in x], quad_churn[1].values * 100,
                width=w, color='#26A69A', label='Treated', edgecolor='white')

axes[1].set_xticks(x)
axes[1].set_xticklabels(quad_order, fontsize=10, rotation=15)
axes[1].set_ylabel('Churn Rate (%)', fontsize=11)
axes[1].set_title('Treatment Effect by Quadrant', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(VIZ_DIR / 'uplift_quadrants.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: uplift_quadrants.png")


# ─── 5e. Uplift Calibration (Best Model) ─────────────────
best_for_cal = max(trainer.results, key=lambda k: trainer.results[k]['metrics']['qini_coefficient'])
cal_df = trainer.results[best_for_cal]['metrics']['uplift_calibration']
if len(cal_df) > 0:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cal_df['predicted_uplift'], cal_df['observed_uplift'],
               color='#1565C0', s=50, alpha=0.85)
    min_v = min(cal_df['predicted_uplift'].min(), cal_df['observed_uplift'].min())
    max_v = max(cal_df['predicted_uplift'].max(), cal_df['observed_uplift'].max())
    ax.plot([min_v, max_v], [min_v, max_v], '--', color='gray', linewidth=1.5)
    for _, row in cal_df.iterrows():
        ax.text(row['predicted_uplift'], row['observed_uplift'],
                f"  bin {int(row['bin'])}", fontsize=8)
    ax.set_xlabel('Predicted Uplift', fontsize=11)
    ax.set_ylabel('Observed Uplift', fontsize=11)
    ax.set_title(f'Uplift Calibration ({pretty_names.get(best_for_cal, best_for_cal)})',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'uplift_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: uplift_calibration.png")


# ─── 5f. Ground Truth Correlation ────────────────────────
if 'persuadability_test' in uplift_data:
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]

    for ax, name, label, color in zip(axes, model_names, model_labels, model_colors):
        uplift_scores = trainer.results[name]['uplift_scores']
        persuadability = uplift_data['persuadability_test']

        # Remove NaN
        valid = ~np.isnan(persuadability)
        if valid.sum() > 0:
            ax.scatter(persuadability[valid], uplift_scores[valid],
                      alpha=0.1, s=5, color=color)

            # Correlation
            corr = np.corrcoef(persuadability[valid], uplift_scores[valid])[0, 1]
            ax.set_title(f'{label} (r = {corr:.3f})',
                        fontsize=13, fontweight='bold')

            # Trend line
            z = np.polyfit(persuadability[valid], uplift_scores[valid], 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, 1, 100)
            ax.plot(x_line, p(x_line), color='red', linewidth=2, linestyle='--')

        ax.set_xlabel('True Persuadability', fontsize=11)
        ax.set_ylabel('Predicted Uplift', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Predicted Uplift vs. Ground Truth Persuadability',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'uplift_vs_ground_truth.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: uplift_vs_ground_truth.png")


# =================================================================
# 6. BUSINESS ROI ESTIMATION
# =================================================================

print("\n[6/7] Computing business ROI...")

# Assume:
# - Average customer LTV: $120/year ($10/month)
# - Cost per retention intervention: $15 (discount + email + ops)
# - Campaign budget: $500,000
# - Universe: the full test set

n_test = len(uplift_data['T_test'])
campaign_cost = 15.0
annual_ltv = 120.0

# Strategy 1: Blanket campaign (target everyone)
blanket_saved = uplift_data['Y_test'][uplift_data['T_test'] == 0].mean() - \
               uplift_data['Y_test'][uplift_data['T_test'] == 1].mean()
blanket_revenue_saved = blanket_saved * n_test * annual_ltv
blanket_cost = n_test * campaign_cost
blanket_roi = blanket_revenue_saved - blanket_cost

# Strategy 2: Targeted (only Persuadable users)
persuadable_mask = quadrants == 'Persuadable'
n_targeted = persuadable_mask.sum()

# Uplift for persuadable users
if n_targeted > 0:
    t_learner = trainer.models['t_learner']
    p_t, p_c = t_learner.predict_proba_both(uplift_data['X_test'][persuadable_mask])
    targeted_uplift = (p_c - p_t).mean()
    targeted_revenue_saved = targeted_uplift * n_targeted * annual_ltv
    targeted_cost = n_targeted * campaign_cost
    targeted_roi = targeted_revenue_saved - targeted_cost
else:
    targeted_uplift = 0
    targeted_revenue_saved = 0
    targeted_cost = 0
    targeted_roi = 0

roi_df = pd.DataFrame([
    {
        'Strategy': 'Blanket Campaign (all users)',
        'Users Targeted': n_test,
        'Est. Churns Prevented': int(blanket_saved * n_test),
        'Revenue Saved ($)': f"{blanket_revenue_saved:,.0f}",
        'Campaign Cost ($)': f"{blanket_cost:,.0f}",
        'Net ROI ($)': f"{blanket_roi:,.0f}",
    },
    {
        'Strategy': 'Targeted (Persuadable only)',
        'Users Targeted': n_targeted,
        'Est. Churns Prevented': int(targeted_uplift * n_targeted),
        'Revenue Saved ($)': f"{targeted_revenue_saved:,.0f}",
        'Campaign Cost ($)': f"{targeted_cost:,.0f}",
        'Net ROI ($)': f"{targeted_roi:,.0f}",
    },
])

print("\n  CAMPAIGN ROI COMPARISON")
print("  " + "-"*65)
print(roi_df.to_string(index=False))
print("  " + "-"*65)

if targeted_roi > blanket_roi:
    improvement = targeted_roi - blanket_roi
    print(f"\n  Targeted strategy outperforms blanket by ${improvement:,.0f}")
else:
    print(f"\n  ROI comparison completed.")

roi_df.to_csv(RESULTS_DIR / 'campaign_roi.csv', index=False)


# =================================================================
# 7. SAVE PREDICTIONS
# =================================================================

print("\n[7/7] Saving uplift predictions...")

# Use best model's predictions
best_model = max(trainer.results,
                 key=lambda k: trainer.results[k]['metrics']['qini_coefficient'])

test_df = uplift_data['full_df'].iloc[uplift_data['test_idx']].copy()
pred_df = pd.DataFrame({
    'user_id': test_df['user_id'].values,
    'treatment': uplift_data['T_test'],
    'outcome': uplift_data['Y_test'],
    'quadrant': quadrants,
    'campaign_type': test_df['campaign_type'].values,
})

for model_name in model_names:
    pred_df[f'uplift_{model_name}'] = trainer.results[model_name]['uplift_scores']

if 'persuadability_test' in uplift_data:
    pred_df['true_persuadability'] = uplift_data['persuadability_test']

output_path = RESULTS_DIR / 'uplift_predictions.parquet'
pred_df.to_parquet(output_path, engine='pyarrow', index=False)
print(f"  Saved: {output_path} ({len(pred_df):,} rows)")

csv_path = RESULTS_DIR / 'uplift_predictions.csv'
pred_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")


# =================================================================
# FINAL SUMMARY
# =================================================================

print(f"\n{'='*60}")
print(f"  UPLIFT MODELING COMPLETE")
print(f"{'='*60}")

print(f"\n  Models trained: {len(trainer.results)}")
for name, result in trainer.results.items():
    qini = result['metrics']['qini_coefficient']
    print(f"    {name:<25s} Qini: {qini:.4f}")

print(f"\n  Best model: {best_model}")
print(f"\n  Quadrant distribution:")
for q in quad_order:
    count = (quadrants == q).sum()
    pct = count / len(quadrants) * 100
    print(f"    {q:<15s}: {count:,} ({pct:.1f}%)")

print(f"\n  Visualizations: {VIZ_DIR}")
print(f"  Predictions: {RESULTS_DIR}")
print(f"\n  Next: Power BI dashboard + final presentation")
print(f"{'='*60}")
