"""
PRISM Notebook 03: Feature Engineering
=======================================

Runs the feature engineering pipeline on the simulated data.
Produces the churn_features table used for all downstream modeling.

This notebook runs locally with Pandas. For Databricks, use the
SQL script: sql/02_feature_engineering.sql

Output: data/features/churn_features.parquet
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.feature_engineering import (
    build_churn_features,
    get_feature_columns,
    get_feature_summary,
)

# =================================================================
# 1. LOAD RAW DATA
# =================================================================

print("Loading raw data from Parquet files...")
data_dir = project_root / 'data' / 'raw'

users = pd.read_parquet(data_dir / 'users.parquet')
daily_activity = pd.read_parquet(data_dir / 'daily_activity.parquet')
subscriptions = pd.read_parquet(data_dir / 'subscriptions.parquet')
support_tickets = pd.read_parquet(data_dir / 'support_tickets.parquet')

print(f"  Users: {len(users):,}")
print(f"  Daily activity: {len(daily_activity):,}")
print(f"  Subscriptions: {len(subscriptions):,}")
print(f"  Support tickets: {len(support_tickets):,}")


# =================================================================
# 2. RUN FEATURE ENGINEERING
# =================================================================

# Snapshot at day 300 (Oct 27, 2024), predict churn in next 30 days
features = build_churn_features(
    users_df=users,
    activity_df=daily_activity,
    subscriptions_df=subscriptions,
    tickets_df=support_tickets,
    snapshot_day=300,
    target_window_days=30,
    start_date=pd.Timestamp('2024-01-01'),
)


# =================================================================
# 3. FEATURE SUMMARY
# =================================================================

feat_cols = get_feature_columns(features)
print(f"\n{'='*60}")
print(f"  FEATURE SUMMARY ({len(feat_cols)} features)")
print(f"{'='*60}")

summary = get_feature_summary(features)
print(summary.to_string())


# =================================================================
# 4. TARGET DISTRIBUTION
# =================================================================

print(f"\n{'='*60}")
print(f"  TARGET DISTRIBUTION")
print(f"{'='*60}")
target_dist = features['churned_target'].value_counts()
print(f"  Not churned (0): {target_dist.get(0, 0):,}")
print(f"  Churned (1):     {target_dist.get(1, 0):,}")
print(f"  Churn rate:      {features['churned_target'].mean():.2%}")


# =================================================================
# 5. FEATURE DISTRIBUTIONS BY CHURN STATUS
# =================================================================

print(f"\n{'='*60}")
print(f"  FEATURE MEANS BY CHURN STATUS")
print(f"{'='*60}")

key_features = [
    'days_since_last_activity',
    'total_sessions_30d',
    'engagement_trend_ratio',
    'total_duration_30d',
    'tickets_last_30d',
    'has_cancel_request',
    'n_payment_failures',
    'tenure_at_snapshot',
    'activity_acceleration',
]

# Filter to features that exist
key_features = [f for f in key_features if f in features.columns]

comparison = features.groupby('churned_target')[key_features].mean()
print(comparison.T.round(2).to_string())


# =================================================================
# 6. SAVE FEATURES
# =================================================================

output_dir = project_root / 'data' / 'features'
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / 'churn_features.parquet'
features.to_parquet(output_path, engine='pyarrow', index=False)
size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"\n  Saved: {output_path}")
print(f"  Size: {size_mb:.1f} MB")
print(f"  Shape: {features.shape}")


# =================================================================
# 7. SAVE CSV (for Power BI)
# =================================================================

csv_path = output_dir / 'churn_features.csv'
features.to_csv(csv_path, index=False)
csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
print(f"  Saved CSV: {csv_path} ({csv_size_mb:.1f} MB)")

print(f"\n{'='*60}")
print(f"  Feature engineering complete!")
print(f"  Next: notebooks/05_survival_modeling.py")
print(f"{'='*60}")
