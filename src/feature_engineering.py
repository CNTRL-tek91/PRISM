"""
PRISM Feature Engineering Module
=================================

Transforms raw tables (users, daily_activity, subscriptions, support_tickets)
into a single feature table for churn/survival/uplift modeling.

Design Principles
-----------------
1. SNAPSHOT APPROACH: All features are computed as of a configurable "snapshot date."
   This prevents data leakage — we never use future information to predict churn.

2. PRODUCTION-REALISTIC: The same functions can be translated 1:1 to PySpark SQL
   for Databricks. See sql/02_feature_engineering.sql for the SQL equivalent.

3. MODULAR: Each feature category is computed in its own function, making it easy
   to add/remove features and debug individual feature groups.

Feature Categories
------------------
- Engagement (recent):  Activity counts and averages at 3d/7d/14d/30d windows
- Engagement (trends):  Ratios and slopes showing engagement change direction
- Recency:              Days since last activity (strongest single predictor)
- Temporal patterns:    Weekend behavior, consistency, active day count
- Subscription:         Tenure, plan, price, renewals, payment failures
- Support:              Ticket counts, categories, satisfaction scores
- Derived/Interaction:  Cross-feature interactions (engagement per dollar, etc.)

Usage
-----
    from src.feature_engineering import build_churn_features

    features = build_churn_features(
        users_df=users,
        activity_df=daily_activity,
        subscriptions_df=subscriptions,
        tickets_df=support_tickets,
        snapshot_day=300,        # Compute features as of day 300
        target_window_days=30,  # Predict churn in next 30 days
    )
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# =================================================================
# MAIN ENTRY POINT
# =================================================================

def build_churn_features(
    users_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    subscriptions_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    snapshot_day: int = 300,
    target_window_days: int = 30,
    start_date: pd.Timestamp = pd.Timestamp('2024-01-01'),
) -> pd.DataFrame:
    """Build the complete feature table for churn prediction.

    Parameters
    ----------
    users_df : DataFrame
        Users table from data simulation.
    activity_df : DataFrame
        Daily activity table.
    subscriptions_df : DataFrame
        Subscription/billing records.
    tickets_df : DataFrame
        Support ticket records.
    snapshot_day : int
        Day offset from start_date at which to compute features.
        Only data BEFORE this day is used (no leakage).
    target_window_days : int
        Number of days after snapshot for the churn target window.
    start_date : Timestamp
        Start date of the observation period.

    Returns
    -------
    DataFrame
        One row per eligible user with all features and target column.
        Target column: 'churned_target' (1 = churned within target window).
    """
    snapshot_date = start_date + pd.Timedelta(days=snapshot_day)
    target_end_date = snapshot_date + pd.Timedelta(days=target_window_days)

    logger.info(f"Building features as of {snapshot_date.date()}")
    logger.info(f"Target window: {snapshot_date.date()} to {target_end_date.date()}")

    print(f"\n{'='*60}")
    print(f"  Feature Engineering Pipeline")
    print(f"  Snapshot: {snapshot_date.date()} | Target: next {target_window_days}d")
    print(f"{'='*60}")

    # Step 1: Base table — eligible users + target variable
    print("\n  [1/7] Creating base table (eligible users + target)...")
    base = _create_base_table(users_df, snapshot_date, target_end_date)
    print(f"         {len(base):,} eligible users "
          f"({base['churned_target'].mean():.1%} will churn)")

    # Step 2: Engagement features
    print("  [2/7] Computing engagement features (3d/7d/14d/30d windows)...")
    engagement = _compute_engagement_features(
        activity_df, base['user_id'], snapshot_date
    )

    # Step 3: Engagement trends
    print("  [3/7] Computing engagement trends & recency...")
    trends = _compute_trend_features(
        activity_df, base['user_id'], snapshot_date
    )

    # Step 4: Temporal patterns
    print("  [4/7] Computing temporal patterns...")
    temporal = _compute_temporal_features(
        activity_df, base['user_id'], snapshot_date
    )

    # Step 5: Subscription features
    print("  [5/7] Computing subscription features...")
    subscription = _compute_subscription_features(
        subscriptions_df, base['user_id'], snapshot_date
    )

    # Step 6: Support ticket features
    print("  [6/7] Computing support ticket features...")
    support = _compute_support_features(
        tickets_df, base['user_id'], snapshot_date
    )

    # Step 7: Join all, compute derived features, fill defaults
    print("  [7/7] Joining features & computing derived metrics...")
    features = base.copy()
    for feat_df in [engagement, trends, temporal, subscription, support]:
        features = features.merge(feat_df, on='user_id', how='left')

    features = _compute_derived_features(features)
    features = _fill_defaults(features)

    n_features = len([c for c in features.columns
                      if c not in ('user_id', 'churned_target')])
    print(f"\n  Result: {len(features):,} users x {n_features} features")
    print(f"  Target distribution: {features['churned_target'].value_counts().to_dict()}")
    print(f"{'='*60}")

    return features


# =================================================================
# STEP 1: BASE TABLE
# =================================================================

def _create_base_table(
    users_df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    target_end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Create base table of eligible users with target variable.

    Eligible = signed up before snapshot AND not yet churned at snapshot.
    Target = churned between snapshot and target_end_date.
    """
    # Filter: signed up before snapshot, still active at snapshot
    eligible = users_df[
        (users_df['signup_date'] <= snapshot_date)
    ].copy()

    # Determine if user churned BEFORE snapshot (exclude these)
    churned_before_snapshot = (
        eligible['churn_date'].notna() &
        (eligible['churn_date'] <= snapshot_date)
    )
    eligible = eligible[~churned_before_snapshot].copy()

    # Target: churn within the target window
    eligible['churned_target'] = (
        eligible['churn_date'].notna() &
        (eligible['churn_date'] > snapshot_date) &
        (eligible['churn_date'] <= target_end_date)
    ).astype(int)

    # Compute tenure as of snapshot
    eligible['tenure_at_snapshot'] = (
        snapshot_date - eligible['signup_date']
    ).dt.days

    # User-level attributes to keep
    base = eligible[[
        'user_id', 'signup_date', 'plan', 'monthly_price',
        'platform', 'country', 'age_bucket', 'segment',
        'tenure_at_snapshot', 'churned_target',
    ]].copy()

    # Encode categoricals
    plan_map = {'free': 0, 'basic': 1, 'premium': 2}
    base['plan_encoded'] = base['plan'].map(plan_map)

    platform_dummies = pd.get_dummies(base['platform'], prefix='platform')
    base = pd.concat([base, platform_dummies], axis=1)

    return base


# =================================================================
# STEP 2: ENGAGEMENT FEATURES
# =================================================================

def _compute_engagement_features(
    activity_df: pd.DataFrame,
    user_ids: pd.Series,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute engagement features at multiple time windows.

    Windows: 3d, 7d, 14d, 30d, 60d lookback from snapshot.
    """
    # Filter to relevant timeframe (60d lookback)
    lookback_60d = snapshot_date - pd.Timedelta(days=60)
    activity = activity_df[
        (activity_df['user_id'].isin(user_ids)) &
        (activity_df['activity_date'] <= snapshot_date) &
        (activity_df['activity_date'] > lookback_60d)
    ].copy()

    # Compute days before snapshot
    activity['days_ago'] = (snapshot_date - activity['activity_date']).dt.days

    features = pd.DataFrame({'user_id': user_ids.values})

    # Activity counts at different windows
    for window in [3, 7, 14, 30, 60]:
        window_data = activity[activity['days_ago'] <= window]
        agg = window_data.groupby('user_id').agg(
            **{f'active_days_{window}d': ('activity_date', 'nunique'),
               f'total_sessions_{window}d': ('n_sessions', 'sum'),
               f'total_duration_{window}d': ('total_duration_min', 'sum')}
        ).reset_index()
        features = features.merge(agg, on='user_id', how='left')

    # Average metrics over last 30 days
    last_30d = activity[activity['days_ago'] <= 30]
    avg_agg = last_30d.groupby('user_id').agg(
        avg_session_duration_30d=('total_duration_min', 'mean'),
        avg_sessions_per_day_30d=('n_sessions', 'mean'),
        avg_content_items_30d=('n_content_items', 'mean'),
        avg_categories_30d=('n_distinct_categories', 'mean'),
        avg_features_used_30d=('n_features_used', 'mean'),
        avg_searches_30d=('n_searches', 'mean'),
        total_shares_30d=('n_shares', 'sum'),
        total_pages_30d=('pages_viewed', 'sum'),
    ).reset_index()
    features = features.merge(avg_agg, on='user_id', how='left')

    return features


# =================================================================
# STEP 3: ENGAGEMENT TRENDS
# =================================================================

def _compute_trend_features(
    activity_df: pd.DataFrame,
    user_ids: pd.Series,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute engagement trend features showing direction of change.

    These are the most powerful churn predictors — a declining trend
    is much more predictive than a low absolute level.
    """
    lookback_60d = snapshot_date - pd.Timedelta(days=60)
    activity = activity_df[
        (activity_df['user_id'].isin(user_ids)) &
        (activity_df['activity_date'] <= snapshot_date) &
        (activity_df['activity_date'] > lookback_60d)
    ].copy()

    activity['days_ago'] = (snapshot_date - activity['activity_date']).dt.days

    # ── Recency: days since last activity ────────────────
    last_activity = activity.groupby('user_id')['days_ago'].min().reset_index()
    last_activity.columns = ['user_id', 'days_since_last_activity']

    # ── Engagement ratio: recent / older (trend direction) ──
    # sessions in last 7d / sessions in days 8-30 (normalized to same period length)
    recent_7d = activity[activity['days_ago'] <= 7].groupby('user_id')['n_sessions'].sum()
    older_8_30 = activity[
        (activity['days_ago'] > 7) & (activity['days_ago'] <= 30)
    ].groupby('user_id')['n_sessions'].sum()

    trend_ratio = pd.DataFrame({
        'user_id': recent_7d.index.union(older_8_30.index),
    })
    trend_ratio = trend_ratio.merge(
        recent_7d.rename('sessions_7d'), on='user_id', how='left'
    ).merge(
        older_8_30.rename('sessions_8_30d'), on='user_id', how='left'
    )
    trend_ratio['sessions_7d'] = trend_ratio['sessions_7d'].fillna(0)
    trend_ratio['sessions_8_30d'] = trend_ratio['sessions_8_30d'].fillna(0)

    # Normalize older period to 7-day equivalent
    trend_ratio['sessions_8_30d_normalized'] = trend_ratio['sessions_8_30d'] / (23/7)
    trend_ratio['engagement_trend_ratio'] = np.where(
        trend_ratio['sessions_8_30d_normalized'] > 0,
        trend_ratio['sessions_7d'] / trend_ratio['sessions_8_30d_normalized'],
        np.where(trend_ratio['sessions_7d'] > 0, 2.0, 0.0)
    )
    # Cap extreme values
    trend_ratio['engagement_trend_ratio'] = np.clip(
        trend_ratio['engagement_trend_ratio'], 0, 5
    )

    # ── Duration trend: avg duration last 7d vs 8-30d ──
    dur_7d = activity[activity['days_ago'] <= 7].groupby('user_id')['total_duration_min'].mean()
    dur_8_30 = activity[
        (activity['days_ago'] > 7) & (activity['days_ago'] <= 30)
    ].groupby('user_id')['total_duration_min'].mean()

    dur_trend = pd.DataFrame({'user_id': dur_7d.index.union(dur_8_30.index)})
    dur_trend = dur_trend.merge(
        dur_7d.rename('duration_7d'), on='user_id', how='left'
    ).merge(
        dur_8_30.rename('duration_8_30d'), on='user_id', how='left'
    )
    dur_trend['duration_trend_ratio'] = np.where(
        dur_trend['duration_8_30d'].fillna(0) > 0,
        dur_trend['duration_7d'].fillna(0) / dur_trend['duration_8_30d'],
        1.0
    )
    dur_trend['duration_trend_ratio'] = np.clip(
        dur_trend['duration_trend_ratio'], 0, 5
    )

    # ── Combine ──
    features = pd.DataFrame({'user_id': user_ids.values})
    features = features.merge(last_activity, on='user_id', how='left')
    features = features.merge(
        trend_ratio[['user_id', 'engagement_trend_ratio']], on='user_id', how='left'
    )
    features = features.merge(
        dur_trend[['user_id', 'duration_trend_ratio']], on='user_id', how='left'
    )

    return features


# =================================================================
# STEP 4: TEMPORAL PATTERNS
# =================================================================

def _compute_temporal_features(
    activity_df: pd.DataFrame,
    user_ids: pd.Series,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute temporal behavior patterns.

    Weekend vs weekday ratio, engagement consistency, etc.
    """
    lookback_30d = snapshot_date - pd.Timedelta(days=30)
    activity = activity_df[
        (activity_df['user_id'].isin(user_ids)) &
        (activity_df['activity_date'] <= snapshot_date) &
        (activity_df['activity_date'] > lookback_30d)
    ].copy()

    activity['is_weekend'] = activity['activity_date'].dt.dayofweek >= 5

    # Weekend engagement ratio
    weekend = activity[activity['is_weekend']].groupby('user_id')['n_sessions'].sum()
    weekday = activity[~activity['is_weekend']].groupby('user_id')['n_sessions'].sum()

    weekend_ratio = pd.DataFrame({
        'user_id': weekend.index.union(weekday.index)
    })
    weekend_ratio = weekend_ratio.merge(
        weekend.rename('weekend_sessions'), on='user_id', how='left'
    ).merge(
        weekday.rename('weekday_sessions'), on='user_id', how='left'
    )
    weekend_ratio['weekend_sessions'] = weekend_ratio['weekend_sessions'].fillna(0)
    weekend_ratio['weekday_sessions'] = weekend_ratio['weekday_sessions'].fillna(0)
    total = weekend_ratio['weekend_sessions'] + weekend_ratio['weekday_sessions']
    weekend_ratio['weekend_ratio'] = np.where(
        total > 0,
        weekend_ratio['weekend_sessions'] / total,
        0.0
    )

    # Engagement consistency (coefficient of variation of daily sessions)
    daily_sessions = activity.groupby(['user_id', 'activity_date'])['n_sessions'].sum()
    daily_sessions = daily_sessions.reset_index()

    consistency = daily_sessions.groupby('user_id')['n_sessions'].agg(
        ['mean', 'std']
    ).reset_index()
    consistency.columns = ['user_id', 'daily_sessions_mean', 'daily_sessions_std']
    consistency['engagement_cv'] = np.where(
        consistency['daily_sessions_mean'] > 0,
        consistency['daily_sessions_std'].fillna(0) / consistency['daily_sessions_mean'],
        0.0
    )

    # Combine
    features = pd.DataFrame({'user_id': user_ids.values})
    features = features.merge(
        weekend_ratio[['user_id', 'weekend_ratio']], on='user_id', how='left'
    )
    features = features.merge(
        consistency[['user_id', 'engagement_cv']], on='user_id', how='left'
    )

    return features


# =================================================================
# STEP 5: SUBSCRIPTION FEATURES
# =================================================================

def _compute_subscription_features(
    subscriptions_df: pd.DataFrame,
    user_ids: pd.Series,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute subscription and billing features."""
    # Filter subscriptions before snapshot
    subs = subscriptions_df[
        (subscriptions_df['user_id'].isin(user_ids)) &
        (subscriptions_df['period_start'] <= snapshot_date)
    ].copy()

    agg = subs.groupby('user_id').agg(
        n_renewals=('renewed', 'sum'),
        n_total_periods=('renewed', 'count'),
        n_payment_failures=('payment_failed', 'sum'),
        n_plan_changes=('plan', lambda x: x.nunique() - 1),
        total_revenue=('monthly_price', 'sum'),
    ).reset_index()

    agg['has_payment_failure'] = (agg['n_payment_failures'] > 0).astype(int)
    agg['payment_failure_rate'] = np.where(
        agg['n_total_periods'] > 0,
        agg['n_payment_failures'] / agg['n_total_periods'],
        0.0
    )

    features = pd.DataFrame({'user_id': user_ids.values})
    features = features.merge(agg, on='user_id', how='left')

    return features


# =================================================================
# STEP 6: SUPPORT TICKET FEATURES
# =================================================================

def _compute_support_features(
    tickets_df: pd.DataFrame,
    user_ids: pd.Series,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute support ticket features."""
    tickets = tickets_df[
        (tickets_df['user_id'].isin(user_ids)) &
        (tickets_df['created_date'] <= snapshot_date)
    ].copy()

    tickets['days_ago'] = (snapshot_date - tickets['created_date']).dt.days

    # Total tickets
    total = tickets.groupby('user_id').agg(
        total_tickets=('ticket_id', 'count'),
        avg_satisfaction=('satisfaction_score', 'mean'),
        avg_resolution_hours=('resolution_hours', 'mean'),
    ).reset_index()

    # Recent tickets (last 30 days)
    recent = tickets[tickets['days_ago'] <= 30].groupby('user_id').agg(
        tickets_last_30d=('ticket_id', 'count'),
    ).reset_index()

    # Recent tickets (last 7 days)
    very_recent = tickets[tickets['days_ago'] <= 7].groupby('user_id').agg(
        tickets_last_7d=('ticket_id', 'count'),
    ).reset_index()

    # Cancellation requests
    cancel = tickets[
        tickets['category'] == 'cancellation_request'
    ].groupby('user_id').size().reset_index(name='n_cancel_requests')

    # Billing issues
    billing = tickets[
        tickets['category'] == 'billing'
    ].groupby('user_id').size().reset_index(name='n_billing_tickets')

    # Combine
    features = pd.DataFrame({'user_id': user_ids.values})
    for df in [total, recent, very_recent, cancel, billing]:
        features = features.merge(df, on='user_id', how='left')

    features['has_cancel_request'] = (
        features['n_cancel_requests'].fillna(0) > 0
    ).astype(int)
    features['has_billing_issue'] = (
        features['n_billing_tickets'].fillna(0) > 0
    ).astype(int)

    return features


# =================================================================
# STEP 7: DERIVED & INTERACTION FEATURES
# =================================================================

def _compute_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction and derived features."""
    f = features.copy()

    # Engagement per dollar (are they getting value for their money?)
    f['sessions_per_dollar'] = np.where(
        f['monthly_price'] > 0,
        f['total_sessions_30d'].fillna(0) / f['monthly_price'],
        f['total_sessions_30d'].fillna(0)  # Free users: just use raw sessions
    )

    # Support intensity (tickets per month of tenure)
    tenure_months = np.maximum(f['tenure_at_snapshot'].fillna(30) / 30, 1)
    f['ticket_rate_monthly'] = f['total_tickets'].fillna(0) / tenure_months

    # Activity acceleration (7d trend relative to what's normal for this user)
    f['activity_acceleration'] = np.where(
        f['total_sessions_30d'].fillna(0) > 0,
        (f['total_sessions_7d'].fillna(0) * 30/7) / f['total_sessions_30d'],
        0.0
    )
    f['activity_acceleration'] = np.clip(f['activity_acceleration'], 0, 5)

    # Tenure x engagement interaction
    f['tenure_x_engagement'] = (
        np.log1p(f['tenure_at_snapshot'].fillna(0)) *
        np.log1p(f['total_sessions_30d'].fillna(0))
    )

    # Is the user increasingly contacting support?
    f['support_escalation'] = np.where(
        f['total_tickets'].fillna(0) > 0,
        f['tickets_last_7d'].fillna(0) / np.maximum(f['total_tickets'].fillna(1), 1),
        0.0
    )

    return f


# =================================================================
# STEP 8: FILL DEFAULTS
# =================================================================

def _fill_defaults(features: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values with sensible defaults."""
    f = features.copy()

    # Numeric columns: fill with 0 (no activity = 0)
    numeric_cols = f.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ('user_id', 'churned_target'):
            f[col] = f[col].fillna(0)

    # Days since last activity: fill with large number (never active)
    if 'days_since_last_activity' in f.columns:
        f['days_since_last_activity'] = f['days_since_last_activity'].fillna(999)

    # Satisfaction: fill with neutral (3.0)
    if 'avg_satisfaction' in f.columns:
        f.loc[f['avg_satisfaction'] == 0, 'avg_satisfaction'] = 3.0

    return f


# =================================================================
# UTILITY: GET FEATURE COLUMN NAMES
# =================================================================

def get_feature_columns(features_df: pd.DataFrame) -> list:
    """Return list of feature column names (excluding ID and target)."""
    exclude = {
        'user_id', 'churned_target', 'signup_date', 'plan',
        'platform', 'country', 'age_bucket', 'segment',
    }
    return [c for c in features_df.columns if c not in exclude]


def get_feature_summary(features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for all features."""
    feat_cols = get_feature_columns(features_df)
    summary = features_df[feat_cols].describe().T
    summary['null_pct'] = features_df[feat_cols].isnull().mean()
    summary['dtype'] = features_df[feat_cols].dtypes
    return summary.round(3)
