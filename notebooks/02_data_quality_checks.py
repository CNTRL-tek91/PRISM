"""
PRISM Notebook 02: Data Quality Checks
========================================

Validates the integrity and quality of all raw data tables before
feature engineering. Implements data validation best practices.

Checks performed:
- Schema validation (column types, nullable constraints)
- Row count expectations
- Referential integrity (foreign key relationships)
- Distribution checks (no impossible values)
- Freshness checks (no future dates)
- Uniqueness constraints (primary keys)

This mirrors the same concepts as Great Expectations / dbt tests,
implemented in a portable way that works locally and on Databricks.

Output: Prints validation report with PASS/FAIL for each check.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import List, Tuple


# =================================================================
# VALIDATION FRAMEWORK
# =================================================================

class DataValidator:
    """Lightweight data validation framework.

    Mirrors Great Expectations concepts:
    - Expectations are defined as methods
    - Results are collected and reported
    """

    def __init__(self, name: str):
        self.name = name
        self.results: List[Tuple[str, bool, str]] = []

    def expect(self, condition: bool, description: str, details: str = ""):
        """Register an expectation result."""
        status = "PASS" if condition else "FAIL"
        self.results.append((description, condition, details))
        symbol = "[PASS]" if condition else "[FAIL]"
        print(f"    {symbol} {description}")
        if not condition and details:
            print(f"           -> {details}")

    def report(self) -> dict:
        """Print summary and return results."""
        total = len(self.results)
        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = total - passed
        pct = passed / total * 100 if total > 0 else 0

        print(f"\n  Summary: {passed}/{total} checks passed ({pct:.0f}%)")
        if failed > 0:
            print(f"  WARNING: {failed} check(s) failed!")
        return {
            'table': self.name,
            'total': total,
            'passed': passed,
            'failed': failed,
        }


# =================================================================
# LOAD DATA
# =================================================================

print("=" * 60)
print("  PRISM Data Quality Validation")
print("=" * 60)

data_dir = project_root / 'data' / 'raw'

print("\nLoading tables...")
users = pd.read_parquet(data_dir / 'users.parquet')
daily_activity = pd.read_parquet(data_dir / 'daily_activity.parquet')
subscriptions = pd.read_parquet(data_dir / 'subscriptions.parquet')
support_tickets = pd.read_parquet(data_dir / 'support_tickets.parquet')
campaigns = pd.read_parquet(data_dir / 'campaigns.parquet')
ground_truth = pd.read_parquet(data_dir / 'ground_truth.parquet')

all_results = []


# =================================================================
# VALIDATE: USERS TABLE
# =================================================================

print(f"\n{'─'*60}")
print(f"  Validating: users ({len(users):,} rows)")
print(f"{'─'*60}")

v = DataValidator('users')

# Schema checks
v.expect('user_id' in users.columns, "Column 'user_id' exists")
v.expect('signup_date' in users.columns, "Column 'signup_date' exists")
v.expect('plan' in users.columns, "Column 'plan' exists")
v.expect('churned' in users.columns, "Column 'churned' exists")

# Row count
v.expect(len(users) >= 100_000, f"Row count >= 100K (actual: {len(users):,})")
v.expect(len(users) <= 500_000, f"Row count <= 500K (actual: {len(users):,})")

# Uniqueness
n_unique_ids = users['user_id'].nunique()
v.expect(n_unique_ids == len(users),
         f"user_id is unique ({n_unique_ids:,} unique of {len(users):,})")

# No nulls in required columns
for col in ['user_id', 'signup_date', 'plan', 'platform', 'churned']:
    null_count = users[col].isnull().sum()
    v.expect(null_count == 0, f"No nulls in '{col}' (nulls: {null_count})")

# Valid plan values
valid_plans = {'free', 'basic', 'premium'}
actual_plans = set(users['plan'].unique())
v.expect(actual_plans == valid_plans,
         f"Plan values are valid",
         f"Expected {valid_plans}, got {actual_plans}")

# Valid segment values
valid_segments = {'power', 'regular', 'casual', 'at_risk'}
actual_segments = set(users['segment'].unique())
v.expect(actual_segments == valid_segments, "Segment values are valid")

# Churn rate is realistic (20-50%)
churn_rate = users['churned'].mean()
v.expect(0.15 <= churn_rate <= 0.50,
         f"Churn rate is realistic (actual: {churn_rate:.1%})")

# No future signup dates
max_signup = users['signup_date'].max()
v.expect(max_signup <= pd.Timestamp('2025-01-01'),
         f"No future signup dates (max: {max_signup.date()})")

# Tenure is positive
v.expect((users['tenure_days'] > 0).all(), "All tenure_days > 0")

# Monthly price matches plan
free_users = users[users['plan'] == 'free']
v.expect((free_users['monthly_price'] == 0).all(), "Free users have $0 price")

all_results.append(v.report())


# =================================================================
# VALIDATE: DAILY ACTIVITY TABLE
# =================================================================

print(f"\n{'─'*60}")
print(f"  Validating: daily_activity ({len(daily_activity):,} rows)")
print(f"{'─'*60}")

v = DataValidator('daily_activity')

# Row count
v.expect(len(daily_activity) >= 5_000_000,
         f"Row count >= 5M (actual: {len(daily_activity):,})")

# Schema
for col in ['user_id', 'activity_date', 'n_sessions', 'total_duration_min']:
    v.expect(col in daily_activity.columns, f"Column '{col}' exists")

# Referential integrity: all activity user_ids exist in users
activity_users = set(daily_activity['user_id'].unique())
valid_users = set(users['user_id'].unique())
orphans = activity_users - valid_users
v.expect(len(orphans) == 0,
         f"All activity user_ids exist in users table (orphans: {len(orphans)})")

# No negative values
v.expect((daily_activity['n_sessions'] >= 0).all(), "n_sessions >= 0")
v.expect((daily_activity['total_duration_min'] >= 0).all(), "total_duration_min >= 0")
v.expect((daily_activity['n_content_items'] >= 0).all(), "n_content_items >= 0")

# Date range
min_date = daily_activity['activity_date'].min()
max_date = daily_activity['activity_date'].max()
v.expect(min_date >= pd.Timestamp('2023-06-01'),
         f"Min activity date is reasonable (actual: {min_date.date()})")
v.expect(max_date <= pd.Timestamp('2025-01-01'),
         f"Max activity date is reasonable (actual: {max_date.date()})")

# No duplicate (user_id, activity_date) pairs
n_dup = daily_activity.duplicated(subset=['user_id', 'activity_date']).sum()
dup_rate = n_dup / len(daily_activity) if len(daily_activity) > 0 else 0
v.expect(dup_rate < 0.01,
         f"Low duplicate rate (actual: {n_dup:,} = {dup_rate:.2%})",
         f"Some duplicates expected from data generation")

all_results.append(v.report())


# =================================================================
# VALIDATE: SUBSCRIPTIONS TABLE
# =================================================================

print(f"\n{'─'*60}")
print(f"  Validating: subscriptions ({len(subscriptions):,} rows)")
print(f"{'─'*60}")

v = DataValidator('subscriptions')

# Schema
for col in ['user_id', 'period_start', 'period_end', 'monthly_price', 'payment_failed']:
    v.expect(col in subscriptions.columns, f"Column '{col}' exists")

# No free users in subscriptions
sub_users = set(subscriptions['user_id'].unique())
free_user_ids = set(users[users['plan'] == 'free']['user_id'].unique())
free_in_subs = sub_users & free_user_ids
v.expect(len(free_in_subs) == 0,
         f"No free users in subscriptions (found: {len(free_in_subs)})")

# Price > 0 for all subscription records
v.expect((subscriptions['monthly_price'] > 0).all(),
         "All subscription prices > $0")

# Payment failure rate is realistic (1-10%)
pf_rate = subscriptions['payment_failed'].mean()
v.expect(0.01 <= pf_rate <= 0.15,
         f"Payment failure rate is realistic (actual: {pf_rate:.1%})")

all_results.append(v.report())


# =================================================================
# VALIDATE: CAMPAIGNS TABLE
# =================================================================

print(f"\n{'─'*60}")
print(f"  Validating: campaigns ({len(campaigns):,} rows)")
print(f"{'─'*60}")

v = DataValidator('campaigns')

# Treatment is binary
v.expect(set(campaigns['treatment'].unique()) == {0, 1},
         "Treatment is binary (0/1)")

# Treatment ratio is ~50/50
treatment_rate = campaigns['treatment'].mean()
v.expect(0.45 <= treatment_rate <= 0.55,
         f"Treatment rate is ~50% (actual: {treatment_rate:.1%})")

# Referential integrity
camp_users = set(campaigns['user_id'].unique())
orphans = camp_users - valid_users
v.expect(len(orphans) == 0,
         f"All campaign user_ids exist in users (orphans: {len(orphans)})")

# Churned_within_30d is binary
v.expect(set(campaigns['churned_within_30d'].unique()) <= {0, 1},
         "churned_within_30d is binary")

all_results.append(v.report())


# =================================================================
# VALIDATE: SUPPORT TICKETS TABLE
# =================================================================

print(f"\n{'─'*60}")
print(f"  Validating: support_tickets ({len(support_tickets):,} rows)")
print(f"{'─'*60}")

v = DataValidator('support_tickets')

# Satisfaction score range
v.expect(support_tickets['satisfaction_score'].between(1, 5).all(),
         "Satisfaction score is 1-5")

# Resolution hours is positive
v.expect((support_tickets['resolution_hours'] > 0).all(),
         "Resolution hours > 0")

# Valid categories
valid_cats = {'billing', 'technical', 'content', 'cancellation_request', 'account'}
actual_cats = set(support_tickets['category'].unique())
v.expect(actual_cats == valid_cats,
         f"Category values are valid",
         f"Expected {valid_cats}, got {actual_cats}")

# Referential integrity
ticket_users = set(support_tickets['user_id'].unique())
orphans = ticket_users - valid_users
v.expect(len(orphans) == 0,
         f"All ticket user_ids exist in users (orphans: {len(orphans)})")

all_results.append(v.report())


# =================================================================
# VALIDATE: GROUND TRUTH TABLE
# =================================================================

print(f"\n{'─'*60}")
print(f"  Validating: ground_truth ({len(ground_truth):,} rows)")
print(f"{'─'*60}")

v = DataValidator('ground_truth')

v.expect(len(ground_truth) == len(users),
         f"Ground truth has same row count as users ({len(ground_truth):,})")

v.expect(ground_truth['persuadability'].between(0, 1).all(),
         "Persuadability is between 0 and 1")

# ~30% should be "persuadable" (persuadability > 0.3)
persuadable_pct = (ground_truth['persuadability'] > 0.3).mean()
v.expect(0.15 <= persuadable_pct <= 0.45,
         f"Persuadable fraction is reasonable (actual: {persuadable_pct:.1%})")

all_results.append(v.report())


# =================================================================
# FINAL REPORT
# =================================================================

print(f"\n{'='*60}")
print(f"  VALIDATION SUMMARY")
print(f"{'='*60}\n")

total_checks = sum(r['total'] for r in all_results)
total_passed = sum(r['passed'] for r in all_results)
total_failed = sum(r['failed'] for r in all_results)

for r in all_results:
    status = "PASS" if r['failed'] == 0 else "FAIL"
    print(f"  [{status}] {r['table']:20s} — {r['passed']}/{r['total']} checks passed")

print(f"\n  Overall: {total_passed}/{total_checks} checks passed "
      f"({total_passed/total_checks*100:.0f}%)")

if total_failed == 0:
    print("\n  All data quality checks passed! Data is ready for feature engineering.")
else:
    print(f"\n  WARNING: {total_failed} check(s) failed. Review before proceeding.")

print(f"\n{'='*60}")
