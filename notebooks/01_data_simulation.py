"""
PRISM Notebook 01: Data Simulation
==================================

Generates synthetic subscription data and saves outputs to data/raw/.
This is the reproducible entrypoint for the PRISM pipeline.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.data_simulator import SubscriptionDataSimulator


N_USERS = 200_000
OBSERVATION_DAYS = 365
SEED = 42

RAW_DIR = project_root / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PRISM Data Simulation Pipeline")
print("=" * 60)
print(f"\nUsers: {N_USERS:,} | Observation days: {OBSERVATION_DAYS} | Seed: {SEED}")

sim = SubscriptionDataSimulator(
    n_users=N_USERS,
    observation_days=OBSERVATION_DAYS,
    seed=SEED,
)

data = sim.generate_all()
sim.save_to_parquet(data, str(RAW_DIR))

print("\nQuick validation snapshot:")
print(f"  users:           {len(data['users']):,}")
print(f"  daily_activity:  {len(data['daily_activity']):,}")
print(f"  subscriptions:   {len(data['subscriptions']):,}")
print(f"  campaigns:       {len(data['campaigns']):,}")
print(f"  support_tickets: {len(data['support_tickets']):,}")
print(f"  ground_truth:    {len(data['ground_truth']):,}")

users = data["users"]
print(f"\n  Churn rate: {users['churned'].mean():.2%}")
print(f"  Plan mix:\n{users['plan'].value_counts(normalize=True).round(3).to_string()}")

print(f"\n{'=' * 60}")
print("  Data simulation complete!")
print("  Next: notebooks/02_data_quality_checks.py")
print(f"{'=' * 60}")

