import pandas as pd

from src.data_simulator import SubscriptionDataSimulator
from src.feature_engineering import build_churn_features


def test_feature_engineering_outputs_required_columns():
    sim = SubscriptionDataSimulator(n_users=300, observation_days=120, seed=11)
    data = sim.generate_all()

    features = build_churn_features(
        users_df=data["users"],
        activity_df=data["daily_activity"],
        subscriptions_df=data["subscriptions"],
        tickets_df=data["support_tickets"],
        snapshot_day=80,
        target_window_days=30,
        start_date=pd.Timestamp("2024-01-01"),
    )

    required_cols = {"user_id", "churned_target", "tenure_at_snapshot"}
    assert required_cols.issubset(features.columns)
    assert len(features) > 0
    assert features["churned_target"].isin([0, 1]).all()

