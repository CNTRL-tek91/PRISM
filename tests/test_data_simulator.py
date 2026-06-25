import pandas as pd

from src.data_simulator import SubscriptionDataSimulator


def test_simulator_generates_expected_tables():
    sim = SubscriptionDataSimulator(n_users=300, observation_days=90, seed=7)
    data = sim.generate_all()

    expected = {
        "users",
        "daily_activity",
        "subscriptions",
        "campaigns",
        "support_tickets",
        "ground_truth",
    }
    assert set(data.keys()) == expected
    assert len(data["users"]) == 300
    assert "user_id" in data["users"].columns
    assert data["users"]["user_id"].is_unique

