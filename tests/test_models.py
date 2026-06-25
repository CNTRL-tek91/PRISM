import pandas as pd


def test_phase_outputs_exist():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    required = [
        root / "data" / "results" / "survival_curves.csv",
        root / "data" / "results" / "uplift_predictions.csv",
        root / "data" / "results" / "ltv_estimates.csv",
        root / "data" / "results" / "roi_strategy_comparison.csv",
    ]
    for p in required:
        assert p.exists(), f"Missing expected output: {p}"


def test_roi_contains_expected_strategies():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    roi = pd.read_csv(root / "data" / "results" / "roi_strategy_comparison.csv")
    expected = {"Random Targeting", "Churn-Targeted", "Uplift x LTV Targeted"}
    assert expected.issubset(set(roi["strategy"]))

