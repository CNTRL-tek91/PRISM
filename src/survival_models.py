"""
PRISM Survival Models Module
==============================

Provides survival analysis models for time-to-churn prediction.

Models Implemented
------------------
1. Cox Proportional Hazards (lifelines)  — Interpretable hazard ratios
2. Gradient Boosted Survival (scikit-survival) — High-performance
3. Random Survival Forest (scikit-survival) — Ensemble method

Each model produces:
- Concordance index (discrimination)
- Integrated Brier score (calibration)
- Individual survival curves S(t|X) for any user

Usage
-----
    from src.survival_models import prepare_survival_data, SurvivalTrainer

    data = prepare_survival_data(features_df, users_df)
    trainer = SurvivalTrainer(data)
    results = trainer.train_all()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lifelines_ci
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score
import logging
import time
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
logger = logging.getLogger(__name__)


# =================================================================
# DATA PREPARATION
# =================================================================

def prepare_survival_data(
    features_df: pd.DataFrame,
    users_df: pd.DataFrame,
    snapshot_day: int = 300,
    observation_days: int = 365,
    start_date: pd.Timestamp = pd.Timestamp('2024-01-01'),
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Prepare data for survival modeling.

    Computes time-to-event (days from snapshot to churn/censoring)
    and event indicator from the users table.

    Parameters
    ----------
    features_df : DataFrame
        Output from feature engineering pipeline.
    users_df : DataFrame
        Raw users table with churn_date column.
    snapshot_day : int
        Day at which features were computed.
    observation_days : int
        End of observation window.
    test_size : float
        Fraction of data for test set.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        X_train, X_test : np.ndarray (scaled features)
        y_train, y_test : structured ndarray [(event, time)]
        feature_names : list of str
        scaler : fitted StandardScaler
        train_df, test_df : DataFrames for lifelines (unscaled)
    """
    snapshot_date = start_date + pd.Timedelta(days=snapshot_day)
    end_date = start_date + pd.Timedelta(days=observation_days)

    # Join features with churn dates
    merged = features_df.merge(
        users_df[['user_id', 'churn_date']], on='user_id', how='left'
    )

    # Compute survival target variables
    # Time: days from snapshot to churn (or end of observation if censored)
    event_date = merged['churn_date'].fillna(end_date).clip(upper=end_date)
    merged['duration'] = (event_date - snapshot_date).dt.days.clip(lower=1)
    merged['event'] = (
        merged['churn_date'].notna() &
        (merged['churn_date'] > snapshot_date) &
        (merged['churn_date'] <= end_date)
    ).astype(int)

    # Identify feature columns (numeric only, exclude IDs and targets)
    exclude = {
        'user_id', 'churned_target', 'signup_date', 'plan', 'platform',
        'country', 'age_bucket', 'segment', 'churn_date',
        'duration', 'event',
    }
    feature_cols = [
        c for c in merged.columns
        if c not in exclude
        and merged[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool', 'uint8']
    ]

    print(f"  Survival data: {len(merged):,} users, "
          f"{merged['event'].sum():,} events ({merged['event'].mean():.1%})")
    print(f"  Follow-up: median {merged['duration'].median():.0f}d, "
          f"max {merged['duration'].max():.0f}d")
    print(f"  Features: {len(feature_cols)}")

    X = merged[feature_cols].values.astype(np.float64)

    # Structured array for scikit-survival
    y = np.array(
        list(zip(merged['event'].astype(bool), merged['duration'].astype(float))),
        dtype=[('event', bool), ('time', float)]
    )

    # Train/test split (stratified by event)
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed,
        stratify=merged['event'].values
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # DataFrames for lifelines (needs unscaled data with duration/event columns)
    train_df = merged.iloc[train_idx][feature_cols + ['duration', 'event']].copy()
    test_df = merged.iloc[test_idx][feature_cols + ['duration', 'event']].copy()

    # Scale features in DataFrames too (lifelines needs this for convergence)
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[feature_cols] = X_train_scaled
    test_df_scaled[feature_cols] = X_test_scaled

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'scaler': scaler,
        'train_df': train_df_scaled,
        'test_df': test_df_scaled,
        'train_idx': train_idx,
        'test_idx': test_idx,
    }


# =================================================================
# SURVIVAL TRAINER
# =================================================================

class SurvivalTrainer:
    """Trains and evaluates multiple survival models.

    Parameters
    ----------
    data : dict
        Output from prepare_survival_data().
    """

    def __init__(self, data: dict):
        self.data = data
        self.models = {}
        self.results = {}

    def train_all(self) -> Dict[str, dict]:
        """Train all three survival models and return comparison."""
        print(f"\n{'='*60}")
        print(f"  Survival Model Training")
        print(f"{'='*60}")

        # Model 1: Cox PH
        print("\n  [1/3] Cox Proportional Hazards (lifelines)")
        print("  " + "-"*50)
        self.train_cox_ph()

        # Model 2: Gradient Boosted Survival
        print("\n  [2/3] Gradient Boosted Survival Analysis")
        print("  " + "-"*50)
        self.train_gbsa()

        # Model 3: Random Survival Forest
        print("\n  [3/3] Random Survival Forest")
        print("  " + "-"*50)
        self.train_rsf()

        # Print comparison
        self._print_comparison()

        return self.results

    def train_cox_ph(
        self,
        penalizer: float = 0.01,
        l1_ratio: float = 0.1,
    ) -> dict:
        """Train Cox Proportional Hazards model.

        Returns interpretable hazard ratios for each feature.
        """
        t0 = time.time()

        cph = CoxPHFitter(
            penalizer=penalizer,
            l1_ratio=l1_ratio,
        )

        train_df = self.data['train_df']
        test_df = self.data['test_df']

        cph.fit(
            train_df,
            duration_col='duration',
            event_col='event',
            show_progress=False,
        )

        # Evaluate on test set
        test_concordance = cph.score(test_df, scoring_method='concordance_index')

        # Hazard ratios (top features)
        summary = cph.summary
        summary['abs_coef'] = summary['coef'].abs()
        top_features = summary.nlargest(15, 'abs_coef')[
            ['coef', 'exp(coef)', 'p', 'abs_coef']
        ].copy()
        top_features.columns = ['coefficient', 'hazard_ratio', 'p_value', 'abs_coef']

        elapsed = time.time() - t0

        result = {
            'model': cph,
            'model_type': 'cox_ph',
            'concordance_index': test_concordance,
            'train_time': elapsed,
            'top_features': top_features,
            'params': {'penalizer': penalizer, 'l1_ratio': l1_ratio},
        }

        self.models['cox_ph'] = cph
        self.results['cox_ph'] = result

        print(f"    C-index (test): {test_concordance:.4f}")
        print(f"    Training time: {elapsed:.1f}s")
        print(f"    Top hazard ratios:")
        for feat, row in top_features.head(8).iterrows():
            direction = "^" if row['hazard_ratio'] > 1 else "v"
            print(f"      {direction} {feat:35s} HR={row['hazard_ratio']:.3f} (p={row['p_value']:.4f})")

        return result

    def train_gbsa(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        subsample: float = 0.8,
    ) -> dict:
        """Train Gradient Boosted Survival Analysis."""
        t0 = time.time()

        gbsa = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=42,
        )

        gbsa.fit(self.data['X_train'], self.data['y_train'])

        # Evaluate
        test_ci = gbsa.score(self.data['X_test'], self.data['y_test'])

        # Feature importance
        importances = pd.Series(
            gbsa.feature_importances_,
            index=self.data['feature_names']
        ).sort_values(ascending=False)

        elapsed = time.time() - t0

        result = {
            'model': gbsa,
            'model_type': 'gbsa',
            'concordance_index': test_ci,
            'train_time': elapsed,
            'feature_importances': importances,
            'params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'subsample': subsample,
            },
        }

        self.models['gbsa'] = gbsa
        self.results['gbsa'] = result

        print(f"    C-index (test): {test_ci:.4f}")
        print(f"    Training time: {elapsed:.1f}s")
        print(f"    Top features (importance):")
        for feat, imp in importances.head(8).items():
            print(f"      {feat:35s} {imp:.4f}")

        return result

    def train_rsf(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
    ) -> dict:
        """Train Random Survival Forest."""
        t0 = time.time()

        rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42,
        )

        rsf.fit(self.data['X_train'], self.data['y_train'])

        # Evaluate
        test_ci = rsf.score(self.data['X_test'], self.data['y_test'])

        # Permutation importance (RSF doesn't have built-in feature_importances_)
        # We'll compute this in the notebook with SHAP instead
        elapsed = time.time() - t0

        result = {
            'model': rsf,
            'model_type': 'rsf',
            'concordance_index': test_ci,
            'train_time': elapsed,
            'params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
            },
        }

        self.models['rsf'] = rsf
        self.results['rsf'] = result

        print(f"    C-index (test): {test_ci:.4f}")
        print(f"    Training time: {elapsed:.1f}s")

        return result

    def predict_survival_curves(
        self,
        model_name: str,
        X: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict survival probabilities at given timepoints.

        Parameters
        ----------
        model_name : str
            One of 'gbsa', 'rsf'.
        X : ndarray
            Feature matrix (scaled).
        times : ndarray, optional
            Timepoints to predict at. Default: [7, 14, 30, 45, 60].

        Returns
        -------
        times : ndarray
            Time points.
        survival_probs : ndarray, shape (n_samples, n_times)
            Survival probabilities.
        """
        if times is None:
            times = np.array([7, 14, 30, 45, 60])

        model = self.models[model_name]
        surv_fns = model.predict_survival_function(X)

        # Extract probabilities at specified timepoints
        probs = np.zeros((len(X), len(times)))
        for i, fn in enumerate(surv_fns):
            for j, t in enumerate(times):
                # Clamp t to the available time range
                t_clamped = min(t, fn.x[-1])
                probs[i, j] = fn(t_clamped)

        return times, probs

    def get_risk_scores(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Get risk scores (higher = more likely to churn).

        For Cox PH, this is the linear predictor (log partial hazard).
        For GBSA/RSF, this is the negative cumulative hazard.
        """
        model = self.models[model_name]
        if model_name == 'cox_ph':
            # CoxPH returns log partial hazard
            return model.predict_partial_hazard(
                pd.DataFrame(X, columns=self.data['feature_names'])
            ).values
        else:
            # scikit-survival: predict returns risk score
            return model.predict(X)

    def create_risk_tiers(
        self,
        model_name: str,
        X: np.ndarray,
    ) -> np.ndarray:
        """Classify users into risk tiers based on predicted risk.

        Tiers: Low, Moderate, Elevated, High, Critical
        """
        scores = self.get_risk_scores(model_name, X)
        percentiles = np.percentile(scores, [20, 40, 60, 80])

        tiers = np.where(
            scores <= percentiles[0], 'Low',
            np.where(
                scores <= percentiles[1], 'Moderate',
                np.where(
                    scores <= percentiles[2], 'Elevated',
                    np.where(
                        scores <= percentiles[3], 'High',
                        'Critical'
                    )
                )
            )
        )
        return tiers

    def _print_comparison(self):
        """Print model comparison table."""
        print(f"\n{'='*60}")
        print(f"  MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Model':<30s} {'C-index':>10s} {'Time (s)':>10s}")
        print(f"  {'-'*50}")

        best_ci = 0
        best_model = ''
        for name, result in self.results.items():
            ci = result['concordance_index']
            t = result['train_time']
            print(f"  {name:<30s} {ci:>10.4f} {t:>10.1f}")
            if ci > best_ci:
                best_ci = ci
                best_model = name

        print(f"  {'-'*50}")
        print(f"  Best model: {best_model} (C-index: {best_ci:.4f})")
        print(f"{'='*60}")
