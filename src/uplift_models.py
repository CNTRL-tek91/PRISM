"""
PRISM Uplift Models Module
============================

Implements causal uplift modeling to identify which users are
actually "persuadable" — i.e., which users will CHANGE their behavior
in response to a retention intervention.

Models Implemented
------------------
1. T-Learner (XGBoost)   — Two separate models for treatment/control
2. S-Learner (XGBoost)   — Single model with treatment as feature
3. Causal Forest (econml) — Non-parametric CATE estimation

Key Concepts
------------
- CATE: Conditional Average Treatment Effect = E[Y(1) - Y(0) | X]
- Uplift: The causal impact of treatment on each individual
- Quadrants:
    * Persuadable:  Treatment reduces churn (target these)
    * Sure Thing:   Won't churn regardless (don't waste budget)
    * Lost Cause:   Will churn regardless (don't waste budget)
    * Sleeping Dog:  Treatment INCREASES churn (avoid targeting!)

Usage
-----
    from src.uplift_models import prepare_uplift_data, UpliftTrainer

    data = prepare_uplift_data(features, campaigns, ground_truth)
    trainer = UpliftTrainer(data)
    results = trainer.train_all()
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import xgboost as xgb
from econml.dml import CausalForestDML
from econml.metalearners import XLearner
import logging
import time
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

try:
    from causalml.inference.tree import UpliftRandomForestClassifier
    HAS_CAUSALML = True
except Exception:
    UpliftRandomForestClassifier = None
    HAS_CAUSALML = False


# =================================================================
# DATA PREPARATION
# =================================================================

def prepare_uplift_data(
    features_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    ground_truth_df: Optional[pd.DataFrame] = None,
    test_size: float = 0.3,
    seed: int = 42,
) -> dict:
    """Prepare data for uplift modeling.

    Joins features with campaign records to create the uplift dataset.

    Parameters
    ----------
    features_df : DataFrame
        Features computed at campaign-time snapshot.
    campaigns_df : DataFrame
        Campaign records with treatment and outcome.
    ground_truth_df : DataFrame, optional
        Hidden persuadability scores for evaluation.
    test_size : float
        Fraction for test set.
    seed : int
        Random seed.

    Returns
    -------
    dict with train/test splits, feature names, and ground truth.
    """
    # Join features with campaign data
    uplift_df = features_df.merge(
        campaigns_df[['user_id', 'treatment', 'churned_within_30d', 'campaign_type']],
        on='user_id',
        how='inner',
    )

    # Join ground truth if available
    if ground_truth_df is not None:
        uplift_df = uplift_df.merge(
            ground_truth_df[['user_id', 'persuadability']],
            on='user_id',
            how='left',
        )

    # Feature columns
    exclude = {
        'user_id', 'churned_target', 'signup_date', 'plan', 'platform',
        'country', 'age_bucket', 'segment', 'treatment',
        'churned_within_30d', 'campaign_type', 'persuadability',
    }
    feature_cols = [
        c for c in uplift_df.columns
        if c not in exclude
        and uplift_df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool', 'uint8']
    ]

    print(f"  Uplift data: {len(uplift_df):,} users")
    print(f"  Treatment: {uplift_df['treatment'].sum():,} treated, "
          f"{(~uplift_df['treatment'].astype(bool)).sum():,} control")
    print(f"  Churn rate (treated):  {uplift_df[uplift_df['treatment']==1]['churned_within_30d'].mean():.2%}")
    print(f"  Churn rate (control): {uplift_df[uplift_df['treatment']==0]['churned_within_30d'].mean():.2%}")
    print(f"  ATE: {uplift_df[uplift_df['treatment']==0]['churned_within_30d'].mean() - uplift_df[uplift_df['treatment']==1]['churned_within_30d'].mean():.4f}")
    print(f"  Features: {len(feature_cols)}")

    # Train/test split
    X = uplift_df[feature_cols].values.astype(np.float64)
    T = uplift_df['treatment'].values.astype(np.int32)
    Y = uplift_df['churned_within_30d'].values.astype(np.int32)

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=T
    )

    data = {
        'X_train': X[train_idx],
        'X_test': X[test_idx],
        'T_train': T[train_idx],
        'T_test': T[test_idx],
        'Y_train': Y[train_idx],
        'Y_test': Y[test_idx],
        'feature_names': feature_cols,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'full_df': uplift_df,
    }

    if ground_truth_df is not None and 'persuadability' in uplift_df.columns:
        data['persuadability_test'] = uplift_df['persuadability'].values[test_idx]

    return data


# =================================================================
# UPLIFT MODELS
# =================================================================

class TLearner:
    """Two-Model approach: separate models for treatment and control.

    The simplest uplift method — train one classifier on treated users
    and another on control users, then take the difference.
    """

    def __init__(self, **xgb_params):
        default_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'random_state': 42,
        }
        default_params.update(xgb_params)

        self.model_t = xgb.XGBClassifier(**default_params)
        self.model_c = xgb.XGBClassifier(**default_params)

    def fit(self, X, treatment, outcome):
        t_mask = treatment == 1
        self.model_t.fit(X[t_mask], outcome[t_mask])
        self.model_c.fit(X[~t_mask], outcome[~t_mask])

    def predict_uplift(self, X):
        """Positive uplift = treatment reduces churn (good)."""
        p_treated = self.model_t.predict_proba(X)[:, 1]
        p_control = self.model_c.predict_proba(X)[:, 1]
        return p_control - p_treated  # Positive = treatment helps

    def predict_proba_both(self, X):
        """Return P(churn|treated) and P(churn|control) for quadrant analysis."""
        p_treated = self.model_t.predict_proba(X)[:, 1]
        p_control = self.model_c.predict_proba(X)[:, 1]
        return p_treated, p_control


class SLearner:
    """Single-Model approach: treatment indicator as an extra feature.

    Simpler but can underestimate treatment effects when the
    treatment signal is weak relative to other features.
    """

    def __init__(self, **xgb_params):
        default_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'random_state': 42,
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBClassifier(**default_params)

    def fit(self, X, treatment, outcome):
        X_with_t = np.column_stack([X, treatment])
        self.model.fit(X_with_t, outcome)

    def predict_uplift(self, X):
        X_treated = np.column_stack([X, np.ones(len(X))])
        X_control = np.column_stack([X, np.zeros(len(X))])
        p_treated = self.model.predict_proba(X_treated)[:, 1]
        p_control = self.model.predict_proba(X_control)[:, 1]
        return p_control - p_treated


# =================================================================
# UPLIFT TRAINER
# =================================================================

class UpliftTrainer:
    """Trains and evaluates multiple uplift models."""

    def __init__(self, data: dict):
        self.data = data
        self.models = {}
        self.results = {}

    def train_all(self) -> Dict[str, dict]:
        """Train all uplift models."""
        print(f"\n{'='*60}")
        print(f"  Uplift Model Training")
        print(f"{'='*60}")

        print("\n  [1/5] T-Learner (Two-Model)")
        print("  " + "-"*50)
        self.train_t_learner()

        print("\n  [2/5] S-Learner (Single-Model)")
        print("  " + "-"*50)
        self.train_s_learner()

        print("\n  [3/5] Uplift Random Forest (causalml)")
        print("  " + "-"*50)
        self.train_uplift_random_forest()

        print("\n  [4/5] X-Learner (econml)")
        print("  " + "-"*50)
        self.train_x_learner()

        print("\n  [5/5] Causal Forest (econml)")
        print("  " + "-"*50)
        self.train_causal_forest()

        self._print_comparison()
        return self.results

    def train_t_learner(self) -> dict:
        """Train T-Learner (two separate XGBoost models)."""
        t0 = time.time()
        model = TLearner()
        model.fit(
            self.data['X_train'],
            self.data['T_train'],
            self.data['Y_train'],
        )

        uplift_scores = model.predict_uplift(self.data['X_test'])
        metrics = self._evaluate_uplift(
            uplift_scores, self.data['T_test'], self.data['Y_test']
        )

        elapsed = time.time() - t0
        result = {
            'model': model,
            'model_type': 't_learner',
            'uplift_scores': uplift_scores,
            'metrics': metrics,
            'train_time': elapsed,
        }

        self.models['t_learner'] = model
        self.results['t_learner'] = result

        print(f"    Qini coefficient: {metrics['qini_coefficient']:.4f}")
        print(f"    Mean uplift: {uplift_scores.mean():.4f}")
        print(f"    Training time: {elapsed:.1f}s")
        return result

    def train_s_learner(self) -> dict:
        """Train S-Learner (treatment as feature)."""
        t0 = time.time()
        model = SLearner()
        model.fit(
            self.data['X_train'],
            self.data['T_train'],
            self.data['Y_train'],
        )

        uplift_scores = model.predict_uplift(self.data['X_test'])
        metrics = self._evaluate_uplift(
            uplift_scores, self.data['T_test'], self.data['Y_test']
        )

        elapsed = time.time() - t0
        result = {
            'model': model,
            'model_type': 's_learner',
            'uplift_scores': uplift_scores,
            'metrics': metrics,
            'train_time': elapsed,
        }

        self.models['s_learner'] = model
        self.results['s_learner'] = result

        print(f"    Qini coefficient: {metrics['qini_coefficient']:.4f}")
        print(f"    Mean uplift: {uplift_scores.mean():.4f}")
        print(f"    Training time: {elapsed:.1f}s")
        return result

    def train_uplift_random_forest(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        min_samples_leaf: int = 100,
    ) -> dict:
        """Train Uplift Random Forest from causalml."""
        t0 = time.time()

        if not HAS_CAUSALML:
            raise ImportError(
                "causalml is not installed. Install `causalml` to run uplift random forest."
            )

        model = UpliftRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            control_name='control',
        )

        treatment_train = np.where(
            self.data['T_train'] == 1, 'treatment', 'control'
        )
        model.fit(
            X=self.data['X_train'],
            treatment=treatment_train,
            y=self.data['Y_train'],
        )

        pred = model.predict(self.data['X_test'])
        if isinstance(pred, pd.DataFrame):
            pred_arr = pred.values
        else:
            pred_arr = np.asarray(pred)

        if pred_arr.ndim == 1:
            uplift_scores = pred_arr.astype(float)
        else:
            uplift_scores = pred_arr[:, 0].astype(float)

        metrics = self._evaluate_uplift(
            uplift_scores, self.data['T_test'], self.data['Y_test']
        )

        elapsed = time.time() - t0
        result = {
            'model': model,
            'model_type': 'uplift_random_forest',
            'uplift_scores': uplift_scores,
            'metrics': metrics,
            'train_time': elapsed,
        }

        self.models['uplift_random_forest'] = model
        self.results['uplift_random_forest'] = result

        print(f"    Qini coefficient: {metrics['qini_coefficient']:.4f}")
        print(f"    AUUC: {metrics['auuc']:.4f}")
        print(f"    Mean uplift: {uplift_scores.mean():.4f}")
        print(f"    Training time: {elapsed:.1f}s")
        return result

    def train_x_learner(self) -> dict:
        """Train X-Learner using XGBoost base models."""
        t0 = time.time()

        model = XLearner(
            models=xgb.XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42,
            ),
            cate_models=xgb.XGBRegressor(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            propensity_model=xgb.XGBClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.1,
                eval_metric='logloss',
                random_state=42,
            ),
        )

        model.fit(
            Y=self.data['Y_train'].astype(float),
            T=self.data['T_train'].astype(float),
            X=self.data['X_train'],
        )

        cate = model.effect(self.data['X_test'])
        uplift_scores = -cate.flatten()  # positive = treatment reduces churn

        metrics = self._evaluate_uplift(
            uplift_scores, self.data['T_test'], self.data['Y_test']
        )

        elapsed = time.time() - t0
        result = {
            'model': model,
            'model_type': 'x_learner',
            'uplift_scores': uplift_scores,
            'cate': cate.flatten(),
            'metrics': metrics,
            'train_time': elapsed,
        }

        self.models['x_learner'] = model
        self.results['x_learner'] = result

        print(f"    Qini coefficient: {metrics['qini_coefficient']:.4f}")
        print(f"    AUUC: {metrics['auuc']:.4f}")
        print(f"    Mean uplift: {uplift_scores.mean():.4f}")
        print(f"    Training time: {elapsed:.1f}s")
        return result

    def train_causal_forest(self, n_estimators: int = 100) -> dict:
        """Train Causal Forest using econml."""
        t0 = time.time()

        cf = CausalForestDML(
            model_y=xgb.XGBRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42,
            ),
            model_t=xgb.XGBRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42,
            ),
            n_estimators=n_estimators,
            min_samples_leaf=20,
            random_state=42,
        )

        cf.fit(
            Y=self.data['Y_train'].astype(float),
            T=self.data['T_train'].astype(float),
            X=self.data['X_train'],
        )

        # CATE: E[Y(1) - Y(0) | X] — negative means treatment reduces churn
        cate = cf.effect(self.data['X_test'])
        uplift_scores = -cate.flatten()  # Convert: positive = treatment helps

        metrics = self._evaluate_uplift(
            uplift_scores, self.data['T_test'], self.data['Y_test']
        )

        elapsed = time.time() - t0
        result = {
            'model': cf,
            'model_type': 'causal_forest',
            'uplift_scores': uplift_scores,
            'cate': cate.flatten(),
            'metrics': metrics,
            'train_time': elapsed,
        }

        self.models['causal_forest'] = cf
        self.results['causal_forest'] = result

        print(f"    Qini coefficient: {metrics['qini_coefficient']:.4f}")
        print(f"    AUUC: {metrics['auuc']:.4f}")
        print(f"    Mean CATE: {cate.mean():.4f}")
        print(f"    Mean uplift: {uplift_scores.mean():.4f}")
        print(f"    Training time: {elapsed:.1f}s")
        return result

    # =============================================================
    # EVALUATION
    # =============================================================

    def _evaluate_uplift(
        self, uplift_scores, treatment, outcome
    ) -> dict:
        """Evaluate uplift model with Qini coefficient and uplift by decile."""
        qini = self._qini_coefficient(uplift_scores, treatment, outcome)
        auuc = self._auuc(uplift_scores, treatment, outcome)
        deciles = self._uplift_by_decile(uplift_scores, treatment, outcome)
        calibration = self._uplift_calibration(uplift_scores, treatment, outcome)

        return {
            'qini_coefficient': qini,
            'auuc': auuc,
            'uplift_by_decile': deciles,
            'uplift_calibration': calibration,
        }

    def _auuc(self, uplift_scores, treatment, outcome) -> float:
        """Area under the uplift curve."""
        order = np.argsort(-uplift_scores)
        t_sorted = treatment[order]
        y_sorted = outcome[order]
        n = len(uplift_scores)

        uplift_curve = [0.0]
        for k in range(1, n + 1):
            t_k = t_sorted[:k]
            y_k = y_sorted[:k]

            t_mask = t_k == 1
            c_mask = t_k == 0

            if t_mask.sum() > 0 and c_mask.sum() > 0:
                churn_t = y_k[t_mask].mean()
                churn_c = y_k[c_mask].mean()
                uplift_curve.append((churn_c - churn_t) * (k / n))
            else:
                uplift_curve.append(uplift_curve[-1])

        x = np.linspace(0, 1, len(uplift_curve))
        return float(np.trapz(uplift_curve, x))

    def _qini_coefficient(self, uplift_scores, treatment, outcome) -> float:
        """Compute Qini coefficient (area between model and random)."""
        n = len(uplift_scores)
        order = np.argsort(-uplift_scores)  # Descending by predicted uplift

        t_sorted = treatment[order]
        y_sorted = outcome[order]

        n_t = treatment.sum()
        n_c = n - n_t

        # Cumulative Qini curve
        qini_values = [0.0]
        for k in range(1, n + 1):
            t_k = t_sorted[:k].sum()
            c_k = k - t_k

            if t_k > 0 and c_k > 0:
                # Incremental gains
                resp_t = y_sorted[:k][t_sorted[:k] == 1].sum()
                resp_c = y_sorted[:k][t_sorted[:k] == 0].sum()
                # For churn: we want FEWER churners in treatment
                # Qini = (churn_control_rate - churn_treatment_rate) * n_targeted
                qini_val = (resp_c / n_c - resp_t / n_t) * k
            else:
                qini_val = qini_values[-1] if qini_values else 0.0

            qini_values.append(qini_val)

        # Qini coefficient = area under Qini curve - area under random
        qini_area = np.trapz(qini_values, dx=1.0/n)
        random_area = qini_values[-1] / 2  # Triangle (random targeting)

        qini_coeff = qini_area - random_area
        return qini_coeff

    def _uplift_by_decile(
        self, uplift_scores, treatment, outcome, n_bins: int = 10
    ) -> pd.DataFrame:
        """Compute actual uplift in each score decile."""
        try:
            decile_labels = pd.qcut(
                uplift_scores, n_bins, labels=False, duplicates='drop'
            )
        except ValueError:
            decile_labels = pd.cut(
                uplift_scores, n_bins, labels=False, duplicates='drop'
            )

        results = []
        for d in sorted(set(decile_labels)):
            mask = decile_labels == d
            t_mask = mask & (treatment == 1)
            c_mask = mask & (treatment == 0)

            churn_t = outcome[t_mask].mean() if t_mask.sum() > 10 else np.nan
            churn_c = outcome[c_mask].mean() if c_mask.sum() > 10 else np.nan
            uplift = churn_c - churn_t if not (np.isnan(churn_t) or np.isnan(churn_c)) else 0

            results.append({
                'decile': int(d),
                'n_users': int(mask.sum()),
                'mean_pred_uplift': uplift_scores[mask].mean(),
                'churn_rate_treated': churn_t,
                'churn_rate_control': churn_c,
                'actual_uplift': uplift,
            })

        return pd.DataFrame(results)

    def _uplift_calibration(
        self, uplift_scores, treatment, outcome, n_bins: int = 10
    ) -> pd.DataFrame:
        """Compare predicted vs observed uplift in score bins."""
        try:
            bins = pd.qcut(uplift_scores, n_bins, labels=False, duplicates='drop')
        except ValueError:
            bins = pd.cut(uplift_scores, n_bins, labels=False, duplicates='drop')

        rows = []
        for b in sorted(set(bins)):
            mask = bins == b
            t_mask = mask & (treatment == 1)
            c_mask = mask & (treatment == 0)

            if t_mask.sum() < 10 or c_mask.sum() < 10:
                continue

            observed = outcome[c_mask].mean() - outcome[t_mask].mean()
            predicted = uplift_scores[mask].mean()
            rows.append({
                'bin': int(b),
                'n_users': int(mask.sum()),
                'predicted_uplift': float(predicted),
                'observed_uplift': float(observed),
            })

        return pd.DataFrame(rows)

    # =============================================================
    # QUADRANT CLASSIFICATION
    # =============================================================

    def classify_quadrants(
        self,
        model_name: str = 't_learner',
        churn_threshold: float = 0.05,
    ) -> np.ndarray:
        """Classify test users into the 4 uplift quadrants.

        Uses T-Learner's dual predictions for precise classification.
        """
        model = self.models[model_name]
        X_test = self.data['X_test']

        if model_name == 't_learner' and hasattr(model, 'predict_proba_both'):
            p_treated, p_control = model.predict_proba_both(X_test)
            uplift = p_control - p_treated

            quadrants = np.where(
                (p_control > churn_threshold) & (uplift > 0.01),
                'Persuadable',
                np.where(
                    (p_control <= churn_threshold) & (p_treated <= churn_threshold),
                    'Sure Thing',
                    np.where(
                        uplift < -0.01,
                        'Sleeping Dog',
                        'Lost Cause'
                    )
                )
            )
        else:
            # For non-T-Learner models, use uplift score only
            uplift = self.results[model_name]['uplift_scores']
            quadrants = np.where(
                uplift > 0.02, 'Persuadable',
                np.where(
                    uplift < -0.01, 'Sleeping Dog',
                    np.where(
                        self.data['Y_test'] == 0, 'Sure Thing', 'Lost Cause'
                    )
                )
            )

        return quadrants

    def _print_comparison(self):
        """Print model comparison."""
        print(f"\n{'='*60}")
        print(f"  UPLIFT MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Model':<23s} {'Qini':>9s} {'AUUC':>9s} {'Mean Uplift':>12s} {'Time':>8s}")
        print(f"  {'-'*66}")

        for name, result in self.results.items():
            qini = result['metrics']['qini_coefficient']
            auuc = result['metrics']['auuc']
            mean_up = result['uplift_scores'].mean()
            t = result['train_time']
            print(f"  {name:<23s} {qini:>9.4f} {auuc:>9.4f} {mean_up:>12.4f} {t:>7.1f}s")

        print(f"{'='*60}")
