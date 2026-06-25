"""
PRISM Data Simulator
====================

Generates synthetic subscription platform data for churn prediction,
uplift modeling, and customer lifetime value estimation.

Tables Generated
----------------
1. users          — 200K user profiles with demographics and churn timing
2. daily_activity — ~10M daily engagement records with realistic decay
3. subscriptions  — ~400K billing records with payment failures
4. campaigns      — ~60K past retention campaign interactions (treatment/control)
5. support_tickets — ~30K customer support interactions
6. ground_truth   — Hidden causal parameters for uplift model evaluation

Design Philosophy
-----------------
The data embeds realistic patterns that models should learn:
- Engagement declines 2-4 weeks before churn (primary signal)
- Support ticket frequency increases before churn
- Payment failures correlate with involuntary churn
- Campaign treatment effects vary by hidden 'persuadability' (for uplift modeling)
- Tenure has a protective effect (longer tenure → lower churn)
- Premium plan users churn less than free users

Usage
-----
    from src.data_simulator import SubscriptionDataSimulator

    sim = SubscriptionDataSimulator(n_users=200_000, seed=42)
    data = sim.generate_all()
    sim.save_to_parquet(data, output_dir='data/raw')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import logging
import time

logger = logging.getLogger(__name__)


class SubscriptionDataSimulator:
    """Generates realistic subscription platform data with embedded
    causal treatment effects for uplift modeling ground truth.

    Parameters
    ----------
    n_users : int
        Number of users to simulate (default: 200,000).
    observation_days : int
        Length of observation window in days (default: 365).
    seed : int
        Random seed for reproducibility (default: 42).
    """

    # ─── Segment Configuration ────────────────────────────────
    # Each segment has different engagement levels and churn rates.
    # The model should learn to distinguish these from features alone.

    SEGMENT_CONFIG = {
        'power': {
            'fraction': 0.10,
            'monthly_churn_rate': 0.005,    # ~6% annual churn
            'daily_active_rate': 0.80,      # Logs in 80% of days
            'avg_sessions': 3.0,            # Sessions per active day
            'avg_duration_min': 45,         # Minutes per active day
        },
        'regular': {
            'fraction': 0.35,
            'monthly_churn_rate': 0.020,    # ~21% annual churn
            'daily_active_rate': 0.45,
            'avg_sessions': 1.5,
            'avg_duration_min': 25,
        },
        'casual': {
            'fraction': 0.35,
            'monthly_churn_rate': 0.040,    # ~38% annual churn
            'daily_active_rate': 0.18,
            'avg_sessions': 1.0,
            'avg_duration_min': 15,
        },
        'at_risk': {
            'fraction': 0.20,
            'monthly_churn_rate': 0.100,    # ~72% annual churn
            'daily_active_rate': 0.08,
            'avg_sessions': 0.8,
            'avg_duration_min': 8,
        },
    }

    # ─── Plan Configuration ───────────────────────────────────
    PLAN_CONFIG = {
        'free':    {'fraction': 0.25, 'price': 0.00,  'churn_modifier': 1.30},
        'basic':   {'fraction': 0.45, 'price': 9.99,  'churn_modifier': 1.00},
        'premium': {'fraction': 0.30, 'price': 14.99, 'churn_modifier': 0.70},
    }

    # ─── Demographic Distributions ────────────────────────────
    PLATFORMS = ['ios', 'android', 'web']
    PLATFORM_WEIGHTS = [0.45, 0.35, 0.20]

    COUNTRIES = ['US', 'UK', 'DE', 'JP', 'BR', 'IN']
    COUNTRY_WEIGHTS = [0.40, 0.15, 0.10, 0.10, 0.12, 0.13]

    AGE_BUCKETS = ['18-24', '25-34', '35-44', '45-54', '55+']
    AGE_WEIGHTS = [0.20, 0.30, 0.25, 0.15, 0.10]

    # ─── Other Constants ──────────────────────────────────────
    SUPPORT_CATEGORIES = ['billing', 'technical', 'content', 'cancellation_request', 'account']
    CAMPAIGN_TYPES = ['email', 'push_notification', 'discount_offer', 'upgrade_offer']

    def __init__(self, n_users: int = 200_000, observation_days: int = 365, seed: int = 42):
        self.n_users = n_users
        self.observation_days = observation_days
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.start_date = pd.Timestamp('2024-01-01')
        self.end_date = self.start_date + pd.Timedelta(days=observation_days - 1)

        logger.info(
            f"Simulator initialized: {n_users:,} users, "
            f"{observation_days} days ({self.start_date.date()} to {self.end_date.date()}), "
            f"seed={seed}"
        )

    # ═══════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all tables and return as a dictionary.

        Returns
        -------
        dict
            Keys: 'users', 'daily_activity', 'subscriptions',
                  'campaigns', 'support_tickets', 'ground_truth'
        """
        t0 = time.time()
        print(f"{'='*60}")
        print(f"  PRISM Data Simulator")
        print(f"  Users: {self.n_users:,} | Days: {self.observation_days}")
        print(f"{'='*60}")

        # Step 1: Users (includes hidden internal columns prefixed with _)
        print("\n[1/6] Generating users...")
        users_internal = self._generate_users()
        print(f"       [OK] {len(users_internal):,} users "
              f"({users_internal['churned'].mean():.1%} churned)")

        # Step 2: Daily activity
        print("\n[2/6] Generating daily activity (this may take 1-2 minutes)...")
        daily_activity = self._generate_daily_activity(users_internal)
        print(f"       [OK] {len(daily_activity):,} daily activity records")

        # Step 3: Subscriptions
        print("\n[3/6] Generating subscription records...")
        subscriptions = self._generate_subscriptions(users_internal)
        print(f"       [OK] {len(subscriptions):,} subscription records")

        # Step 4: Campaigns
        print("\n[4/6] Generating campaign history...")
        campaigns = self._generate_campaigns(users_internal)
        print(f"       [OK] {len(campaigns):,} campaign records")

        # Step 5: Support tickets
        print("\n[5/6] Generating support tickets...")
        support_tickets = self._generate_support_tickets(users_internal)
        print(f"       [OK] {len(support_tickets):,} support tickets")

        # Step 6: Separate ground truth from user-visible data
        print("\n[6/6] Preparing final tables...")
        ground_truth = users_internal[['user_id', 'persuadability', 'segment']].copy()
        ground_truth.rename(columns={'segment': 'true_segment'}, inplace=True)

        # Clean users table: drop internal columns
        internal_cols = [c for c in users_internal.columns if c.startswith('_')]
        internal_cols += ['persuadability']
        users_clean = users_internal.drop(columns=internal_cols)

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  Generation complete in {elapsed:.1f}s")
        print(f"  Tables: users ({len(users_clean):,}), "
              f"daily_activity ({len(daily_activity):,}),")
        print(f"          subscriptions ({len(subscriptions):,}), "
              f"campaigns ({len(campaigns):,}),")
        print(f"          support_tickets ({len(support_tickets):,}), "
              f"ground_truth ({len(ground_truth):,})")
        print(f"{'='*60}")

        return {
            'users': users_clean,
            'daily_activity': daily_activity,
            'subscriptions': subscriptions,
            'campaigns': campaigns,
            'support_tickets': support_tickets,
            'ground_truth': ground_truth,
        }

    def save_to_parquet(self, data: Dict[str, pd.DataFrame], output_dir: str) -> None:
        """Save all tables as Parquet files.

        Parameters
        ----------
        data : dict
            Output from generate_all().
        output_dir : str
            Directory path to save files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, df in data.items():
            filepath = output_path / f"{name}.parquet"
            df.to_parquet(filepath, engine='pyarrow', index=False)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  Saved {name}.parquet ({len(df):,} rows, {size_mb:.1f} MB)")

        print(f"\nAll files saved to: {output_path.resolve()}")

    def save_to_csv(self, data: Dict[str, pd.DataFrame], output_dir: str) -> None:
        """Save all tables as CSV files (for Power BI or manual inspection)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, df in data.items():
            filepath = output_path / f"{name}.csv"
            df.to_csv(filepath, index=False)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  Saved {name}.csv ({len(df):,} rows, {size_mb:.1f} MB)")

    # ═══════════════════════════════════════════════════════════
    # PRIVATE: TABLE GENERATORS
    # ═══════════════════════════════════════════════════════════

    def _generate_users(self) -> pd.DataFrame:
        """Generate the users table with demographics, plan, and churn timing.

        Internal columns (prefixed with _) are used by other generators
        and stripped before the final output.
        """
        n = self.n_users

        # ─── Demographics ─────────────────────────────────────
        segments = self.rng.choice(
            list(self.SEGMENT_CONFIG.keys()), n,
            p=[c['fraction'] for c in self.SEGMENT_CONFIG.values()]
        )
        plans = self.rng.choice(
            list(self.PLAN_CONFIG.keys()), n,
            p=[c['fraction'] for c in self.PLAN_CONFIG.values()]
        )
        platforms = self.rng.choice(self.PLATFORMS, n, p=self.PLATFORM_WEIGHTS)
        countries = self.rng.choice(self.COUNTRIES, n, p=self.COUNTRY_WEIGHTS)
        age_buckets = self.rng.choice(self.AGE_BUCKETS, n, p=self.AGE_WEIGHTS)

        # ─── Signup Dates ─────────────────────────────────────
        # Users signed up throughout the observation period and before
        # Spread signups: 40% before observation, 60% during
        signup_offsets = self.rng.uniform(-180, self.observation_days * 0.7, n).astype(int)
        signup_offsets = np.clip(signup_offsets, -180, self.observation_days - 30)
        signup_dates = self.start_date + pd.to_timedelta(signup_offsets, unit='D')

        # Internal: day offset from start_date (for fast computation)
        signup_day = signup_offsets.copy()

        # ─── Monthly Price ────────────────────────────────────
        monthly_price = np.array([self.PLAN_CONFIG[p]['price'] for p in plans])

        # ─── Persuadability (HIDDEN — ground truth for uplift) ─
        # Beta(2, 5) → most users have low persuadability
        # ~30% have persuadability > 0.3 (the "persuadable" segment)
        persuadability = self.rng.beta(2, 5, n)

        # ─── Churn Timing ─────────────────────────────────────
        # Monthly churn rate per user = segment_base × plan_modifier × individual noise
        segment_churn_rates = np.array([
            self.SEGMENT_CONFIG[s]['monthly_churn_rate'] for s in segments
        ])
        plan_modifiers = np.array([
            self.PLAN_CONFIG[p]['churn_modifier'] for p in plans
        ])
        # Individual variation (lognormal noise around base rate)
        individual_noise = self.rng.lognormal(0, 0.3, n)

        monthly_churn_rate = segment_churn_rates * plan_modifiers * individual_noise
        monthly_churn_rate = np.clip(monthly_churn_rate, 0.001, 0.30)

        # Convert monthly rate to daily hazard
        daily_hazard = 1 - (1 - monthly_churn_rate) ** (1/30)

        # Draw time-to-churn from geometric distribution
        # (discrete analog of exponential — days until churn event)
        time_to_churn_from_signup = self.rng.geometric(daily_hazard)

        # Churn day (relative to start_date)
        churn_day = signup_day + time_to_churn_from_signup

        # Users who churn after observation window are censored (not churned)
        churned = churn_day <= self.observation_days
        churn_day_clipped = np.where(churned, churn_day, self.observation_days + 999)

        # Churn dates (NaT for censored users)
        churn_dates = pd.NaT
        churn_date_series = pd.Series(
            [self.start_date + pd.Timedelta(days=int(d)) if c else pd.NaT
             for d, c in zip(churn_day, churned)],
            dtype='datetime64[ns]'
        )

        # End of active period (churn day or observation end)
        end_day = np.where(churned, churn_day, self.observation_days)

        # Tenure in days (from signup to churn or observation end)
        tenure_days = end_day - signup_day
        tenure_days = np.maximum(tenure_days, 1)

        # ─── Engagement Parameters (internal) ─────────────────
        daily_active_rate = np.array([
            self.SEGMENT_CONFIG[s]['daily_active_rate'] for s in segments
        ]) * self.rng.uniform(0.7, 1.3, n)  # Individual variation
        daily_active_rate = np.clip(daily_active_rate, 0.02, 0.95)

        avg_sessions = np.array([
            self.SEGMENT_CONFIG[s]['avg_sessions'] for s in segments
        ]) * self.rng.uniform(0.6, 1.4, n)

        avg_duration = np.array([
            self.SEGMENT_CONFIG[s]['avg_duration_min'] for s in segments
        ]) * self.rng.uniform(0.5, 1.5, n)

        # ─── Assemble DataFrame ───────────────────────────────
        users = pd.DataFrame({
            'user_id': np.arange(n),
            'signup_date': signup_dates,
            'plan': plans,
            'monthly_price': monthly_price,
            'platform': platforms,
            'country': countries,
            'age_bucket': age_buckets,
            'segment': segments,
            'churn_date': churn_date_series,
            'churned': churned.astype(int),
            'tenure_days': tenure_days,
            # Internal columns (used by other generators, removed before output)
            '_signup_day': signup_day,
            '_end_day': end_day,
            '_churn_day': churn_day_clipped,
            '_daily_active_rate': daily_active_rate,
            '_avg_sessions': avg_sessions,
            '_avg_duration': avg_duration,
            '_monthly_churn_rate': monthly_churn_rate,
            'persuadability': persuadability,
        })

        return users

    def _generate_daily_activity(self, users: pd.DataFrame) -> pd.DataFrame:
        """Generate daily activity records for all users.

        Activity rate declines as churn approaches, creating the
        engagement-decline signal that models should learn.

        Uses vectorized day-by-day generation for efficiency.
        """
        # Pre-extract arrays for speed
        user_ids = users['user_id'].values
        signup_days = users['_signup_day'].values
        end_days = users['_end_day'].values
        churn_days = users['_churn_day'].values
        active_rates = users['_daily_active_rate'].values
        avg_sessions = users['_avg_sessions'].values
        avg_durations = users['_avg_duration'].values

        # Pre-allocate lists for each column (faster than building DataFrames in loop)
        all_user_ids = []
        all_day_offsets = []
        all_sessions = []
        all_duration = []
        all_content = []
        all_categories = []
        all_features = []
        all_searches = []
        all_shares = []
        all_pages = []

        for day in tqdm(range(self.observation_days), desc="       Days", leave=False):
            # ─── Who is active today? ─────────────────────────
            # User must have signed up before today and not yet churned
            active_mask = (signup_days <= day) & (end_days > day)
            active_idx = np.where(active_mask)[0]
            n_active = len(active_idx)

            if n_active == 0:
                continue

            # ─── Engagement Decay ─────────────────────────────
            # Activity drops as churn approaches (sigmoid decay)
            # Full activity until 60 days before churn, then rapid decline
            days_until_end = end_days[active_idx] - day
            decay = np.where(
                days_until_end < 60,
                1.0 / (1.0 + np.exp(-(days_until_end - 15) / 8)),
                1.0
            )

            # ─── Day-of-week effect ───────────────────────────
            dow = day % 7
            dow_factor = 1.12 if dow >= 5 else 1.0  # Weekends slightly more active

            # ─── Login probability ────────────────────────────
            login_prob = active_rates[active_idx] * decay * dow_factor
            login_prob = np.clip(login_prob, 0, 0.99)

            # ─── Who logs in today? ───────────────────────────
            logged_in = self.rng.random(n_active) < login_prob
            li_idx = active_idx[logged_in]
            n_logged = len(li_idx)

            if n_logged == 0:
                continue

            # ─── Generate engagement metrics ──────────────────
            li_decay = decay[logged_in]
            li_sessions = avg_sessions[li_idx]
            li_duration = avg_durations[li_idx]

            sessions = self.rng.poisson(
                np.maximum(li_sessions * li_decay, 0.3)
            ).astype(np.int32)
            sessions = np.maximum(sessions, 1)

            duration = self.rng.lognormal(
                np.log(np.maximum(li_duration * li_decay, 1)), 0.5
            ).astype(np.int32)
            duration = np.maximum(duration, 1)

            content = self.rng.poisson(
                np.maximum(sessions * 1.8 * li_decay, 0.3)
            ).astype(np.int32)

            categories = np.minimum(
                self.rng.poisson(np.maximum(1.5 * li_decay, 0.2)),
                5
            ).astype(np.int32)
            categories = np.maximum(categories, 1)

            features = self.rng.poisson(
                np.maximum(2.0 * li_decay, 0.2)
            ).astype(np.int32)

            searches = self.rng.poisson(
                np.maximum(0.8 * li_decay, 0.05)
            ).astype(np.int32)

            shares = self.rng.poisson(0.1, n_logged).astype(np.int32)

            pages = self.rng.poisson(
                np.maximum(sessions * 3.5, 1)
            ).astype(np.int32)
            pages = np.maximum(pages, 1)

            # ─── Append to lists ──────────────────────────────
            all_user_ids.append(user_ids[li_idx])
            all_day_offsets.append(np.full(n_logged, day, dtype=np.int32))
            all_sessions.append(sessions)
            all_duration.append(duration)
            all_content.append(content)
            all_categories.append(categories)
            all_features.append(features)
            all_searches.append(searches)
            all_shares.append(shares)
            all_pages.append(pages)

        # ─── Assemble DataFrame ───────────────────────────────
        day_offsets = np.concatenate(all_day_offsets)

        daily_activity = pd.DataFrame({
            'user_id': np.concatenate(all_user_ids),
            'activity_date': self.start_date + pd.to_timedelta(day_offsets, unit='D'),
            'n_sessions': np.concatenate(all_sessions),
            'total_duration_min': np.concatenate(all_duration),
            'n_content_items': np.concatenate(all_content),
            'n_distinct_categories': np.concatenate(all_categories),
            'n_features_used': np.concatenate(all_features),
            'n_searches': np.concatenate(all_searches),
            'n_shares': np.concatenate(all_shares),
            'pages_viewed': np.concatenate(all_pages),
        })

        return daily_activity

    def _generate_subscriptions(self, users: pd.DataFrame) -> pd.DataFrame:
        """Generate monthly subscription/billing records.

        Each user has one record per billing month they were active.
        Includes plan changes and payment failure events.
        """
        records = []

        for _, user in tqdm(users.iterrows(), total=len(users),
                            desc="       Subscriptions", leave=False,
                            miniters=10000):
            # Only paid users have subscription records
            if user['plan'] == 'free':
                continue

            signup_day = int(user['_signup_day'])
            end_day = int(user['_end_day'])
            active_months = max(1, (end_day - signup_day) // 30)

            current_plan = user['plan']
            price = self.PLAN_CONFIG[current_plan]['price']

            for month in range(active_months):
                period_start_day = signup_day + month * 30
                period_end_day = period_start_day + 30

                if period_start_day > self.observation_days:
                    break

                # Payment failure (~3% chance per month, higher for at_risk)
                fail_rate = 0.03 if user['segment'] != 'at_risk' else 0.08
                payment_failed = int(self.rng.random() < fail_rate)

                # Plan change (~2% chance per month)
                plan_changed = int(self.rng.random() < 0.02)
                if plan_changed:
                    other_plans = [p for p in ['basic', 'premium'] if p != current_plan]
                    current_plan = self.rng.choice(other_plans)
                    price = self.PLAN_CONFIG[current_plan]['price']

                # Renewal (all months except last if churned)
                is_last_month = (month == active_months - 1) and user['churned'] == 1
                renewed = 0 if is_last_month else 1

                records.append({
                    'user_id': user['user_id'],
                    'period_start': self.start_date + pd.Timedelta(days=period_start_day),
                    'period_end': self.start_date + pd.Timedelta(days=period_end_day),
                    'plan': current_plan,
                    'monthly_price': price,
                    'payment_failed': payment_failed,
                    'renewed': renewed,
                })

        return pd.DataFrame(records)

    def _generate_campaigns(self, users: pd.DataFrame) -> pd.DataFrame:
        """Generate historical retention campaign records.

        Simulates a past A/B test where a subset of users received
        a retention intervention. Treatment effects depend on the
        user's hidden persuadability score.

        This is the GROUND TRUTH for uplift modeling.
        """
        # Select users who were active mid-observation for the campaign
        campaign_day = self.observation_days // 2  # Campaign ran at midpoint
        eligible_mask = (
            (users['_signup_day'] < campaign_day) &
            (users['_end_day'] > campaign_day)
        )
        eligible = users[eligible_mask]

        # Sample ~30% of eligible users for the campaign
        n_campaign = min(int(len(eligible) * 0.30), 60_000)
        campaign_users = eligible.sample(n=n_campaign, random_state=self.seed)

        # Random treatment assignment (50/50)
        treatment = self.rng.binomial(1, 0.5, n_campaign)

        # Campaign type distribution
        campaign_type = self.rng.choice(
            self.CAMPAIGN_TYPES, n_campaign,
            p=[0.35, 0.25, 0.25, 0.15]
        )

        # ─── Outcome: Churned within 30 days ─────────────────
        # Base 30-day churn probability
        base_churn_30d = campaign_users['_monthly_churn_rate'].values

        # Treatment effect depends on persuadability
        persuadability = campaign_users['persuadability'].values
        treatment_effect = np.where(
            persuadability > 0.30,
            # Persuadable users: intervention REDUCES churn by 10-20pp
            -self.rng.uniform(0.10, 0.20, n_campaign),
            np.where(
                persuadability > 0.15,
                # Neutral users: minimal effect
                self.rng.uniform(-0.02, 0.02, n_campaign),
                # "Sleeping dogs": intervention slightly INCREASES churn
                self.rng.uniform(0.01, 0.05, n_campaign),
            )
        )

        # Final churn probability: base + treatment × treatment_effect
        churn_prob = base_churn_30d + treatment * treatment_effect
        churn_prob = np.clip(churn_prob, 0.01, 0.95)

        # Draw outcomes
        churned_within_30d = self.rng.binomial(1, churn_prob)

        # Did they respond/convert from the campaign offer?
        # Conversion is correlated with positive treatment effect
        conversion_prob = np.where(treatment == 1, 0.15, 0.05)
        conversion_prob *= np.where(persuadability > 0.3, 1.5, 0.8)
        conversion_prob = np.clip(conversion_prob, 0.02, 0.40)
        converted = self.rng.binomial(1, conversion_prob)

        # Campaign date (with some spread)
        campaign_dates = (
            self.start_date
            + pd.Timedelta(days=campaign_day)
            + pd.to_timedelta(self.rng.integers(0, 14, n_campaign), unit='D')
        )

        campaigns = pd.DataFrame({
            'campaign_id': np.arange(n_campaign),
            'user_id': campaign_users['user_id'].values,
            'campaign_date': campaign_dates,
            'campaign_type': campaign_type,
            'treatment': treatment,
            'converted': converted,
            'churned_within_30d': churned_within_30d,
        })

        return campaigns

    def _generate_support_tickets(self, users: pd.DataFrame) -> pd.DataFrame:
        """Generate customer support ticket records.

        Ticket frequency is higher for users who will churn,
        especially in the categories 'billing' and 'cancellation_request'.
        """
        records = []

        for _, user in tqdm(users.iterrows(), total=len(users),
                            desc="       Support tickets", leave=False,
                            miniters=20000):
            signup_day = int(user['_signup_day'])
            end_day = int(min(user['_end_day'], self.observation_days))
            active_period = max(1, end_day - signup_day)

            # Base ticket rate (per 30 days)
            if user['segment'] == 'at_risk':
                base_rate = 0.35
            elif user['segment'] == 'casual':
                base_rate = 0.15
            elif user['segment'] == 'regular':
                base_rate = 0.08
            else:
                base_rate = 0.04

            # Churners submit more tickets before churning
            if user['churned'] == 1:
                base_rate *= 2.5

            # Number of tickets during active period
            expected_tickets = base_rate * (active_period / 30)
            n_tickets = self.rng.poisson(expected_tickets)

            if n_tickets == 0:
                continue

            # Generate ticket dates (clustered toward end of active period for churners)
            if user['churned'] == 1:
                # Beta distribution skewed toward end
                offsets = self.rng.beta(3, 1.5, n_tickets)
            else:
                # Uniform distribution
                offsets = self.rng.uniform(0, 1, n_tickets)

            ticket_days = signup_day + (offsets * active_period).astype(int)
            ticket_days = np.clip(ticket_days, signup_day, end_day - 1)

            # Category distribution (churners more likely to file billing/cancellation)
            if user['churned'] == 1:
                cat_weights = [0.30, 0.15, 0.10, 0.35, 0.10]
            else:
                cat_weights = [0.20, 0.30, 0.25, 0.05, 0.20]

            categories = self.rng.choice(self.SUPPORT_CATEGORIES, n_tickets, p=cat_weights)

            priorities = self.rng.choice(
                ['low', 'medium', 'high'], n_tickets, p=[0.40, 0.40, 0.20]
            )

            resolution_hours = self.rng.lognormal(2.5, 1.0, n_tickets).round(1)
            resolution_hours = np.clip(resolution_hours, 0.5, 168)  # Max 1 week

            # Satisfaction: lower for churners, especially cancellation requests
            base_satisfaction = 3.5 if user['churned'] == 0 else 2.5
            satisfaction = self.rng.normal(base_satisfaction, 0.8, n_tickets).round(0)
            satisfaction = np.clip(satisfaction, 1, 5).astype(int)

            for i in range(n_tickets):
                records.append({
                    'user_id': user['user_id'],
                    'created_date': self.start_date + pd.Timedelta(days=int(ticket_days[i])),
                    'category': categories[i],
                    'priority': priorities[i],
                    'resolution_hours': resolution_hours[i],
                    'satisfaction_score': satisfaction[i],
                })

        tickets = pd.DataFrame(records)
        if len(tickets) > 0:
            tickets.insert(0, 'ticket_id', np.arange(len(tickets)))
        return tickets


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    """Generate PRISM dataset and save to data/raw/."""
    logging.basicConfig(level=logging.INFO)

    sim = SubscriptionDataSimulator(
        n_users=200_000,
        observation_days=365,
        seed=42,
    )

    data = sim.generate_all()

    # Save as Parquet (primary format)
    output_dir = Path(__file__).parent.parent / 'data' / 'raw'
    sim.save_to_parquet(data, str(output_dir))

    # Print summary statistics
    print("\n" + "="*60)
    print("  DATA SUMMARY")
    print("="*60)

    users = data['users']
    print(f"\n  Users: {len(users):,}")
    print(f"  Churn rate: {users['churned'].mean():.1%}")
    print(f"  Plan distribution:")
    for plan, count in users['plan'].value_counts().items():
        print(f"    {plan:>8s}: {count:>7,} ({count/len(users):.1%})")
    print(f"  Segment distribution:")
    for seg, count in users['segment'].value_counts().items():
        print(f"    {seg:>8s}: {count:>7,} ({count/len(users):.1%})")
    print(f"  Churn rate by segment:")
    for seg in ['power', 'regular', 'casual', 'at_risk']:
        seg_users = users[users['segment'] == seg]
        print(f"    {seg:>8s}: {seg_users['churned'].mean():.1%}")

    da = data['daily_activity']
    print(f"\n  Daily activity records: {len(da):,}")
    print(f"  Avg sessions/day: {da['n_sessions'].mean():.1f}")
    print(f"  Avg duration/day: {da['total_duration_min'].mean():.0f} min")

    camp = data['campaigns']
    print(f"\n  Campaign records: {len(camp):,}")
    print(f"  Treatment rate: {camp['treatment'].mean():.1%}")
    print(f"  Churn rate (control): "
          f"{camp[camp['treatment']==0]['churned_within_30d'].mean():.1%}")
    print(f"  Churn rate (treated): "
          f"{camp[camp['treatment']==1]['churned_within_30d'].mean():.1%}")

    print(f"\n  Support tickets: {len(data['support_tickets']):,}")
    print(f"\n  Ground truth (for evaluation): {len(data['ground_truth']):,}")


if __name__ == '__main__':
    main()
