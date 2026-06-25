-- =================================================================
-- PRISM Feature Engineering Pipeline (PySpark SQL / Databricks)
-- =================================================================
--
-- This script computes all churn prediction features using SQL.
-- Designed to run in a Databricks notebook as PySpark SQL.
--
-- SNAPSHOT APPROACH: All features are computed as of a configurable
-- snapshot date. No future data is used (no leakage).
--
-- To run in Databricks:
--   1. Upload Parquet files to DBFS
--   2. Create Delta tables from Parquet
--   3. Execute this SQL in a notebook cell
--
-- Prerequisites:
--   - Tables: users, daily_activity, subscriptions, support_tickets
-- =================================================================

-- Configuration: Set snapshot date
-- In production, this would be CURRENT_DATE or a parameterized date
SET snapshot_date = '2024-10-27';
SET target_end_date = '2024-11-26';


-- =================================================================
-- STEP 1: BASE TABLE — Eligible users + target variable
-- =================================================================
CREATE OR REPLACE TEMP VIEW base_users AS
SELECT
    u.user_id,
    u.signup_date,
    u.plan,
    u.monthly_price,
    u.platform,
    u.country,
    u.age_bucket,
    u.segment,
    -- Tenure as of snapshot
    DATEDIFF('${snapshot_date}', u.signup_date) AS tenure_at_snapshot,
    -- Plan encoding
    CASE u.plan
        WHEN 'free' THEN 0
        WHEN 'basic' THEN 1
        WHEN 'premium' THEN 2
    END AS plan_encoded,
    -- TARGET: Did this user churn within the next 30 days?
    CASE
        WHEN u.churn_date IS NOT NULL
             AND u.churn_date > '${snapshot_date}'
             AND u.churn_date <= '${target_end_date}'
        THEN 1
        ELSE 0
    END AS churned_target
FROM users u
WHERE
    -- Must have signed up before snapshot
    u.signup_date <= '${snapshot_date}'
    -- Must not have already churned before snapshot
    AND (u.churn_date IS NULL OR u.churn_date > '${snapshot_date}');


-- =================================================================
-- STEP 2: ENGAGEMENT FEATURES (multi-window aggregation)
-- =================================================================
CREATE OR REPLACE TEMP VIEW engagement_features AS
SELECT
    b.user_id,

    -- ── 3-day window ──
    COUNT(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 3 THEN 1 END) AS active_days_3d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 3 THEN a.n_sessions END), 0) AS total_sessions_3d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 3 THEN a.total_duration_min END), 0) AS total_duration_3d,

    -- ── 7-day window ──
    COUNT(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 7 THEN 1 END) AS active_days_7d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 7 THEN a.n_sessions END), 0) AS total_sessions_7d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 7 THEN a.total_duration_min END), 0) AS total_duration_7d,

    -- ── 14-day window ──
    COUNT(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 14 THEN 1 END) AS active_days_14d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 14 THEN a.n_sessions END), 0) AS total_sessions_14d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 14 THEN a.total_duration_min END), 0) AS total_duration_14d,

    -- ── 30-day window ──
    COUNT(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 30 THEN 1 END) AS active_days_30d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 30 THEN a.n_sessions END), 0) AS total_sessions_30d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 30 THEN a.total_duration_min END), 0) AS total_duration_30d,

    -- ── 60-day window ──
    COUNT(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 60 THEN 1 END) AS active_days_60d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 60 THEN a.n_sessions END), 0) AS total_sessions_60d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 60 THEN a.total_duration_min END), 0) AS total_duration_60d,

    -- ── 30-day averages ──
    COALESCE(AVG(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 30 THEN a.total_duration_min END), 0) AS avg_session_duration_30d,
    COALESCE(AVG(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 30 THEN a.n_sessions END), 0) AS avg_sessions_per_day_30d,
    COALESCE(AVG(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 30 THEN a.n_content_items END), 0) AS avg_content_items_30d,
    COALESCE(AVG(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 30 THEN a.n_distinct_categories END), 0) AS avg_categories_30d,
    COALESCE(AVG(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 30 THEN a.n_features_used END), 0) AS avg_features_used_30d,
    COALESCE(AVG(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 30 THEN a.n_searches END), 0) AS avg_searches_30d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 30 THEN a.n_shares END), 0) AS total_shares_30d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 30 THEN a.pages_viewed END), 0) AS total_pages_30d

FROM base_users b
LEFT JOIN daily_activity a
    ON b.user_id = a.user_id
    AND a.activity_date <= '${snapshot_date}'
    AND a.activity_date > DATE_SUB('${snapshot_date}', 60)
GROUP BY b.user_id;


-- =================================================================
-- STEP 3: ENGAGEMENT TRENDS
-- =================================================================
CREATE OR REPLACE TEMP VIEW trend_features AS
WITH recent_vs_older AS (
    SELECT
        b.user_id,
        -- Recent 7 days
        COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 7
                      THEN a.n_sessions END), 0) AS sessions_7d,
        COALESCE(AVG(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) <= 7
                      THEN a.total_duration_min END), 0) AS duration_7d,
        -- Older period (8-30 days ago)
        COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) BETWEEN 8 AND 30
                      THEN a.n_sessions END), 0) AS sessions_8_30d,
        COALESCE(AVG(CASE WHEN DATEDIFF('${snapshot_date}', a.activity_date) BETWEEN 8 AND 30
                      THEN a.total_duration_min END), 0) AS duration_8_30d,
        -- Recency
        MIN(DATEDIFF('${snapshot_date}', a.activity_date)) AS days_since_last_activity
    FROM base_users b
    LEFT JOIN daily_activity a
        ON b.user_id = a.user_id
        AND a.activity_date <= '${snapshot_date}'
        AND a.activity_date > DATE_SUB('${snapshot_date}', 60)
    GROUP BY b.user_id
)
SELECT
    user_id,
    COALESCE(days_since_last_activity, 999) AS days_since_last_activity,
    -- Engagement trend ratio (recent vs older, normalized to same period length)
    CASE
        WHEN sessions_8_30d / (23.0/7.0) > 0
        THEN LEAST(sessions_7d / (sessions_8_30d / (23.0/7.0)), 5.0)
        WHEN sessions_7d > 0 THEN 2.0
        ELSE 0.0
    END AS engagement_trend_ratio,
    -- Duration trend
    CASE
        WHEN duration_8_30d > 0
        THEN LEAST(duration_7d / duration_8_30d, 5.0)
        ELSE 1.0
    END AS duration_trend_ratio
FROM recent_vs_older;


-- =================================================================
-- STEP 4: TEMPORAL PATTERNS
-- =================================================================
CREATE OR REPLACE TEMP VIEW temporal_features AS
WITH daily_stats AS (
    SELECT
        a.user_id,
        a.activity_date,
        a.n_sessions,
        DAYOFWEEK(a.activity_date) IN (1, 7) AS is_weekend
    FROM daily_activity a
    INNER JOIN base_users b ON a.user_id = b.user_id
    WHERE a.activity_date <= '${snapshot_date}'
      AND a.activity_date > DATE_SUB('${snapshot_date}', 30)
),
user_temporal AS (
    SELECT
        user_id,
        -- Weekend ratio
        COALESCE(
            SUM(CASE WHEN is_weekend THEN n_sessions ELSE 0 END) * 1.0 /
            NULLIF(SUM(n_sessions), 0),
            0
        ) AS weekend_ratio,
        -- Engagement consistency (CV = std/mean)
        COALESCE(
            STDDEV(n_sessions) / NULLIF(AVG(n_sessions), 0),
            0
        ) AS engagement_cv
    FROM daily_stats
    GROUP BY user_id
)
SELECT * FROM user_temporal;


-- =================================================================
-- STEP 5: SUBSCRIPTION FEATURES
-- =================================================================
CREATE OR REPLACE TEMP VIEW subscription_features AS
SELECT
    b.user_id,
    COALESCE(COUNT(s.renewed), 0) AS n_total_periods,
    COALESCE(SUM(s.renewed), 0) AS n_renewals,
    COALESCE(SUM(s.payment_failed), 0) AS n_payment_failures,
    CASE WHEN SUM(s.payment_failed) > 0 THEN 1 ELSE 0 END AS has_payment_failure,
    COALESCE(
        SUM(s.payment_failed) * 1.0 / NULLIF(COUNT(s.renewed), 0),
        0
    ) AS payment_failure_rate,
    COUNT(DISTINCT s.plan) - 1 AS n_plan_changes,
    COALESCE(SUM(s.monthly_price), 0) AS total_revenue
FROM base_users b
LEFT JOIN subscriptions s
    ON b.user_id = s.user_id
    AND s.period_start <= '${snapshot_date}'
GROUP BY b.user_id;


-- =================================================================
-- STEP 6: SUPPORT TICKET FEATURES
-- =================================================================
CREATE OR REPLACE TEMP VIEW support_features AS
SELECT
    b.user_id,
    -- Total tickets
    COALESCE(COUNT(t.ticket_id), 0) AS total_tickets,
    COALESCE(AVG(t.satisfaction_score), 3.0) AS avg_satisfaction,
    COALESCE(AVG(t.resolution_hours), 0) AS avg_resolution_hours,
    -- Recent tickets
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', t.created_date) <= 30 THEN 1 ELSE 0 END), 0) AS tickets_last_30d,
    COALESCE(SUM(CASE WHEN DATEDIFF('${snapshot_date}', t.created_date) <= 7 THEN 1 ELSE 0 END), 0) AS tickets_last_7d,
    -- Category flags
    COALESCE(SUM(CASE WHEN t.category = 'cancellation_request' THEN 1 ELSE 0 END), 0) AS n_cancel_requests,
    CASE WHEN SUM(CASE WHEN t.category = 'cancellation_request' THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS has_cancel_request,
    COALESCE(SUM(CASE WHEN t.category = 'billing' THEN 1 ELSE 0 END), 0) AS n_billing_tickets,
    CASE WHEN SUM(CASE WHEN t.category = 'billing' THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS has_billing_issue
FROM base_users b
LEFT JOIN support_tickets t
    ON b.user_id = t.user_id
    AND t.created_date <= '${snapshot_date}'
GROUP BY b.user_id;


-- =================================================================
-- STEP 7: FINAL FEATURE TABLE (join all + derived features)
-- =================================================================
CREATE OR REPLACE TABLE churn_features AS
SELECT
    b.*,
    -- Engagement features
    e.active_days_3d, e.total_sessions_3d, e.total_duration_3d,
    e.active_days_7d, e.total_sessions_7d, e.total_duration_7d,
    e.active_days_14d, e.total_sessions_14d, e.total_duration_14d,
    e.active_days_30d, e.total_sessions_30d, e.total_duration_30d,
    e.active_days_60d, e.total_sessions_60d, e.total_duration_60d,
    e.avg_session_duration_30d, e.avg_sessions_per_day_30d,
    e.avg_content_items_30d, e.avg_categories_30d,
    e.avg_features_used_30d, e.avg_searches_30d,
    e.total_shares_30d, e.total_pages_30d,

    -- Trends
    t.days_since_last_activity,
    t.engagement_trend_ratio,
    t.duration_trend_ratio,

    -- Temporal
    tp.weekend_ratio,
    tp.engagement_cv,

    -- Subscription
    sub.n_total_periods, sub.n_renewals,
    sub.n_payment_failures, sub.has_payment_failure,
    sub.payment_failure_rate, sub.n_plan_changes,
    sub.total_revenue,

    -- Support
    sup.total_tickets, sup.avg_satisfaction, sup.avg_resolution_hours,
    sup.tickets_last_30d, sup.tickets_last_7d,
    sup.n_cancel_requests, sup.has_cancel_request,
    sup.n_billing_tickets, sup.has_billing_issue,

    -- ── Derived Features ──
    -- Sessions per dollar (value perception)
    CASE
        WHEN b.monthly_price > 0 THEN e.total_sessions_30d / b.monthly_price
        ELSE e.total_sessions_30d
    END AS sessions_per_dollar,

    -- Ticket rate (tickets per month of tenure)
    sup.total_tickets / GREATEST(b.tenure_at_snapshot / 30.0, 1.0) AS ticket_rate_monthly,

    -- Activity acceleration
    CASE
        WHEN e.total_sessions_30d > 0
        THEN LEAST((e.total_sessions_7d * 30.0 / 7.0) / e.total_sessions_30d, 5.0)
        ELSE 0.0
    END AS activity_acceleration,

    -- Tenure x Engagement interaction
    LN(1 + b.tenure_at_snapshot) * LN(1 + e.total_sessions_30d) AS tenure_x_engagement,

    -- Support escalation (recent / total)
    CASE
        WHEN sup.total_tickets > 0
        THEN sup.tickets_last_7d * 1.0 / sup.total_tickets
        ELSE 0.0
    END AS support_escalation

FROM base_users b
LEFT JOIN engagement_features e ON b.user_id = e.user_id
LEFT JOIN trend_features t ON b.user_id = t.user_id
LEFT JOIN temporal_features tp ON b.user_id = tp.user_id
LEFT JOIN subscription_features sub ON b.user_id = sub.user_id
LEFT JOIN support_features sup ON b.user_id = sup.user_id;


-- =================================================================
-- VERIFICATION QUERIES
-- =================================================================

-- Row count
SELECT COUNT(*) AS total_users, SUM(churned_target) AS churned, AVG(churned_target) AS churn_rate
FROM churn_features;

-- Feature completeness (null check)
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN days_since_last_activity IS NULL THEN 1 ELSE 0 END) AS null_recency,
    SUM(CASE WHEN total_sessions_30d IS NULL THEN 1 ELSE 0 END) AS null_sessions,
    SUM(CASE WHEN engagement_trend_ratio IS NULL THEN 1 ELSE 0 END) AS null_trend,
    SUM(CASE WHEN total_tickets IS NULL THEN 1 ELSE 0 END) AS null_tickets
FROM churn_features;

-- Feature distributions by churn status
SELECT
    churned_target,
    COUNT(*) AS n,
    AVG(days_since_last_activity) AS avg_recency,
    AVG(total_sessions_30d) AS avg_sessions_30d,
    AVG(engagement_trend_ratio) AS avg_trend,
    AVG(total_tickets) AS avg_tickets,
    AVG(has_cancel_request) AS pct_cancel_request
FROM churn_features
GROUP BY churned_target;
