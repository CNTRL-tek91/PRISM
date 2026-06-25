# PRISM Power BI Dashboard Notes

## Scope

This dashboard has 4 pages:
1. Executive Summary
2. Risk Segmentation
3. Uplift Analysis
4. Campaign ROI

## Data source files

Load these CSVs from `data/powerbi/`:
- `executive_kpis.csv`
- `user_scores.csv`
- `campaign_recommendations.csv`
- `roi_strategy_comparison.csv`
- `roi_budget_sensitivity.csv`

## Data model

Primary key:
- `user_scores[user_id]`

Related fact table:
- `campaign_recommendations[user_id]` (many-to-one to `user_scores`)

Independent summary tables:
- `executive_kpis`
- `roi_strategy_comparison`
- `roi_budget_sensitivity`

## Suggested DAX measures

- `Users Scored = DISTINCTCOUNT(user_scores[user_id])`
- `Avg LTV 12m = AVERAGE(user_scores[ltv_12m])`
- `Avg Uplift = AVERAGE(user_scores[best_uplift_score])`
- `PRISM Targets = CALCULATE(COUNTROWS(user_scores), user_scores[is_targeted_prism] = 1)`
- `Strategy Net ROI = SUM(roi_strategy_comparison[net_roi_usd])`

## Page design

## 1) Executive Summary
- KPI cards: users scored, avg LTV, avg uplift, best strategy, best net ROI.
- Bar chart: strategy net ROI (`roi_strategy_comparison`).
- Line chart: budget sensitivity (`roi_budget_sensitivity`).

## 2) Risk Segmentation
- Bar chart: user count by `risk_tier`.
- Histogram/box: `risk_score` distribution.
- Matrix: `risk_tier` x `ltv_tier_12m` with avg LTV.
- Slicer: `risk_tier`, `ltv_tier_12m`.

## 3) Uplift Analysis
- Bar chart: user count by `quadrant`.
- Scatter: `best_uplift_score` vs `ltv_12m`, color by `quadrant`.
- Bar chart: campaign type mix from `campaign_recommendations`.

## 4) Campaign ROI
- Clustered bars: revenue saved / campaign cost / net ROI by strategy.
- Line chart: budget vs net ROI by strategy.
- Table: top recommended users from `campaign_recommendations` with `target_rank <= 100`.

## Build checklist

- Confirm numeric columns imported as numeric (not text).
- Format ROI/LTV fields as currency.
- Add slicers for strategy and risk tier.
- Add tooltips for user-level scatter chart.
- Save as `powerbi/prism_dashboard.pbix`.

