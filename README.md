<div align="center">

# PRISM

**Predictive Retention and Intelligent Subscriber Modeling**

An end-to-end churn intelligence project that goes past "who will churn" to answer "who is worth saving, and what is the financial impact."

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-f7931e?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194e2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

## Purpose

PRISM is my attempt to model the full decision the way a real subscription business would, from risk to persuadability to dollars.

It answers three questions in order:

1. Who is likely to churn, and when?
2. Who is actually persuadable by a retention campaign?
3. What is the expected financial impact of each targeting strategy?

## What it does

The pipeline runs in stages, each building on the last.

**Survival modeling.** Rather than a binary "churn in 30 days" label, I model time to churn with Cox Proportional Hazards, Gradient Boosted Survival Analysis, and a Random Survival Forest. This gives a survival curve per user and a risk tier, not just a yes or no.

**Causal uplift modeling.** A high-risk user is not the same as a persuadable user, and some users actually churn faster if you bother them. I estimate individual treatment effects with five uplift models (T-Learner, S-Learner, X-Learner, Causal Forest, and an Uplift Random Forest) and sort users into Persuadable, Sure Thing, Lost Cause, and Sleeping Dog groups.

**Lifetime value.** A discounted cash flow model combines each user's survival curve with their monthly revenue to estimate 6, 12, and 24 month LTV.

**Campaign optimization.** Finally I compare targeting strategies under a fixed budget, ranking users by expected uplift times LTV, and report the ROI of each approach.

## Key results

The data is simulated, with a hidden persuadability score baked in so the uplift models can be measured against ground truth.

| Result | Value |
|--------|-------|
| Best survival model (concordance index) | **0.895** (Gradient Boosted Survival) |
| Random Survival Forest | 0.892 |
| Best uplift model | S-Learner (highest Qini) |
| Audience split | 71.8% Sure Thing, 17.8% Persuadable |
| Mean 12-month LTV | $88.30 |
| Best strategy | Uplift x LTV targeting |

The most useful finding is a business one. Targeting by predicted uplift times LTV saved about 75 percent more revenue than targeting by churn risk alone, for the same spend. Under the current cost assumptions every strategy still comes out slightly negative on ROI, and I kept that result honest rather than tuning the numbers to look good. The point of the project is that the model ranking and the causal lift can be strong even when the intervention economics need work, which is exactly the conversation a data scientist should be able to have with a stakeholder.

## Tech stack

| Area | Tools |
|------|-------|
| Core | Python, Pandas, NumPy |
| Survival analysis | lifelines, scikit-survival |
| Causal and uplift | EconML, causalml, XGBoost |
| Experiment tracking | MLflow |
| Reporting | Power BI, Matplotlib, Seaborn |
| Data engineering | SQL, PySpark (Databricks-ready) |

## Pipeline execution order

Run the staged scripts in order:

```bash
python notebooks/02_data_quality_checks.py
python notebooks/03_feature_engineering.py
python notebooks/05_survival_modeling.py
python notebooks/06_uplift_modeling.py
python notebooks/07_ltv_estimation.py
python notebooks/08_campaign_optimization.py
python notebooks/09_final_report.py
```

The survival and uplift runs log parameters, metrics, and models to MLflow, so you can compare runs in the MLflow UI.

## Repository structure

```
PRISM/
├── src/             data_simulator, feature_engineering, survival_models,
│                    uplift_models, ltv_estimator, campaign_optimizer
├── notebooks/       Staged pipeline scripts, simulation through final report
├── sql/             Feature engineering written in Spark SQL for productionization
├── powerbi/         Dashboard data model notes
├── visualizations/  Figures
├── docs/            Methodology and result summaries
├── tests/           Unit tests for the simulator, features, and outputs
└── requirements.txt
```

## What I would do next

- Calibrate intervention cost and offer design by risk segment, so the ROI turns positive where it should.
- Personalize treatment by campaign type and user context.
- Add SHAP explanations to the campaign selection so stakeholders can see why a user was targeted.
- Test threshold policies to find the break-even targeting depth.

## A note on the repository

The simulated data files and the MLflow run history are large, so they are not tracked in git. Running the pipeline above regenerates the data, the model outputs, and the Power BI tables from scratch.

## License

Released under the MIT License. See [LICENSE](LICENSE).

Built by Kcey Stadalman.
