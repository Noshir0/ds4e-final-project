---
title: NYC Taxi Fare Linear Regression Lab
emoji: 🚕
colorFrom: yellow
colorTo: red
sdk: streamlit
app_file: streamlit_app.py
pinned: false
license: mit
---

# 🚕 NYC Taxi Fare — Linear Regression Lab

**NYU DS-4-Everyone — Final Project **

A multi-page **Streamlit** app that solves a real business problem with
**linear regression**: estimating the **total cost of a New York City taxi
trip** from its characteristics (distance, duration, time, pickup/drop-off
borough, …) so a ride-hailing / taxi platform can show an **up-front fare
estimate**, power dynamic pricing and flag anomalous charges.

The layout follows the course example *"Linear Regression Lab"*
(<https://mlprojectpy-mkpmppx8frxy7fcxkf5pvv.streamlit.app/>): a sidebar
**Dashboard**, a **Dataset Preview**, a **Description** (describe) table, visual
analysis, prediction, explainability and experiment tracking.

---

## The business problem

> *Given the details of a NYC taxi trip, what will the total cost be, and which
> factors drive it the most?*

We model the continuous target `total` (US dollars) so a taxi app can quote an
up-front fare and detect over- / under-billing.

## The six pages (matches the grading rubric)

| # | Page | What it does |
|---|------|--------------|
| 1 | **Business Case & Data** | Problem statement, **Dataset Preview**, feature dictionary, **Description** (describe) |
| 2 | **Data Visualization** | Distributions, **correlation matrix**, cost-vs-distance, cost by borough/day/payment + insights |
| 3 | **Prediction** | Train & **switch between 2+ models** (Linear, Ridge, Random Forest); metrics + interactive fare estimate |
| 4 | **Explainable AI** | **SHAP** global importance, beeswarm, and linear coefficients |
| 5 | **Hyperparameter Tuning** | Grid sweep with experiment tracking + optional **Weights & Biases** logging; auto-selects the best model |
| 6 | **Conclusion** | Live model comparison, key findings, business recommendations, limitations & next steps |

## Dataset

**New York City taxi trips** — NYC TLC Yellow & Green taxi records (the public
NYC Taxi data, here via seaborn's `taxis` sample). **6,309 trips** after
cleaning, with engineered `hour`, `weekday` and `trip_minutes` columns.

- **Numerical features:** `distance`, `trip_minutes`, `passengers`, `hour`
- **Categorical features:** `weekday`, `payment`, `color`, `pickup_borough`, `dropoff_borough`
- **Target:** `total` (total trip cost in US$)

Fits the project guidelines: ≥8 columns, 500–500k rows, a mix of categorical &
numerical columns, and a continuous target. ⚠️ `fare`, `tip` and `tolls` add up
to `total`, so they are kept for exploration but **excluded from the model** to
avoid data leakage.

## Tech stack

Streamlit · NumPy · Pandas · Matplotlib · Seaborn · Scikit-Learn · SHAP · Weights & Biases

## Project files

```
streamlit_final_project/
├── streamlit_app.py        # all the app logic (5 pages)
├── nyc_taxis.csv           # the dataset (NYC taxi trips, 6,309 rows)
├── banner.png              # header image
├── requirements.txt        # dependencies
├── .streamlit/config.toml  # theme
└── README.md
```

---

## Run locally

```bash
cd streamlit_final_project
pip install -r requirements.txt
streamlit run streamlit_app.py        #  NOT  python streamlit_app.py
```

Then open http://localhost:8501.

## Deploy to Streamlit Community Cloud

1. Push this folder to a **public GitHub repo**.
2. Go to <https://share.streamlit.io> → **New app**.
3. Select the repo, branch, and `streamlit_app.py` as the main file → **Deploy**.

## Deploy to Hugging Face Spaces

1. Create a **Space** at <https://huggingface.co/new-space> → SDK: **Streamlit**.
2. Upload `streamlit_app.py`, `requirements.txt`, `nyc_taxis.csv`, `banner.png`.

## (Optional) Weights & Biases tracking

On page 5, tick *Log this sweep to Weights & Biases* and paste your API key
(from <https://wandb.ai/authorize>). Each configuration is logged as a run.
Without a key, experiments are tracked locally — the app works either way.
