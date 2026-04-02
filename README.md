# 🚀 TrendPredictor AI: The Executive Operations & Marketing Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-LightGBM_%7C_XGBoost_%7C_RF_%7C_LR-FF6F00?style=for-the-badge)
![Optimization](https://img.shields.io/badge/Optimization-Optuna_Bayesian-brightgreen?style=for-the-badge)
![Data](https://img.shields.io/badge/Data-Pandas_%7C_NumPy-150458?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)

---

## 📊 What This Does

Most business forecasting models rely strictly on historical sales — they only look backwards to predict what is ahead.

**TrendPredictor AI** is an automated, end-to-end Machine Learning pipeline built on a **Triangulated Signal Architecture**. It ingests and cleans messy data from three separate sources, combines them into a single feature matrix, trains four competing AI models, and promotes the winner to generate next-week demand forecasts and prescribe exact business actions for every product.

**The three data signals:**
1. **Internal Momentum** — Your historical business sales data (weekly units sold, pricing)
2. **Consumer Intent** — Google Trends search volume (real demand signal from the market)
3. **Commercial Confidence** — Meta/Facebook Ads spend (what competitors are actually paying to promote)

By fusing predictive machine learning with financial constraints, the pipeline acts as an automated Data Scientist — translating raw CSVs into C-suite-ready intelligence with zero manual analysis.

---

## 🔒 Portfolio Edition Notice

This repository showcases the architecture, output artifacts, and comprehensive documentation of a proprietary Machine Learning pipeline. To protect intellectual property, the core predictive engine is represented as a structural skeleton. I am fully available for live, line-by-line technical walkthroughs of the complete codebase during interviews.

---

## ⚙️ Technical Architecture

### 1. Dynamic Model Zoo with Bayesian Hyperparameter Tuning

The pipeline does not rely on a single hardcoded algorithm. It runs a live competition between four models on every execution:

| Model | Library | Fallback if not installed |
|---|---|---|
| Linear Regression | scikit-learn | — (always available) |
| Random Forest | scikit-learn | — (always available) |
| Gradient Boosting / **LightGBM** | lightgbm | `GradientBoostingRegressor` |
| Extra Trees / **XGBoost** | xgboost | `ExtraTreesRegressor` |

Each tree-based model is tuned with **Optuna (TPE Bayesian sampling)** before competing. The model with the lowest validation MAE is automatically promoted to generate all final predictions.

### 2. Strict Time-Series Data Integrity — Zero Leakage

The pipeline enforces two separate strategies to prevent the model from learning from the future:

- **Dynamic shuffle flag:** `do_shuffle = not USE_REAL_DATA` — real data preserves strict chronological order (`shuffle=False`) to prevent future-training-past leakage. Synthetic cross-sectional data uses `shuffle=True` because its 2,000 products have no chronological dependency.
- **TimeSeriesSplit CV during Optuna:** When tuning on real data, cross-validation uses `TimeSeriesSplit` rather than `KFold`, so hyperparameters are never chosen based on leaky shuffled folds.

### 3. Graceful Degradation — Built to Never Crash

| Scenario | Behaviour |
|---|---|
| LightGBM not installed | Silently falls back to `GradientBoostingRegressor`, results labelled as LightGBM |
| XGBoost not installed | Silently falls back to `ExtraTreesRegressor`, results labelled as XGBoost |
| Optuna not installed | Falls back to randomised grid search with 20 trials |
| SHAP not installed | Falls back to permutation importance for feature attribution |
| `ad_spend_usd` column missing from Facebook export | Defaults entire column to `$0.0` with a printed warning — pipeline continues |
| `xlsxwriter` not installed | Falls back to `openpyxl` for Excel generation — no output is lost |
| Negative sales predicted by linear models | `np.clip(preds, 0, None)` enforces non-negativity — no impossible revenue totals |
| Product with $0 ad spend | `np.divide` with zero-masking forces `ROI = 0%` rather than `infinity` |

### 4. The 5-Tier ROI-Aware Action Engine

Every product receives a precise business recommendation based on two combined signals — predicted sales velocity AND ad profitability:

| Action | Condition |
|---|---|
| 🔥 REORDER & SCALE ADS | Sales growing >20% AND ROI > 150% |
| 📦 REORDER FAST | Sales growing >20% (but ad ROI is tight) |
| ⚠️ RUN PROMO | Sales declining >20% BUT ROI > 100% (worth saving) |
| 🛑 KILL ADS & LIQUIDATE | Sales declining >20% AND ROI < 50% |
| ✅ MAINTAIN | All other cases — stable, no urgent action |

---

## 💼 Business Intelligence Outputs

The pipeline produces two professionally formatted Excel workbooks and 16 automated charts on every run.

### 📦 Product_Catalog_Detailed.xlsx — For Operations & Supply Chain

A **51-column operational database** with two sheets:
- `Product_Catalog` — AI forecasts alongside auto-generated SKUs, supplier mapping, stock levels, reorder points, warehouse locations, dimensional weights, conversion rates, customer LTV, competitor pricing, and market share estimates
- `Catalog_Summary` — 16-row KPI dashboard: total stock value, forecasted revenue/profit, avg ratings, products needing reorder

### 📈 Marketing_Intelligence_Report.xlsx — For C-Suite & Marketing

A **10-tab executive dashboard:**

| Tab | Contents |
|---|---|
| Executive Summary | One-page KPI dashboard: revenue forecast, demand split, action counts, model accuracy |
| All_Products_Predictions | Full 22-column predictions table for every product |
| Top_30_Products | Ranked top performers by predicted sales |
| Urgent_Actions | Every product needing immediate reorder or promotion |
| Category_Performance | Revenue, profit, and ROI breakdown by category |
| Segment_Analysis | Budget / Mid-Range / Premium tier breakdown |
| Campaign_Ideas | Auto-generated campaign brief per category with bundle recommendations |
| Weekly_Patterns | Day-of-week revenue and sales distribution |
| Model_Performance | Train / Validation / Test metrics: MAE, RMSE, R², MAPE, Bias |
| Model_Comparison | All four model scores side by side with winner flagged |

### 📊 16 Automated Charts

10 model diagnostic charts (prediction vs actual, residuals, error distribution, learning curve, SHAP/permutation importance, Q-Q plot) and 6 marketing intelligence charts (top products, revenue by category, demand tiers, action plan distribution, price vs sales, weekly trends).

When running inside **Kaggle or Jupyter**, all 16 charts render inline below the code cell automatically — no file download needed.

*(See `sample_outputs_AI/` and `sample_outputs_marketing/` folders above for visual examples.)*

---

## 🔬 Dual-Audience Documentation

Writing code is only half the job — driving adoption is the other. This repository includes professionally authored documentation that demonstrates the ability to communicate complex data science to any audience.

**📖 User Manual** — A step-by-step guide for non-technical operations and marketing teams. Covers how to export data from Google Trends, Facebook Ads Manager, and your sales platform, how to configure the pipeline, and how to interpret every business recommendation — with no Python knowledge required. Includes a full Kaggle walkthrough for users who cannot install Python locally.

**🔬 Code Explanation Guide** — A comprehensive architectural breakdown for technical stakeholders and engineering auditors. Every function is explained twice: strict mathematical and engineering terminology for engineers, and plain-English business logic for non-technical stakeholders.

> 📬 **Available on request** — The full Code Explanation Guide is not published here to protect intellectual property. If you are a hiring manager or technical reviewer, reach out directly and I will send it within 24 hours: [lucascamargo@outlook.com.au](mailto:lucascamargo@outlook.com.au)

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy openpyxl xlsxwriter lightgbm xgboost optuna shap

# Run on synthetic data (no CSV files needed — good for testing)
python pipeline_architecture_skeleton.py
```

> 💡 The full working script is available for live review during interviews. The skeleton file in this repo demonstrates architecture and documentation standards.

**Required CSV files when using real data:**

| File | Required Columns |
|---|---|
| `google_trends.csv` | `product_name`, `trend_score`, `week_date` |
| `facebook_ads.csv` | `product_name`, `ad_count`, `ad_spend_usd`, `week_date` |
| `business_sales.csv` | `product_name`, `price`, `past_week_sales`, `past_week_2_sales`, `past_week_3_sales`, `future_week_sales`, `week_date` |

---

## 📂 Repository Structure

```
TrendPredictor-AI/
│
├── README.md                          ← You are here
├── pipeline_architecture_skeleton.py  ← Full architecture with IP removed
├── User_Manual.pdf                    ← Step-by-step guide for non-technical teams
│
├── sample_outputs_AI/                 ← Model performance & diagnostic charts
└── sample_outputs_marketing/          ← Marketing intelligence & business charts
```

> 🔒 **IP Protection Notice:** The working `pipeline_improved.py` is not included in this repository. The skeleton file contains every function signature, full docstring, and architectural comment — but the proprietary mathematics, ETL cleaning logic, and Bayesian tuning algorithms have been removed. I am available for a live, line-by-line technical walkthrough during any interview or review.

---

## 🤝 Let's Connect

I built this pipeline to solve the real-world disconnect between raw marketing data and inventory ROI. I am looking for an AI or Machine Learning role where I can build resilient, automated systems that directly drive business revenue.
If you are looking for someone who applies ML mathematics, writes documented code, and communicates results clearly to both technical and non-technical teams — let's talk.

- 📧 **Email:** [lucascamargo@outlook.com.au](mailto:lucascamargo@outlook.com.au)
- 🔗 **LinkedIn:** [linkedin.com/in/lucasgcamargo](https://www.linkedin.com/in/lucasgcamargo/)

---

## © Copyright

© 2026 Lucas Camargo. All Rights Reserved.

This code, architecture, documentation, and all associated intellectual property are proprietary and confidential. No part of this repository may be reproduced, distributed, or transmitted in any form or by any means — including copying, recording, or any information storage and retrieval system — without the prior written permission of the author.
