# 🚢 Titanic — Passenger Survival Predictor

> End-to-end ML pipeline predicting whether a Titanic passenger survived,
> based on class, demographics, and cabin details.  
> Covers data download → EDA → feature engineering → model comparison → SHAP explainability → FastAPI serving.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-orange)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-red)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourname/titanic-ml/blob/main/colab_pipeline.ipynb)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [1. Data Download](#1-data-download)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Preprocessing & Feature Engineering](#3-preprocessing--feature-engineering)
  - [4. Model Training & MLflow Tracking](#4-model-training--mlflow-tracking)
  - [5. Evaluation](#5-evaluation)
  - [6. SHAP Explainability](#6-shap-explainability)
  - [7. FastAPI Serving](#7-fastapi-serving)
- [Results](#results)
- [MLflow UI](#mlflow-ui)
- [API Reference](#api-reference)
- [Next Steps](#next-steps)
- [Citation](#citation)
- [License](#license)

---

## Problem Statement

Binary classification: given a passenger's ticket class, sex, age, family size, fare, and port of embarkation, predict whether they survived (`survived` / `died`) the Titanic disaster.

The dataset has a mild class imbalance (~62% died, ~38% survived), addressed via `class_weight='balanced'` for sklearn models and `scale_pos_weight` for XGBoost.

---

## Dataset

**Titanic — Machine Learning from Disaster**

| Property | Value |
|---|---|
| Primary source | [Seaborn built-in](https://github.com/mwaskom/seaborn-data) / [Kaggle](https://www.kaggle.com/c/titanic/data) |
| File | `titanic.csv` |
| Rows | 891 |
| Features | 11 raw → 11 engineered |
| Target | `survived` — 0 = died, 1 = survived |
| Class balance | ~62% died / ~38% survived |

**Feature groups:**

| Group | Raw features | Engineered |
|---|---|---|
| Passenger | `pclass`, `sex`, `age`, `embarked` | `title` (extracted from name) |
| Family | `sibsp`, `parch` | `family_size`, `is_alone` |
| Fare | `fare` | `fare_log` (log1p transform) |

> Dropped before modelling: `name`, `ticket`, `cabin`, `boat`, `body`, `home.dest`, `alive` — these either leak the target or have too many unique values to encode usefully.

---

## Project Structure

```
.
├── colab_pipeline.ipynb      # Full pipeline notebook (run in Colab)
├── api_app.py                # FastAPI inference server
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── titanic.csv       # Downloaded dataset (git-ignored)
│   └── processed/            # Reserved for future use
├── models/
│   ├── best_model.pkl        # Serialized best model (git-ignored)
│   └── transformer.pkl       # Fitted ColumnTransformer (git-ignored)
├── reports/
│   ├── evaluation.png        # Confusion matrix, ROC, Precision-Recall
│   ├── shap_importance.png   # SHAP bar chart (top 15 features)
│   └── shap_beeswarm.png     # SHAP beeswarm plot
├── mlruns/                   # MLflow experiment store (git-ignored)
└── README.md
```

---

## Requirements

```
pandas numpy matplotlib seaborn scikit-learn
xgboost lightgbm shap optuna
mlflow fastapi uvicorn pyngrok nest-asyncio
requests joblib
```

```bash
pip install -r requirements.txt
```

Running in **Google Colab**? Cell 1 of the notebook installs everything automatically.

---

## Quickstart

### Option A — Google Colab (recommended)

Click **Open in Colab** above → **Runtime → Run all**.

### Option B — Local Jupyter

```bash
git clone https://github.com/yourname/titanic-ml.git
cd titanic-ml
pip install -r requirements.txt
jupyter notebook colab_pipeline.ipynb
```

### Option C — API only

```bash
# Requires models/best_model.pkl and models/transformer.pkl
uvicorn api_app:app --host 0.0.0.0 --port 8000
```

---

## Pipeline Walkthrough

### 1. Data Download

Three sources, tried in order with automatic fallback:

1. **Seaborn built-in** — `sns.load_dataset('titanic')`, reliable in Colab with no auth
2. **GitHub mirrors** — two public CSV mirrors of the Kaggle training set
3. **Manual upload fallback** — `RuntimeError` with the Kaggle download link if all fail

Column names are normalised to lowercase after loading. The `survived` column is cast to binary int regardless of whether the source returns strings, booleans, or integers.

---

### 2. Exploratory Data Analysis

Three plots generated automatically:

- **Class balance** — bar chart + pie chart (62% / 38% split)
- **Numeric distributions by survival** — 2×2 grid for `age`, `fare`, `sibsp`, `parch`
- **Survival rate by category** — bar charts for `sex`, `pclass`, `embarked`

---

### 3. Preprocessing & Feature Engineering

**Derived features created before the pipeline:**

| Feature | Formula | Rationale |
|---|---|---|
| `family_size` | `sibsp + parch + 1` | Lone passengers vs large families behaved differently |
| `is_alone` | `1 if family_size == 1` | Binary flag — alone vs not |
| `fare_log` | `log1p(fare)` | Compresses the right-skewed fare distribution |
| `title` | Regex on `name` | Encodes social status (Mr/Mrs/Miss/Master/Officer/Royalty/Sir) |

**ColumnTransformer:**

| Type | Columns | Steps |
|---|---|---|
| Numeric (8) | `age`, `fare`, `fare_log`, `sibsp`, `parch`, `family_size`, `is_alone`, `pclass` | Median imputation → `StandardScaler` |
| Nominal (3) | `sex`, `embarked`, `title` | Mode imputation → `OneHotEncoder(handle_unknown='ignore')` |

- Train / test split: **80 / 20**, stratified on `survived`, `random_state=42`
- Fitted transformer saved to `models/transformer.pkl`

---

### 4. Model Training & MLflow Tracking

Three models logged to the **`titanic-survival`** MLflow experiment:

| Run name | Model | Key settings |
|---|---|---|
| `logistic_regression` | `LogisticRegression` | `C=1.0`, `max_iter=500`, `class_weight='balanced'` |
| `random_forest` | `RandomForestClassifier` | 300 trees, `max_depth=8`, `class_weight='balanced'` |
| `xgboost_optuna` | `XGBClassifier` | 20-trial Optuna TPE search (`seed=42`), `scale_pos_weight` auto-set |

Each run logs: all hyperparameters, four metrics (AUC-ROC, F1, precision, recall), and the full model artifact via `mlflow.sklearn.log_model` with an inferred input signature.

The best model by ROC-AUC is saved to `models/best_model.pkl`.

---

### 5. Evaluation

Three plots saved to `reports/evaluation.png`:

- **Confusion matrix** — with `Died` / `Survived` labels
- **ROC curve** — with AUC score
- **Precision-Recall curve**

---

### 6. SHAP Explainability

`shap.TreeExplainer` on up to 500 test samples. Two plots:

- **`reports/shap_importance.png`** — mean absolute SHAP, top 15 features  
  *(expect `sex_female`, `fare_log`, `pclass`, `title_Mr` to dominate)*
- **`reports/shap_beeswarm.png`** — direction and magnitude per prediction

---

### 7. FastAPI Serving

`api_app.py` loads both pickled artefacts at startup and exposes two endpoints. The server starts in a daemon thread on port 8000.

A two-passenger smoke test fires automatically — **Rose** (1st class, female) and **Jack** (3rd class, male) — to sanity-check the predictions.

---

## Results

Reference run — Google Colab, Python 3.10, XGBoost 2.x, `TPESampler(seed=42)`

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| XGBoost + Optuna | **~0.88** | ~0.80 | ~0.79 | ~0.81 |
| Random Forest | ~0.87 | ~0.79 | ~0.78 | ~0.80 |
| Logistic Regression | ~0.84 | ~0.76 | ~0.75 | ~0.77 |

---

## MLflow UI

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

To expose from Colab, uncomment the **ngrok cell** (cell 8) and add your token from [dashboard.ngrok.com](https://dashboard.ngrok.com).

---

## API Reference

### `POST /predict`

**Request body:**

```json
{
  "pclass":   3,
  "sex":      "male",
  "age":      20,
  "sibsp":    0,
  "parch":    0,
  "fare":     7.25,
  "embarked": "S",
  "title":    "Mr"
}
```

**Response:**

```json
{
  "prediction":  "died",
  "probability": 0.1342
}
```

**Field notes:**

| Field | Type | Values | Default |
|---|---|---|---|
| `pclass` | int | 1, 2, 3 | required |
| `sex` | str | `male`, `female` | required |
| `age` | float | passenger age | `29.0` (median) |
| `embarked` | str | `S`, `C`, `Q` | `S` |
| `title` | str | `Mr`, `Mrs`, `Miss`, `Master`, `Officer`, `Royalty`, `Sir` | inferred from `sex` |

`age` and `embarked` are optional — missing values are filled with the training median/mode by the transformer.

---

## Next Steps

- **Streamlit dashboard** — interactive passenger form with SHAP force plot per prediction
- **Deploy API** — Railway, Render, or Hugging Face Spaces (FastAPI + free tier)
- **Cross-validation** — replace single split with stratified 5-fold for tighter metric bounds
- **Threshold tuning** — shift the 0.5 decision threshold; `recall` for survivors may matter more than `precision` in a rescue context
- **LightGBM** — already installed, straightforward fourth MLflow run

---

## Citation

```bibtex
@misc{titanic2012,
  title  = {Titanic: Machine Learning from Disaster},
  author = {Kaggle},
  year   = {2012},
  url    = {https://www.kaggle.com/c/titanic}
}
```

---

## License

MIT ©
