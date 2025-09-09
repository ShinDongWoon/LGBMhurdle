
# g2_hurdle (Global 2-Stage Hurdle Modeling Toolkit)

# g2-hurdle: A High-Performance Two-Stage Hurdle Model for Intermittent Demand Forecasting

[![wSMAPE Score](https://img.shields.io/badge/wSMAPE-0.555008-blue)](https://dacon.io/)

## Overview

**g2-hurdle** is a production-ready, high-performance toolkit for tackling a notoriously difficult machine learning problem: **intermittent demand forecasting**. This is common in retail and food industries, where many items have sporadic sales (i.e., many days with zero sales), a pattern that traditional regression models fail to capture effectively.

This codebase implements a **Two-Stage Hurdle Model** using LightGBM. The architecture deconstructs the prediction problem into two distinct, more manageable sub-problems:

1.  **Classification Stage**: Will an item sell on a given day? (A binary yes/no prediction).
2.  **Regression Stage**: If it sells, what will be the quantity? (A regression on positive-only sales data).

This strategic division of labor allows each model to specialize, leading to a significantly more accurate and robust forecasting system. The architecture is validated using a rigorous time-series cross-validation methodology, achieving a **wSMAPE of 0.5550080421** on the target dataset.

---

## Pipeline Architecture Deep Dive

The end-to-end pipeline is engineered for robustness and performance, from data ingestion to final prediction. It is composed of two primary workflows: `train` and `predict`.

![Pipeline Flow](https://i.imgur.com/8z2k5oX.png)

### 1. Data Ingestion & Preprocessing (`g2_hurdle.utils.io`)

* **Functionality**: The pipeline begins by loading raw training (`train.csv`) and test data (`TEST_*.csv`). It intelligently resolves the data schema by identifying date, target, and series-identifying columns from a list of candidates defined in the `YAML` configuration. Dates are parsed into `datetime` objects, and series identifiers are cast to `category` types for efficiency.
* **Performance Contribution**: A robust and flexible data loading mechanism makes the pipeline adaptable to different datasets without code changes. Sorting the data by time series and date ensures correct sequential processing, which is fundamental for generating time-dependent features like lags and rolling statistics.

### 2. Advanced Feature Engineering (`g2_hurdle.fe`)

* **Functionality**: This is the core of the model's predictive power. A rich feature set is generated to capture the complex temporal dynamics of the sales data.
    * **Calendar Features (`calendar.py`)**: Extracts fundamental time-based signals like year, month, day of the week, week of the year, and flags for month start/end. Cyclical features (e.g., day of the week) are encoded using `sin`/`cos` transformations to help the model understand their periodic nature.
    * **Fourier Features (`fourier.py`)**: Models complex seasonalities (e.g., weekly, yearly) by generating Fourier terms. This is a powerful technique to capture multi-layered cyclical patterns without the high dimensionality of one-hot encoding.
    * **Lag & Rolling Features (`lags_rolling.py`)**: Creates features based on historical sales.
        * **Lags**: The sales quantity from N days ago (e.g., 1, 7, 28 days). This captures autoregressive effects and weekly patterns.
        * **Rolling Aggregates**: Statistical summaries (mean, std, min, max) over various time windows (e.g., 7, 14, 28 days). This captures recent trends and volatility.
    * **Intermittency Features (`intermittency.py`)**: These are custom-designed features crucial for zero-inflated data. It calculates metrics like `days_since_last_sale` and rolling counts of zero-sale days, providing a strong signal to the classification model.
* **Performance Contribution**: Tree-based models like LightGBM thrive on informative features. This comprehensive feature engineering strategy provides the model with a multi-faceted view of the data, allowing it to learn from trends, seasonality, and the specific patterns of intermittent demand, which directly translates to higher accuracy.

### 3. Model Training & Time-Aware Validation (`g2_hurdle.pipeline.train`)

* **Functionality**: The training pipeline is designed to produce a model that generalizes well to future, unseen data.
    * **Time-Series Cross-Validation (`cv/tscv.py`)**: Instead of a standard random split, the code uses a `rolling_forecast_origin_split`. This method creates multiple validation folds, where each fold uses past data to predict a subsequent future period, perfectly mimicking a real-world forecasting scenario and preventing data leakage.
    * **Two-Stage Model Training**:
        1.  **Classifier (`model/classifier.py`)**: An `LGBMClassifier` is trained on the full dataset to predict the probability of a sale occurring (`sales > 0`).
        2.  **Regressor (`model/regressor.py`)**: An `LGBMRegressor` is trained *only on data points where a sale occurred*. This allows the regressor to focus exclusively on learning the distribution of positive sales values without being skewed by the numerous zeros.
    * **Optimal Threshold Search (`model/threshold.py`)**: After cross-validation, the pipeline collects all out-of-fold predictions. It then performs a grid search to find the optimal probability threshold (e.g., 0.45 instead of the default 0.5) that minimizes the overall wSMAPE score. This fine-tunes the classifier's output for the specific business metric.
    * **Final Retraining**: Once the optimal settings are determined, the final classifier and regressor are retrained on the entire available training dataset.
* **Performance Contribution**:
    * The **Hurdle architecture** is the key to handling zero-inflation. By separating the problems, each model becomes more effective.
    * **Rolling-origin CV** provides a highly reliable estimate of the model's true performance and prevents overfitting.
    * **Threshold optimization** is a critical final-mile tuning step that directly optimizes the model's decision-making process against the target metric, squeezing out significant performance gains.

### 4. Recursive Inference (`g2_hurdle.pipeline.predict` & `recursion.py`)

* **Functionality**: Making predictions for a future horizon (e.g., 7 days) requires a specialized strategy.
    * **Recursive Forecasting**: The pipeline predicts one day at a time. The prediction for Day 1 is generated. Then, this predicted value is used to update the dynamic features (like lags and rolling means) needed to make a prediction for Day 2. This process repeats for the entire forecast horizon.
    * **Numba Optimization**: The core feature update logic within the recursive loop is JIT-compiled with `numba` for maximum computational performance, making the inference process fast and efficient even for a large number of time series.
* **Performance Contribution**: The recursive strategy is essential for maintaining feature coherence across the forecast horizon. A naive approach of predicting all 7 days at once would fail because the features for Day 2 to Day 7 would be stale. By dynamically updating features with the latest predictions, the model makes much more informed and accurate multi-step forecasts.

---

## Quick Start

### Local Environment

**1. Install Dependencies**
```bash
python dependency.py
## Quick Start
```bash
python -m g2_hurdle.cli train \
  --train_csv data/train.csv \
  --config g2_hurdle/configs/base.yaml

python -m g2_hurdle.cli predict \
  --test_dir data/test \
  --sample_submission data/sample_submission.csv \
  --out_path outputs/submission.csv
```
All imports are relative; drop this folder as project root and run the commands.

## Quickstart (Colab)

Run the top-level scripts directly in a notebook:

```python
!python dependency.py  # installs required packages
!python train.py       # uses g2_hurdle/configs/korean.yaml and data/train.csv
!python predict.py     # uses g2_hurdle/configs/korean.yaml, data/test, and data/sample_submission.csv
```

By default, both scripts load the configuration from `g2_hurdle/configs/korean.yaml`.
`train.py` reads `data/train.csv` and stores model artifacts in `./artifacts`.
`predict.py` consumes the artifacts, expects test files in `data/test` with a
`data/sample_submission.csv`, and writes predictions to `outputs/submission.csv`.

## GPU configuration

To enable GPU acceleration, set `runtime.use_gpu` to `true` in the YAML
configuration. The pipeline will automatically set `device_type: gpu` for the
LightGBM models. The older `device` parameter is deprecated and should not be
used.

## Dependencies

Run `python dependency.py` to install dependencies.


## Data configuration

Columns used by the toolkit can be provided directly in the `data` section of the
YAML config:

```yaml
data:
  date_col: ds
  target_col: y
  id_cols: [series_id]
```

If these keys are omitted, `resolve_schema` falls back to the corresponding
`*_col_candidates` lists to infer column names.
