# LGBM Hurdle Model for Time Series Forecasting

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/shindongwoon/lgbmhurdle)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![wSMAPE Score](https://img.shields.io/badge/wSMAPE-0.555008-orange)](.)


## 📜 Project Overview

This project is engineered to tackle the complex challenge of **forecasting time series data characterized by intermittent demand**. Intermittent demand, where sales data is dominated by frequent zero values, poses a significant hurdle for traditional forecasting models. Standard algorithms often struggle to accurately predict sporadic, non-zero events.

To address this, we employ a sophisticated **Hurdle Model** architecture. This approach strategically decomposes the forecasting problem into two distinct, more manageable sub-problems:
1.  A **Binary Classification** task to predict *whether* a sale will occur (i.e., hurdle the zero-demand barrier).
2.  A **Regression** task to predict *how much* will be sold, conditioned on a sale actually happening.

Both models are implemented using **LightGBM**, a high-performance gradient boosting framework renowned for its speed and accuracy, ensuring a robust and efficient solution.

---

## 🚀 Key Features

-   **Hurdle Model Architecture**: By decoupling the prediction of sale occurrence from sale quantity, the model architecture is specifically tailored to handle zero-inflated datasets, maximizing predictive accuracy for intermittent demand patterns.
-   **High-Performance LightGBM Engine**: Utilizes LightGBM as the backbone for both classification and regression tasks, leveraging its exceptional speed and state-of-the-art performance in gradient boosting.
-   **Advanced Feature Engineering**: Implements a comprehensive suite of engineered features to capture complex temporal dynamics:
    -   **Calendar-based Features**: Extracts signals from timestamps (e.g., day of week, week of year, month, quarter).
    -   **Lag and Rolling Window Statistics**: Incorporates historical trends and patterns using lagged values and rolling statistical aggregations (mean, std, min, max).
    -   **Fourier Transform Features**: Precisely models complex seasonality (e.g., weekly, yearly cycles) by transforming temporal features into the frequency domain.
-   **Recursive Forecasting Pipeline**: Employs a recursive, multi-step forecasting strategy. For long prediction horizons, the model uses its own previous predictions as inputs to generate features for subsequent steps, ensuring stable and accurate long-term forecasts.
-   **Efficient Caching System**: A built-in caching mechanism serializes the results of computationally expensive feature engineering steps. This dramatically reduces runtime during iterative experiments and model tuning, accelerating the development cycle.
-   **Configuration-Driven Workflow**: The entire pipeline is managed via `YAML` configuration files. This approach decouples logic from parameters, ensuring experimental **reproducibility** and simplifying hyperparameter tuning and feature selection.

---

## 📊 Data Schema Definition

For the model to operate correctly, the input and output data must adhere to the following specifications.

### 1. Input Data Schema

All source data for training (`train.csv`) and inference (`TEST_*.csv`) must be structured in a **long format**. Each row should represent a single observation for a specific item on a specific date.

**Required Features:**

| Column Name       | Data Type           | Description                                                                                             |
| :---------------- | :------------------ | :------------------------------------------------------------------------------------------------------ |
| `Date`            | `Date` or `String`  | The date of the sales record (e.g., `YYYY-MM-DD`).                                                      |
| `Series_Identifier` | `String`            | A unique identifier for each distinct time series (e.g., a combination of store and item ID).         |
| `Sales_Quantity`  | `Numeric` (Integer) | The actual quantity sold on that date. This serves as the **target variable** for the model to predict. |

### 2. Output Data Schema

The final prediction output (`sample_submission.csv`) must be pivoted into a **wide-format** structure. In this format, each time series (`Series_Identifier`) becomes a separate column.

**Output Table Structure:**

-   **Index**: The rows are indexed by `Date`.
-   **Column Headers**: The first column is `Date`, followed by columns for every unique `Series_Identifier`.
-   **Values**: Each cell contains the **predicted `Sales_Quantity`** for the corresponding `Date` (row) and `Series_Identifier` (column).

**Example Structure:**

| Date       | StoreA_ItemX | StoreA_ItemY | StoreB_ItemZ | ... |
| :--------- | :----------- | :----------- | :----------- | :-- |
| 2025-09-16 | 15           | 0            | 21           | ... |
| 2025-09-17 | 12           | 3            | 18           | ... |
| ...        | ...          | ...          | ...          | ... |

This standard format facilitates easy analysis and submission of time series forecasting results.

---

## ⚙️ Code Pipeline Analysis

The codebase is modularly structured into three core stages: **Data Preprocessing & Feature Engineering**, **Model Training**, and **Recursive Inference**.

### 1. End-to-End Pipeline Flow

1.  **Configuration Loading (`configs/`)**: The pipeline initializes by loading hyperparameters, file paths, and feature definitions from `YAML` configuration files (e.g., `base.yaml`, `korean.yaml`).
2.  **Data Ingestion & Preprocessing (`fe/preprocess.py`)**: Raw data is loaded, and initial cleaning is performed, including data type casting and handling of missing values.
3.  **Feature Engineering (`fe/`)**: A rich set of features is constructed from the preprocessed data.
    -   `calendar.py`: Generates calendar-based features.
    -   `lags_rolling.py`: Creates lag and rolling window statistical features.
    -   `fourier.py`: Produces Fourier term features for seasonality modeling.
4.  **Model Training (`pipeline/train.py`)**:
    -   **Data Splitting**: Utilizes Time Series Cross-Validation (`cv/tscv.py`) to create robust training and validation splits that respect the temporal order of the data.
    -   **Classifier Training (`model/classifier.py`)**: A LightGBM classifier is trained on a binary target (1 if sales > 0, else 0).
    -   **Regressor Training (`model/regressor.py`)**: A LightGBM regressor is trained exclusively on data where sales > 0 to predict the actual sales quantity.
5.  **Model Serialization**: The trained classifier and regressor model artifacts are saved to disk.
6.  **Inference (`pipeline/predict.py` & `recursion.py`)**:
    -   The pipeline iterates one step at a time over the prediction horizon.
    -   **Classification Prediction**: The classifier predicts the probability of a sale occurring on the next day.
    -   **Hurdle Application**: If the predicted probability exceeds a predefined threshold, the regressor predicts the sales quantity. Otherwise, the prediction is set to 0.
    -   **Recursive Update**: The new prediction is used to update the lag and rolling features for the subsequent day's forecast. This loop continues until the entire prediction horizon is covered.
7.  **Submission Generation**: The final predictions are formatted to match the required `sample_submission.csv` schema.

### 2. Module Contributions to Performance

| Module Path                       | Core Function                   | Contribution to Performance                                                                                                                                                             |
| :-------------------------------- | :------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`g2_hurdle/fe/`** | **Feature Engineering** | Fundamentally improves model accuracy by translating complex temporal patterns (trends, seasonality, autocorrelation) into a format the model can learn. Lag and rolling features are critical. |
| **`g2_hurdle/utils/cache.py`** | **Feature Caching** | Drastically reduces experiment turnaround time by caching the results of expensive feature computations. This accelerates development and hyperparameter tuning by avoiding redundant work.     |
| **`g2_hurdle/model/`** | **Hurdle Model Implementation** | Directly addresses the intermittent demand problem by splitting it into classification and regression. This prevents the regression model from being biased by zero-inflation, significantly improving the final wSMAPE score. |
| **`g2_hurdle/cv/tscv.py`** | **Time Series Cross-Validation**| Ensures a robust and reliable evaluation of the model's generalization performance by preventing data leakage (i.e., training on future data).                                          |
| **`g2_hurdle/pipeline/recursion.py`** | **Recursive Forecasting Logic** | Enables accurate multi-step forecasting by dynamically updating features at each step with the latest available information (i.e., previous predictions). This yields more stable and precise long-range predictions. |
| **`g2_hurdle/configs/`** | **Configuration Management** | Guarantees experimental reproducibility and flexibility by externalizing all hyperparameters and settings. This allows for rapid, code-free iteration on different model configurations.  |

---

## 📊 Results

Rigorous evaluation of this codebase demonstrated a highly competitive performance, achieving a **wSMAPE score of 0.5550080421**. This result validates the effectiveness of the Hurdle Model architecture combined with sophisticated feature engineering for solving intermittent demand forecasting problems.

-   **wSMAPE (Weighted Symmetric Mean Absolute Percentage Error)**: An industry-standard metric that weights errors based on the actual demand volume, preventing high-percentage errors on low-volume items from disproportionately penalizing the model's score.

---

## 🚀 Getting Started on Colab

### 1. Prerequisites

-   Python 3.8+
-   Nvidia GPU (Recommended for training)

### 2. Quickstart (Colab)

```bash
# 1. Clone the repository
!git clone [https://github.com/shindongwoon/lgbmhurdle.git](https://github.com/shindongwoon/lgbmhurdle.git)
%cd lgbmhurdle

# 2. Install dependencies
!python dependency.py

# 3. Run model training
!python train.py

# 4. Run prediction
!python predict.py
## 📜 프로젝트 개요 (Project Overview)

본 프로젝트는 **간헐적 수요(Intermittent Demand)** 특성을 가진 시계열 데이터의 미래 판매량을 예측하는 것을 목표로 합니다. 간헐적 수요란 '0' 값이 빈번하게 나타나는 데이터를 의미하며, 일반적인 시계열 예측 모델로는 정확한 예측이 어렵습니다.

이러한 문제를 해결하기 위해 본 프로젝트에서는 **허들 모델(Hurdle Model)** 접근법을 채택했습니다. 허들 모델은 '판매가 발생할지 여부'를 예측하는 **이진 분류(Binary Classification)** 문제와 '판매가 발생했을 때 얼마나 팔릴지'를 예측하는 **회귀(Regression)** 문제로 나누어 접근합니다. 두 모델 모두 강력한 성능을 자랑하는 **LightGBM**을 기반으로 구현되었습니다.

---

## 🚀 주요 특징 (Key Features)

- **허들 모델 아키텍처 (Hurdle Model Architecture)**: 판매 발생 여부와 판매량을 분리하여 예측함으로써 '0'이 많은 데이터에 대한 예측 성능을 극대화합니다.
- **고성능 LightGBM 활용 (High-Performance LightGBM)**: 빠른 학습 속도와 높은 예측 정확도를 자랑하는 LightGBM을 분류 및 회귀 모델의 백본으로 사용합니다.
- **고급 피처 엔지니어링 (Advanced Feature Engineering)**:
    - **시간 기반 피처**: 날짜, 요일, 월, 주차 등 시간 정보를 활용한 피처
    - **지연 및 롤링 통계 피처**: 과거 데이터의 추세와 패턴을 학습하기 위한 Lag 및 Rolling window 피처
    - **푸리에 변환 피처**: 계절성(Seasonality)을 정교하게 모델링하기 위한 Fourier-transform 피처
- **재귀적 예측 파이프라인 (Recursive Prediction Pipeline)**: 예측 대상 기간이 길어질 때, 이전 예측값을 다시 피처로 사용하여 다음 시점의 값을 예측하는 재귀적 구조를 채택하여 안정적인 장기 예측을 수행합니다.
- **효율적인 캐싱 시스템 (Efficient Caching System)**: 피처 엔지니어링 과정에서 생성된 데이터를 캐싱하여 반복적인 학습 및 실험 시의 연산 비용을 획기적으로 줄입니다.
- **설정 파일 기반 관리 (Configuration File Management)**: `YAML` 형식의 설정 파일을 통해 모델 하이퍼파라미터, 피처 목록 등을 체계적으로 관리하여 실험의 재현성을 보장합니다.

---
<img width="512" height="512" alt="Gemini_Generated_Image_2m0zym2m0zym2m0z" src="https://github.com/user-attachments/assets/8bec8402-4d52-429c-873b-f8426e368281" />


<img width="512" height="512" alt="Gemini_Generated_Image_2m0zyn2m0zyn2m0z" src="https://github.com/user-attachments/assets/680ad867-937b-4b2f-9418-93eca84c9a7b" />

<img width="512" height="512" alt="Gemini_Generated_Image_2m0zyo2m0zyo2m0z" src="https://github.com/user-attachments/assets/d05af1d9-7bcb-455d-8465-ef54b263a38d" />

---
## 📊 데이터 스키마 정의 (Data Schema Definition)

본 예측 모델이 정상적으로 동작하기 위해서는 입력 데이터와 출력 데이터가 다음 명세에 따라 정확하게 구성되어야 합니다.

### 1. 입력 데이터 스키마 (Input Data Schema)

학습(`train.csv`) 및 평가(`TEST_*.csv`)에 사용되는 모든 원본 데이터는 **Long-Format** 데이터 구조를 따라야 합니다. 각 행은 특정 영업일의 특정 상품에 대한 판매 기록을 나타내는 개별 관측치(Observation)입니다.

**필수 피처(Features):**

| 컬럼명 (Column Name) | 데이터 타입 (Data Type) | 설명 (Description)                                                               |
| :------------------- | :---------------------- | :------------------------------------------------------------------------------- |
| `영업일자`           | `Date` or `String`      | 매출이 발생한 날짜입니다. (예: `YYYY-MM-DD`)                                       |
| `영업장명_메뉴명`      | `String`                | 각 상품(Item)을 고유하게 식별하는 **식별자(Identifier)**입니다.                        |
| `매출수량`           | `Numeric` (Integer)     | 해당 영업일에 발생한 상품의 실제 판매량으로, 모델이 예측하고자 하는 **타겟 변수(Target Variable)**입니다. |

### 2. 출력 데이터 스키마 (Output Data Schema)

최종 예측 결과물인 `sample_submission.csv` 파일은 입력 데이터와 달리 **Wide-Format** 데이터 구조로 피봇(Pivot)된 형태여야 합니다. 이는 각 상품(`영업장명_메뉴명`)이 개별적인 컬럼이 되는 매트릭스(Matrix) 구조를 의미합니다.

**출력 테이블 구조:**

-   **인덱스 (Index):** 테이블의 행(Row)은 `영업일자`가 기준이 됩니다.
-   **컬럼 헤더 (Column Headers):** 테이블의 첫 번째 열은 `영업일자`이며, 두 번째 열부터는 `영업장명_메뉴명`의 모든 고유값(Unique Values)이 컬럼명으로 위치합니다.
-   **값 (Values):** 테이블의 각 셀(Cell)에는 해당 `영업일자`(행)와 `영업장명_메뉴명`(열)에 해당하는 **예측된 `매출수량`**이 기입됩니다.

**구조 예시:**

| 영업일자   | StoreA_ItemX | StoreA_ItemY | StoreB_ItemZ | ... |
| :--------- | :----------- | :----------- | :----------- | :-- |
| 2025-09-16 | 15           | 0            | 21           | ... |
| 2025-09-17 | 12           | 3            | 18           | ... |
| ...        | ...          | ...          | ...          | ... |

이러한 구조는 각 상품의 일별 예측 판매량을 한눈에 파악하기 용이한 표준적인 시계열 예측 결과 제출 형식입니다.

---

## ⚙️ 코드 파이프라인 분석 (Code Pipeline Analysis)

본 코드베이스는 크게 **데이터 전처리 및 피처 엔지니어링**, **모델 학습**, **재귀적 예측**의 세 단계로 구성됩니다.

### 1. 전체 코드 파이프라인 흐름

1.  **초기 설정 (`configs/`)**: `base.yaml`, `korean.yaml` 등의 설정 파일을 로드하여 프로젝트 전반의 하이퍼파라미터, 데이터 경로, 사용할 피처 목록 등을 정의합니다.
2.  **데이터 로딩 및 전처리 (`fe/preprocess.py`)**: 원본 데이터를 로드하고, 기본적인 데이터 타입 변환 및 결측치 처리 등의 전처리 과정을 수행합니다.
3.  **피처 엔지니어링 (`fe/`)**: 전처리된 데이터를 기반으로 모델 학습에 사용될 다양한 피처를 생성합니다.
    - `calendar.py`: 시간 관련 피처 생성
    - `lags_rolling.py`: Lag 및 Rolling 통계 피처 생성
    - `fourier.py`: 계절성 피처 생성
4.  **모델 학습 (`pipeline/train.py`)**:
    - **데이터 분할**: 시계열 데이터의 특성을 고려하여 Time Series Cross-Validation 방식으로 학습/검증 데이터를 분할합니다 (`cv/tscv.py`).
    - **분류 모델 학습 (`model/classifier.py`)**: 타겟 값이 0인지 아닌지를 레이블로 하여 LightGBM 분류기를 학습시킵니다.
    - **회귀 모델 학습 (`model/regressor.py`)**: 타겟 값이 0이 아닌 데이터만을 대상으로, 실제 판매량을 예측하는 LightGBM 회귀 모델을 학습시킵니다.
5.  **모델 저장**: 학습된 분류기와 회귀기 모델 객체를 지정된 경로에 저장합니다.
6.  **예측 (`pipeline/predict.py` & `recursion.py`)**:
    - 테스트 데이터에 대해 하루(one-step)씩 예측을 진행합니다.
    - **분류기 예측**: 내일 판매가 발생할 확률을 예측합니다.
    - **허들 적용**: 예측된 확률이 사전에 정의된 임계값(Threshold)을 넘으면 회귀 모델을 통해 판매량을 예측하고, 넘지 않으면 0으로 예측합니다.
    - **재귀 업데이트**: 예측된 값을 기반으로 다음 날 예측에 필요한 Lag, Rolling 피처를 업데이트하고 이 과정을 예측 기간이 끝날 때까지 반복합니다.
7.  **결과 제출**: 최종 예측 결과를 `sample_submission.csv` 형식에 맞추어 생성합니다.

### 2. 각 부분의 기능 및 성능 향상 기여

| 모듈 경로 (Module Path) | 핵심 기능 (Core Function) | 성능 향상 기여 방식 (Contribution to Performance) |
| :--- | :--- | :--- |
| **`g2_hurdle/fe/`** | **피처 엔지니어링** | 시계열 데이터의 복잡한 패턴(추세, 계절성, 자기상관성)을 모델이 학습할 수 있는 형태로 변환하여 **예측 정확도를 근본적으로 향상**시킵니다. 특히 Lag, Rolling 피처는 시계열 예측의 핵심입니다. |
| **`g2_hurdle/utils/cache.py`** | **피처 캐싱** | 대용량 데이터에 대한 피처 엔지니어링은 많은 시간이 소요됩니다. 생성된 피처를 파일로 저장하고 재사용함으로써, 반복 실험 시 **전체 실행 시간을 극적으로 단축**시켜 개발 효율성을 높입니다. |
| **`g2_hurdle/model/`** | **허들 모델 구현** | **Classifier**와 **Regressor**로 역할을 분담하여 간헐적 수요 문제에 특화된 접근을 합니다. 이는 단일 회귀 모델이 '0' 값에 의해 학습이 왜곡되는 것을 방지하고, 두 가지 문제를 각각 최적화하여 **wSMAPE 점수를 크게 개선**합니다. |
| **`g2_hurdle/cv/tscv.py`** | **시계열 교차 검증** | 미래의 데이터가 과거 데이터의 학습에 사용되는 것을 방지(Data Leakage 방지)합니다. 이를 통해 모델의 일반화 성능을 보다 **정확하고 신뢰성 있게 평가**할 수 있습니다. |
| **`g2_hurdle/pipeline/recursion.py`** | **재귀적 예측 로직** | 다중 시점 예측(Multi-step Forecasting) 시, 매 시점마다 최신 정보를 반영한 피처를 생성하여 예측을 수행합니다. 이는 단순히 모델 하나로 전체 기간을 예측하는 것보다 훨씬 **정교하고 안정적인 장기 예측**을 가능하게 합니다. |
| **`g2_hurdle/configs/`** | **설정 관리** | 모든 하이퍼파라미터와 설정을 코드가 아닌 외부 파일로 분리하여 관리합니다. 이를 통해 **실험의 재현성을 확보**하고, 코드 수정 없이 다양한 조건으로 손쉽게 실험을 진행할 수 있습니다. |

---

## 📊 실험 결과 (Results)

본 코드베이스를 사용하여 객관적으로 성능을 검증한 결과, **wSMAPE 점수 0.5550080421**를 달성했습니다. 이는 허들 모델과 정교한 피처 엔지니어링이 간헐적 수요 예측 문제에 매우 효과적임을 입증하는 결과입니다.

- **wSMAPE (Weighted Symmetric Mean Absolute Percentage Error)**: 수요량의 크기에 따라 가중치를 부여하는 평가지표로, 수요가 적은 항목의 오차에 과도한 페널티를 주는 것을 방지합니다.

---

## 🚀 Colab 에서 시작하기 (Getting Started)

### 1. 요구사항 (Prerequisites)

- Python 3.8+
- Nvidia GPU

### 2. Quickstart (Colab)

```bash
# 1. 저장소 클론
!git clone [https://github.com/shindongwoon/lgbmhurdle.git](https://github.com/shindongwoon/lgbmhurdle.git)
cd lgbmhurdle

# 2. 의존성 설치
!python dependency.py

# 3. 모델 훈련 진행
!python train.py

# 4. 모델 예측 진행
!python predict.py
```
All imports are relative; drop this folder as project root and run the commands.

By default, both scripts load the configuration from `g2_hurdle/configs/korean.yaml`.
`train.py` reads `data/train.csv` and stores model artifacts in `./artifacts`.
`predict.py` consumes the artifacts, expects test files in `data/test` with a
`data/sample_submission.csv`, and writes predictions to `outputs/submission.csv`.

## GPU configuration

To enable GPU acceleration, set `runtime.use_gpu` to `true` in the YAML
configuration. The pipeline will automatically set `device_type: gpu` for the
LightGBM models. The older `device` parameter is deprecated and should not be
used.

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

To clip negative values before feature engineering, list the columns under
`non_negative_cols`:

```yaml
data:
  non_negative_cols: [sales]
```

Any negative values in these columns will be replaced with zero during both
training and prediction.
