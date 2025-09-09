# LGBM Hurdle Model for Time Series Forecasting

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/shindongwoon/lgbmhurdle)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![wSMAPE Score](https://img.shields.io/badge/wSMAPE-0.555008-orange)](.)

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

## ⚙️ 코드 파이프라인 분석 (Code Pipeline Analysis)

본 코드베이스는 크게 **데이터 전처리 및 피처 엔지니어링**, **모델 학습**, **재귀적 예측**의 세 단계로 구성됩니다.

### 1. 전체 코드 파이프라인 흐름

![Pipeline Flow](https://i.imgur.com/gK9qASs.png)

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

## 🚀 시작하기 (Getting Started)

### 1. 요구사항 (Prerequisites)

- Python 3.8+
- poetry

## 📂 데이터 디렉토리 구조 (Data Directory Structure)

보안상의 이유로 원본 데이터는 저장소에 포함되어 있지 않습니다. 코드를 정상적으로 실행하려면 프로젝트 최상위 경로에 `data` 디렉토리를 생성하고 아래와 같은 구조로 데이터를 배치해야 합니다.
"""
lgbmhurdle/
├── data/
│   ├── train.csv               # 훈련 데이터
│   ├── test/                   # 평가(테스트) 데이터 폴더
│   │   ├── TEST_00.csv
│   │   ├── TEST_01.csv
│   │   └── ...                 # 여러 개의 테스트 파일
│   └── sample_submission.csv   # 제출 양식 파일
├── g2_hurdle/
├── scripts/
└── ...
"""
- **/data/train.csv**: 모델 학습을 위한 시계열 데이터입니다.
- **/data/test/**: 예측을 수행할 평가용 데이터셋입니다. 각 `TEST_*.csv` 파일은 특정 기간의 데이터를 담고 있으며, 모델은 이 파일의 마지막 시점으로부터 7일 후까지의 판매량을 예측해야 합니다.
- **/data/sample_submission.csv**: 최종 예측 결과를 어떤 형식으로 저장해야 하는지 정의하는 샘플 파일입니다.

---
### 2. 설치 (Installation)

```bash
# 1. 저장소 클론
git clone [https://github.com/shindongwoon/lgbmhurdle.git](https://github.com/shindongwoon/lgbmhurdle.git)
cd lgbmhurdle

# 2. 의존성 설치
poetry install

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
