
# g2_hurdle (Global 2-Stage Hurdle Modeling Toolkit)

## 개요 (Overview)

이 코드베이스는 레스토랑 메뉴 수요 예측을 위한 2단계 허들 모델(Two-Stage Hurdle Model)을 구현합니다. 이 모델은 특정 메뉴가 특정 날짜에 판매될지 여부를 먼저 예측한 다음(1단계: Classification), 판매가 발생할 경우 그 수량을 예측합니다(2단계: Regression). 이러한 접근 방식은 간헐적인 수요 패턴(판매가 발생하지 않는 날이 많은 경우)을 보이는 시계열 데이터에 효과적입니다.

**wSMAPE 점수:** 0.5550080421

---

## 파이프라인 (Pipeline)

### 1. 데이터 전처리 (Data Preprocessing)

- **데이터 로딩**: `g2_hurdle/utils/io.py`의 `load_data` 함수를 사용하여 학습 데이터(`train.csv`)와 테스트 데이터(`test/TEST_*.csv`)를 불러옵니다. 이 과정에서 `g2_hurdle/configs/korean.yaml` 설정 파일에 정의된 날짜, 타겟, 시리즈 컬럼을 식별합니다.
- **데이터 스키마 확인**: `g2_hurdle/data/schema.py`의 `resolve_schema`를 통해 데이터의 스키마(날짜, 타겟, 시리즈 컬럼 등)를 명확히 정의합니다.
- **기본 변환**: 날짜 컬럼을 `datetime` 객체로 변환하고, 타겟 변수(매출 수량)를 숫자형으로 변환합니다.

### 2. 피처 엔지니어링 (Feature Engineering)

- **`g2_hurdle/fe/__init__.py`의 `run_feature_engineering` 함수를 통해 다음과 같은 피처들을 생성합니다:**
    - **달력 피처**: `g2_hurdle/fe/calendar.py`
        - 년, 월, 일, 요일, 주차, 분기 등 날짜와 관련된 기본적인 피처를 생성합니다. 이를 통해 모델이 시간의 흐름과 주기성을 학습할 수 있습니다.
    - **푸리에 변환 피처**: `g2_hurdle/fe/fourier.py`
        - 연간, 분기별, 월간 주기성을 모델링하기 위해 푸리에 변환을 적용한 피처를 생성합니다. 이는 계절적 패턴을 더 잘 포착하는 데 도움을 줍니다.
    - **지연 및 롤링 집계 피처**: `g2_hurdle/fe/lags_rolling.py`
        - 과거의 매출 수량(지연 피처)과 특정 기간 동안의 이동 평균, 표준편차 등(롤링 집계 피처)을 생성합니다. 이를 통해 최근의 판매 추세와 변동성을 모델이 학습하게 됩니다.
    - **간헐성 피처**: `g2_hurdle/fe/intermittency.py`
        - 마지막 판매일로부터 경과된 시간 등 간헐적인 판매 패턴과 관련된 피처를 생성합니다. 이는 판매가 드물게 발생하는 상품의 특성을 모델링하는 데 중요합니다.

### 3. 모델 학습 (Model Training)

- **학습/검증 데이터 분할**: `g2_hurdle/pipeline/train.py`의 `_split_train_valid_by_tail_dates` 함수를 사용하여 시계열 데이터의 특성을 고려한 방식으로 학습 데이터와 검증 데이터를 분할합니다.
- **2단계 허들 모델 학습**:
    - **1단계: Classifier**: `g2_hurdle/model/classifier.py`
        - 매출이 발생했는지 여부(0 또는 1)를 예측하는 LightGBM `LGBMClassifier` 모델을 학습합니다.
    - **2단계: Regressor**: `g2_hurdle/model/regressor.py`
        - 매출이 발생한 경우의 매출 수량을 예측하는 LightGBM `LGBMRegressor` 모델을 학습합니다.
- **최적 임계값 탐색**: `g2_hurdle/model/threshold.py`의 `find_optimal_threshold` 함수를 사용하여 검증 데이터에 대한 wSMAPE 점수를 최적화하는 확률 임계값을 찾습니다. 이 임계값은 Classifier가 예측한 확률을 바탕으로 매출 발생 여부를 최종 결정하는 데 사용됩니다.

### 4. 추론 (Inference)

- **`g2_hurdle/pipeline/predict.py`의 `run_predict` 함수를 통해 예측을 수행합니다.**
- **재귀적 예측**: `g2_hurdle/pipeline/recursion.py`
    - 테스트 기간 동안 하루씩 예측을 수행하며, 예측된 값을 다음 날의 피처로 사용하는 재귀적 방식을 사용합니다. 이는 미래 시점의 피처를 동적으로 생성하여 예측의 정확도를 높이는 데 기여합니다.
- **최종 예측 생성**: 학습된 Classifier와 Regressor, 그리고 최적 임계값을 사용하여 최종 매출 수량을 예측하고, `sample_submission.csv` 형식에 맞춰 결과를 저장합니다.

---

## 중간 과정의 의미 (Significance of Intermediate Processes)

### 허들 모델 (Hurdle Model)

- 많은 메뉴들이 매일 판매되지 않는 '간헐적 수요' 패턴을 보입니다. 허들 모델은 이러한 문제에 효과적인 접근법으로, '판매 발생 여부'와 '판매량'이라는 두 가지 문제를 분리하여 모델링합니다. 이를 통해 각 문제에 최적화된 모델을 학습시켜 전반적인 예측 성능을 향상시킬 수 있습니다.

### 피처 엔지니어링 (Feature Engineering)

- **시간 기반 피처**: 달력 피처와 푸리에 변환 피처는 모델이 요일, 월, 계절 등 시간에 따른 주기적인 패턴을 학습하는 데 필수적입니다.
- **과거 데이터 기반 피처**: 지연 및 롤링 집계 피처는 최근의 판매 추세나 변동성을 모델에 알려주는 중요한 정보입니다. 예를 들어, 어제 많이 팔린 메뉴는 오늘도 많이 팔릴 가능성이 높다는 정보를 모델이 활용할 수 있게 합니다.
- **간헐성 피처**: 마지막 판매일로부터 얼마나 지났는지 등의 정보는 가끔 판매되는 메뉴의 수요를 예측하는 데 결정적인 역할을 합니다.

### 시계열 교차 검증 (Time Series Cross-Validation)

- 일반적인 교차 검증 방식과 달리, 시계열 데이터에서는 시간의 흐름을 고려하여 과거 데이터로 미래 데이터를 예측하는 방식으로 검증해야 합니다. 이 코드베이스에서는 `rolling_forecast_origin_split`을 사용하여 이러한 시계열 특성을 반영한 검증을 수행함으로써, 모델의 일반화 성능을 더 정확하게 평가합니다.

---
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
