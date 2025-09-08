
# g2_hurdle (Global 2-Stage Hurdle Modeling Toolkit)

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
