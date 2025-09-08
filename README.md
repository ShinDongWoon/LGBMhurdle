
# g2_hurdle (Global 2-Stage Hurdle Modeling Toolkit)

## Quick Start
```bash
python -m g2_hurdle.cli train \
  --train_csv data/test.csv \
  --config g2_hurdle/configs/base.yaml

python -m g2_hurdle.cli predict \
  --test_dir data/test \
  --sample_submission sample_submission.csv \
  --out_path outputs/submission.csv
```
All imports are relative; drop this folder as project root and run the commands.
