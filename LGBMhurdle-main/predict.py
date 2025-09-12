import os
import yaml
from g2_hurdle.pipeline.predict import run_predict

def main():
    with open("g2_hurdle/configs/korean.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {
        "test_dir": "data/test",
        "sample_submission": "data/sample_submission.csv",
        "out_path": "outputs/submission.csv",
        "artifacts_dir": "./artifacts",
    }
    os.makedirs("outputs", exist_ok=True)
    run_predict(cfg)

if __name__ == "__main__":
    main()
