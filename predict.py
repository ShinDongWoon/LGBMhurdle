import os
import yaml
from g2_hurdle.pipeline.predict import run_predict


def deep_merge(base, overlay):
    """Recursively merge two dictionaries."""
    result = base.copy()
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def main():
    with open("g2_hurdle/configs/base.yaml", "r") as f:
        cfg_base = yaml.safe_load(f)
    with open("g2_hurdle/configs/korean.yaml", "r") as f:
        cfg_korean = yaml.safe_load(f)
    cfg = deep_merge(cfg_base, cfg_korean)
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
