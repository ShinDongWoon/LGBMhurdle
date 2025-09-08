
import argparse
import os
import yaml

from .pipeline.train import run_train
from .pipeline.predict import run_predict

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(prog="g2_hurdle", description="Global 2-Stage Hurdle Model CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train hurdle models")
    p_train.add_argument("--train_csv", required=True, help="Path to training CSV (expected: data/train.csv)")
    p_train.add_argument("--config", required=True, help="YAML config path")

    p_pred = sub.add_parser("predict", help="Predict with trained models and create submission")
    p_pred.add_argument("--test_dir", required=True, help="Directory containing TEST_00.csv ... TEST_09.csv")
    p_pred.add_argument("--sample_submission", required=True, help="Path to sample_submission.csv")
    p_pred.add_argument("--out_path", required=True, help="Output CSV path")
    p_pred.add_argument("--artifacts_dir", default="./artifacts", help="Artifacts dir (default: ./artifacts)")

    args = parser.parse_args()
    cfg = load_config(args.config)

    # inject runtime paths
    if args.cmd == "train":
        cfg["paths"] = {"train_csv": args.train_csv}
        run_train(cfg)
    elif args.cmd == "predict":
        cfg["paths"] = {
            "test_dir": args.test_dir,
            "sample_submission": args.sample_submission,
            "out_path": args.out_path,
            "artifacts_dir": getattr(args, "artifacts_dir", "./artifacts"),
        }
        run_predict(cfg)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
