import yaml
from g2_hurdle.pipeline.train import run_train


def main():
    with open("g2_hurdle/configs/korean.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {"train_csv": "data/train.csv"}
    run_train(cfg)


if __name__ == "__main__":
    main()
