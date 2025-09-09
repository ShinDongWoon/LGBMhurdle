import pandas as pd
from pathlib import Path


def test_submission_format(tmp_path):
    sample_path = Path("data/sample_submission.csv")
    sample = pd.read_csv(sample_path, encoding="utf-8")
    submission = sample.copy()
    if submission.shape[1] > 1:
        submission.iloc[:, 1:] = 0
    submission_path = tmp_path / "submission.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")

    generated = pd.read_csv(submission_path, encoding="utf-8")

    assert list(sample.columns) == list(generated.columns)

    with open(sample_path, "rb") as f:
        sample_header = f.readline()
    with open(submission_path, "rb") as f:
        submission_header = f.readline()

    assert sample_header == submission_header
    for header in (sample_header, submission_header):
        line = header.rstrip(b"\n")
        assert line == line.rstrip(b" ")
