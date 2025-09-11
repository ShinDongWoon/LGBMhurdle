# -*- coding: utf-8 -*-
"""
lgbm_sanity_check.py

목표: LightGBM에서 "[Warning] No further splits with positive gain, best gain: -inf"가 반복될 때,
데이터/세팅의 근본 원인을 자동 점검하고, 바로 학습 가능한 '안정 레시피'로 재시도한다.

사용법:
    python lgbm_sanity_check.py --csv ./train.csv --nrows 300000

필요 패키지:
    pandas, numpy, lightgbm (필요시: pip install lightgbm pandas numpy)
"""
import argparse, sys, os, warnings
import numpy as np, pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--date_col", type=str, default="영업일자")
    ap.add_argument("--key_col", type=str, default="store_menu_id")
    ap.add_argument("--target_col", type=str, default="매출수량")
    ap.add_argument("--nrows", type=int, default=None, help="디버깅을 위해 일부만 읽고 싶다면 지정")
    ap.add_argument("--valid_days", type=int, default=28, help="검증 기간(일)")
    ap.add_argument("--min_child_samples", type=int, default=30)
    ap.add_argument("--min_sum_hessian_in_leaf", type=float, default=1e-3)
    ap.add_argument("--max_bin", type=int, default=1023)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def safe_import_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        print("[ERR] lightgbm import 실패:", e, file=sys.stderr)
        print("=> pip install lightgbm", file=sys.stderr)
        sys.exit(1)

def make_calendar_feats(df, date_col):
    df["dom"] = df[date_col].dt.day.astype(np.int16)
    df["month"] = df[date_col].dt.month.astype(np.int16)
    df["week"] = df[date_col].dt.isocalendar().week.astype(np.int16)
    df["is_month_end"]   = df[date_col].dt.is_month_end.astype(np.int8)
    for col, period in [("week", 52), ("month", 12)]:
        val = df[col].astype(float)
        df[f"{col}_sin"] = np.sin(2 * np.pi * val / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * val / period)
    df.drop(columns=["month"], inplace=True)
    return df

def add_group_lags(df, key_col, target_col, lags=(1,7,14,28)):
    df = df.sort_values([key_col, "영업일자"])
    for L in lags:
        df[f"lag_{L}"] = df.groupby(key_col)[target_col].shift(L)
    # 이동평균/합
    df["ma7"]  = df.groupby(key_col)[target_col].shift(1).rolling(7, min_periods=1).mean()
    df["sum7"] = df.groupby(key_col)[target_col].shift(1).rolling(7, min_periods=1).sum()
    # zero-run 길이
    # (이전 상태에서 0이 연속으로 몇 번 나왔는지)
    z = (df[target_col] == 0).astype(int)
    df["zero_run"] = (
        z.groupby(df[key_col])
         .transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
         .where(z == 1, 0)
    )
    df.fillna(0, inplace=True)
    return df

def build_dataset(args):
    df = pd.read_csv(args.csv, nrows=args.nrows, low_memory=False)
    # 날짜 파싱
    if args.date_col not in df.columns:
        # 자동 탐색
        for c in df.columns:
            if any(k in str(c).lower() for k in ["date","ds","일자","날짜","기준일자"]):
                args.date_col = c; break
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.dropna(subset=[args.date_col])
    # 열 이름 통일(편의)
    if args.date_col != "영업일자":
        df = df.rename(columns={args.date_col: "영업일자"})
    if args.key_col not in df.columns or args.target_col not in df.columns:
        raise ValueError("key/target 컬럼을 찾을 수 없습니다. --key_col/--target_col 확인")
    # 피처 제작
    df = make_calendar_feats(df, "영업일자")
    df = add_group_lags(df, args.key_col, args.target_col)
    # 카테고리형
    for c in [args.key_col]:
        df[c] = df[c].astype("category")
    # 학습 가능한 구간만
    base_cols = ["dom","week","is_month_end","week_sin","week_cos","month_sin","month_cos",
                 "zero_run","lag_1","lag_7","lag_14","lag_28","ma7","sum7", args.key_col]
    X = df[base_cols]
    y = df[args.target_col]
    d = df[["영업일자"]].copy()
    return X, y, d, base_cols

def time_split(X, y, d, valid_days):
    max_date = d["영업일자"].max()
    valid_start = max_date - pd.Timedelta(days=valid_days-1)
    trn_idx = d["영업일자"] < valid_start
    val_idx = ~trn_idx
    return trn_idx.values, val_idx.values, valid_start, max_date

def finite_report(X, y):
    rep = {}
    rep["x_isfinite"] = np.isfinite(np.asarray(pd.DataFrame(X).select_dtypes(include=[np.number]))).all()
    rep["y_isfinite"] = np.isfinite(y.values).all()
    rep["x_has_nan"] = pd.DataFrame(X).isna().any().any()
    rep["y_has_nan"] = pd.isna(y).any()
    return rep

def fold_report(X, y, d, trn_idx, val_idx):
    def subrep(idx, name):
        sub = {
            f"{name}_n": int(idx.sum()),
            f"{name}_y_var": float(pd.Series(y.loc[idx]).var(ddof=1)) if idx.sum()>1 else float("nan"),
            f"{name}_unique_dow": int(d.loc[idx, "영업일자"].dt.weekday.nunique()),
            f"{name}_unique_month": int(d.loc[idx, "영업일자"].dt.month.nunique()),
        }
        return sub
    r = {}
    r.update(subrep(trn_idx, "train"))
    r.update(subrep(val_idx, "valid"))
    return r

def train_lgbm(args, X, y, trn_idx, val_idx):
    lgb = safe_import_lgbm()
    # 수치/범주 분리
    cat_cols = [i for i,c in enumerate(X.columns) if str(X.columns[i]) == args.key_col]
    # 안전 파라미터
    params = dict(
        objective="rmse",  # 우선 단순 회귀로 학습 안정성 확인
        metric="rmse",
        learning_rate=0.03,
        n_estimators=3000,
        num_leaves=127,
        max_bin=args.max_bin,
        min_child_samples=args.min_child_samples,
        min_sum_hessian_in_leaf=args.min_sum_hessian_in_leaf,
        feature_pre_filter=False,
        min_data_in_bin=1,
        bagging_fraction=0.9,
        bagging_freq=1,
        feature_fraction=0.9,
        seed=args.seed,
        verbose=-1,
        zero_as_missing=False,
        force_col_wise=True,   # 안전
    )
    # Dataset 생성
    dtrain = lgb.Dataset(X.iloc[trn_idx], y.iloc[trn_idx], categorical_feature=cat_cols, free_raw_data=False)
    dvalid = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx], categorical_feature=cat_cols, free_raw_data=False)
    # 학습
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train","valid"],
        num_boost_round=5000,
        callbacks=[lgb.early_stopping(200)],
        keep_training_booster=True,
    )
    return model

def main():
    args = parse_args()
    X, y, d, cols = build_dataset(args)
    trn_idx, val_idx, valid_start, valid_end = time_split(X, y, d, args.valid_days)

    print("### Fold info")
    print(f"valid window: {valid_start.date()} ~ {valid_end.date()}")
    rep = fold_report(X, y, d, trn_idx, val_idx)
    for k,v in rep.items():
        print(f"{k}: {v}")

    fin = finite_report(X, y)
    print("\n### Finite/NaN check")
    for k,v in fin.items():
        print(f"{k}: {v}")

    # 경고: 폴드 내 y 분산 0이면 바로 종료
    if rep["train_y_var"] == 0 or np.isnan(rep["train_y_var"]):
        print("\n[FAIL] train fold의 y 분산이 0 또는 NaN → 분할 이득 없음의 직접 원인")
        sys.exit(2)

    # 학습 시도
    try:
        print("\n### Train LightGBM (safe recipe)")
        model = train_lgbm(args, X, y, trn_idx, val_idx)
        print("\nBest iteration:", model.best_iteration)
        print("Valid RMSE:", model.best_score.get("valid", {}).get("rmse"))
    except Exception as e:
        print("\n[EXC] LightGBM 학습 중 예외:", e)
        print("=> GPU/버전 문제 또는 카테고리 처리 충돌 가능성. CPU 실행/버전 교체/원핫으로 재시도 권장.")
        sys.exit(3)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
