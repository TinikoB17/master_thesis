"""Standalone experiment (not part of the Snakemake pipeline): a small neural network
(multi-layer perceptron) evaluated with the same repeated nested CV as repeated_cv.py.

Same samples, outer splits, in-fold preprocessing and tuning metric (balanced MAE) as the
"all miRNAs" models, so the results are directly comparable with
models/repeated_cv/*_summary.txt.

Run from the repository root:
    python scripts/neural_network_cv.py              # 5 folds x 10 repeats
    python scripts/neural_network_cv.py --repeats 2  # quicker try
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import set_config
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.parallel import Parallel, delayed
from threadpoolctl import threadpool_limits

from cv_preprocessing import META_COLS, PrevalenceFilterLog, ProjectZScore, SkewedPowerTransform
from metrics import AGE_BINS, balanced_mae, balanced_mae_scorer

set_config(transform_output="pandas")

INPUT_H5AD = "test_output/age_filtered_blood_healthy.h5ad"
OUTPUT_DIR = Path("models/experimental")
N_SPLITS = 5
RANDOM_STATE = 42

PARAM_GRID = {
    "model__regressor__hidden_layer_sizes": [(32,), (64,), (64, 32)],
    # L2 penalty; strong values are needed with ~350 inputs and ~600 training samples
    "model__regressor__alpha": [0.1, 1.0, 10.0, 30.0],
}


def build_search() -> GridSearchCV:
    network = MLPRegressor(
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,  # holds out 15 % of the training fold to stop before overfitting
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline(steps=[
        ("prevalence", PrevalenceFilterLog()),
        ("zscore", ProjectZScore()),
        ("preprocess", SkewedPowerTransform()),
        ("scale", StandardScaler()),
        # Networks train poorly on a raw target around 60 years, so age is standardized too
        ("model", TransformedTargetRegressor(regressor=network, transformer=StandardScaler())),
    ])
    return GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring=balanced_mae_scorer,
        cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
    )


def load_data():
    adata = sc.read_h5ad(INPUT_H5AD)
    X = adata.to_df().join(adata.obs[META_COLS].astype(str))
    y = adata.obs["Age"].astype(float)
    return X, y


def run_outer_fold(fold_id, repeat, train_idx, test_idx, X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    with threadpool_limits(limits=1):
        search = build_search().fit(X_train, y_train)
        y_pred = search.predict(X_test)

    fold_result = {
        "repeat": repeat,
        "fold": fold_id,
        "MAE": mean_absolute_error(y_test, y_pred),
        "balanced_MAE": balanced_mae(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "baseline_MAE": mean_absolute_error(y_test, np.full(len(y_test), y_train.mean())),
        "inner_cv_balanced_MAE": -search.best_score_,
        "best_params": str(search.best_params_),
    }
    predictions = pd.DataFrame({
        "Sample": X_test.index,
        "repeat": repeat,
        "fold": fold_id,
        "Project": X_test["Project"].values,
        "actual_age": y_test.values,
        "predicted_age": y_pred,
    })
    return fold_result, predictions


def summarize(folds: pd.DataFrame, oof: pd.DataFrame, n_repeats: int) -> str:
    lines = [
        "Model: mlp (sklearn MLPRegressor), all miRNAs",
        f"Repeated nested CV: {N_SPLITS} outer folds x {n_repeats} repeats, {N_SPLITS}-fold inner CV",
        f"Samples: {oof['Sample'].nunique()}",
        "",
        "Across all outer folds (mean +- sd, [2.5%, 97.5%]):",
    ]
    for metric in ["MAE", "balanced_MAE", "RMSE", "R2", "baseline_MAE"]:
        values = folds[metric]
        low, high = np.percentile(values, [2.5, 97.5])
        lines.append(f"  {metric}: {values.mean():.3f} +- {values.std(ddof=1):.3f}  [{low:.3f}, {high:.3f}]")

    errors = (oof["predicted_age"] - oof["actual_age"])
    age_bins = pd.cut(oof["actual_age"], bins=AGE_BINS, include_lowest=True)
    per_bin = pd.DataFrame({"MAE": errors.abs(), "bias": errors}).groupby(age_bins, observed=True).mean()
    lines += ["", "Out-of-fold error per age bin (pooled over repeats):"]
    for age_bin, row in per_bin.iterrows():
        lines.append(f"  {str(age_bin):<16} MAE {row['MAE']:6.2f}   bias {row['bias']:+6.2f}")

    lines += ["", "Selected hyperparameters (count over outer folds):"]
    for params, count in folds["best_params"].value_counts().items():
        lines.append(f"  {count:>3}x {params}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=10, help="outer CV repeats (default: 10)")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel outer folds (default: all cores)")
    args = parser.parse_args()

    X, y = load_data()
    print(f"mlp: {X.shape[0]} samples, {X.shape[1] - len(META_COLS)} miRNAs")

    # Same random_state as repeated_cv.py, so the outer splits match the pipeline models
    outer_cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=args.repeats, random_state=RANDOM_STATE)
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_outer_fold)(i % N_SPLITS, i // N_SPLITS, train_idx, test_idx, X, y)
        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X))
    )
    folds = pd.DataFrame([r[0] for r in results])
    oof = pd.concat([r[1] for r in results], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUTPUT_DIR / "mlp_folds.tsv", sep="\t", index=False)
    oof.to_csv(OUTPUT_DIR / "mlp_oof_predictions.tsv", sep="\t", index=False)
    summary = summarize(folds, oof, args.repeats)
    (OUTPUT_DIR / "mlp_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
