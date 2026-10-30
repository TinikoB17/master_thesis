import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import set_config
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.parallel import Parallel, delayed
from threadpoolctl import threadpool_limits

from cv_preprocessing import (META_COLS, CorrelationFilter, PrevalenceFilterLog, ProjectZScore,
                              SkewedPowerTransform)
from metrics import balanced_mae, balanced_mae_scorer

# Repeated nested cross-validation on all healthy blood samples (age 30-80).
# Outer loop: RepeatedKFold (5 folds x 10 repeats) -> performance estimate.
# Inner loop: GridSearchCV (5 folds) -> hyperparameter tuning on the balanced MAE
# (mean of per-age-bin MAEs, see metrics.py), so rare young/old ages weigh as much as common ones.
# Every step that learns from data runs inside the pipeline, i.e. it is refit on each
# training fold only: prevalence filter, per-project z-score, correlation filter,
# power transform, feature selection, model.

set_config(transform_output="pandas")

INPUT_H5AD = snakemake.input.age_filtered
MODEL_NAME = snakemake.wildcards.model
FOLDS_OUT = snakemake.output.folds
OOF_OUT = snakemake.output.oof_predictions
SUMMARY_OUT = snakemake.output.summary
PLOT_OUT = snakemake.output.plot
N_JOBS = snakemake.threads

N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42


def build_search(model_name: str) -> GridSearchCV:
    # Grids and feature sets mirror the existing *_tuned scripts.
    correlated = model_name in {"linear_regression", "linear_regression_pol2", "lasso"}

    steps = [
        ("prevalence", PrevalenceFilterLog()),
        ("zscore", ProjectZScore()),
    ]
    if correlated:
        steps.append(("correlation", CorrelationFilter()))
    steps.append(("preprocess", SkewedPowerTransform(power_transform=model_name != "hgb")))
    steps.append(("feature_selection", SelectKBest(score_func=f_regression)))

    if model_name == "linear_regression":
        steps.append(("model", LinearRegression()))
        param_grid = {"feature_selection__k": [50, 70, 100, 200, "all"]}
    elif model_name == "linear_regression_all":
        steps.append(("model", LinearRegression()))
        param_grid = {"feature_selection__k": [50, 100, 120, 150, 200, 250, "all"]}
    elif model_name == "linear_regression_pol2":
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
        steps.append(("model", LinearRegression()))
        param_grid = {"feature_selection__k": [5, 10, 20]}
    elif model_name in {"lasso", "lasso_all"}:
        steps.append(("model", Lasso(random_state=RANDOM_STATE, max_iter=100000, tol=1e-4, selection="cyclic")))
        # 0.001-10: with balanced-MAE tuning the best alpha sat at the old lower bound (0.1)
        param_grid = {"model__alpha": np.logspace(-3, 1, 25), "feature_selection__k": ["all"]}
    elif model_name == "hgb":
        steps.append(("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)))
        param_grid = {
            "feature_selection__k": ["all"],
            "model__max_iter": [1000],
            "model__learning_rate": [0.05],
            "model__max_leaf_nodes": [31],
            "model__min_samples_leaf": [2],
            "model__l2_regularization": [0.1],
            "model__loss": ["absolute_error"],
            "model__max_depth": [8],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return GridSearchCV(
        estimator=Pipeline(steps=steps),
        param_grid=param_grid,
        scoring=balanced_mae_scorer,
        cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        refit=True,
    )


def load_data():
    adata = sc.read_h5ad(INPUT_H5AD)
    X = adata.to_df().join(adata.obs[META_COLS].astype(str))
    y = adata.obs["Age"].astype(float)
    return X, y


def run_outer_fold(fold_id, repeat, train_idx, test_idx, X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # One thread per outer fold: the folds already run in parallel, and HGB's OpenMP
    # threads in every worker would otherwise oversubscribe the CPU.
    with threadpool_limits(limits=1):
        search = build_search(MODEL_NAME)
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
    baseline = np.full(len(y_test), y_train.mean())

    best = search.best_estimator_
    n_mirnas = best.named_steps["feature_selection"].get_support().sum()

    fold_result = {
        "repeat": repeat,
        "fold": fold_id,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "MAE": mean_absolute_error(y_test, y_pred),
        "balanced_MAE": balanced_mae(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "MedianAE": median_absolute_error(y_test, y_pred),
        "baseline_MAE": mean_absolute_error(y_test, baseline),
        "n_features_used": int(n_mirnas),
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


def summarize(folds: pd.DataFrame, oof: pd.DataFrame) -> str:
    lines = [
        f"Model: {MODEL_NAME}",
        f"Repeated nested CV: {N_SPLITS} outer folds x {N_REPEATS} repeats, {N_SPLITS}-fold inner CV",
        f"Samples: {oof['Sample'].nunique()}",
        "",
        "Across all outer folds (mean +- sd, [2.5%, 97.5%]):",
    ]
    for metric in ["MAE", "balanced_MAE", "RMSE", "R2", "MedianAE", "baseline_MAE"]:
        values = folds[metric]
        low, high = np.percentile(values, [2.5, 97.5])
        lines.append(f"  {metric}: {values.mean():.3f} +- {values.std(ddof=1):.3f}  [{low:.3f}, {high:.3f}]")

    # Pooled out-of-fold metrics: one prediction per sample per repeat
    per_repeat = oof.groupby("repeat").apply(
        lambda d: pd.Series({
            "MAE": mean_absolute_error(d["actual_age"], d["predicted_age"]),
            "balanced_MAE": balanced_mae(d["actual_age"], d["predicted_age"]),
            "R2": r2_score(d["actual_age"], d["predicted_age"]),
        }),
        include_groups=False,
    )
    lines += [
        "",
        "Pooled out-of-fold predictions per repeat (mean +- sd across repeats, [min, max]):",
    ]
    for metric in ["MAE", "balanced_MAE", "R2"]:
        values = per_repeat[metric]
        lines.append(f"  {metric}: {values.mean():.3f} +- {values.std(ddof=1):.3f}  [{values.min():.3f}, {values.max():.3f}]")

    lines += ["", "Selected hyperparameters (count over outer folds):"]
    for params, count in folds["best_params"].value_counts().items():
        lines.append(f"  {count:>3}x {params}")
    return "\n".join(lines) + "\n"


def plot_oof(oof: pd.DataFrame):
    mean_pred = oof.groupby("Sample").agg(actual_age=("actual_age", "first"),
                                          predicted_age=("predicted_age", "mean"))
    low, high = mean_pred["actual_age"].min(), mean_pred["actual_age"].max()
    plt.figure(figsize=(6, 6))
    plt.scatter(mean_pred["actual_age"], mean_pred["predicted_age"], alpha=0.6)
    plt.plot([low, high], [low, high], linestyle="--", color="darkorange")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (mean over repeats)")
    plt.title(f"{MODEL_NAME} - repeated nested CV predictions")
    plt.tight_layout()
    plt.savefig(PLOT_OUT)
    plt.close()


def main():
    X, y = load_data()
    print(f"{MODEL_NAME}: {X.shape[0]} samples, {X.shape[1] - len(META_COLS)} miRNAs")

    outer_cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    jobs = [
        (i % N_SPLITS, i // N_SPLITS, train_idx, test_idx)
        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X))
    ]
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_outer_fold)(fold_id, repeat, train_idx, test_idx, X, y)
        for fold_id, repeat, train_idx, test_idx in jobs
    )

    folds = pd.DataFrame([r[0] for r in results])
    oof = pd.concat([r[1] for r in results], ignore_index=True)

    folds.to_csv(FOLDS_OUT, sep="\t", index=False)
    oof.to_csv(OOF_OUT, sep="\t", index=False)

    summary = summarize(folds, oof)
    print(summary)
    with open(SUMMARY_OUT, "w") as file:
        file.write(summary)

    plot_oof(oof)


if __name__ == "__main__":
    main()
