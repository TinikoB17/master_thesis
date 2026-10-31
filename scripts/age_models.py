import anndata as ad
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import f_regression
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import (make_scorer, mean_absolute_error, median_absolute_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Shared pieces of the age-prediction workflow: data loading, the preprocessing steps
# (each learns from the training data only), model definitions with their tuning grids,
# and the metrics.
#
# Input X: raw RPMM per miRNA plus the metadata columns "Project" (str) and "Sex" (0/1, m = 1).
# Pipeline: prevalence filter + log2(1+x) -> per-project z-score -> top-k miRNAs by
# F-test with age (Sex always kept) -> model.

META_COLS = ["Project", "Sex"]


def load_cohort(path):
    adata = ad.read_h5ad(path)
    X = adata.to_df()
    X["Project"] = adata.obs["Project"].astype(str)
    X["Sex"] = (adata.obs["Sex"].astype(str) == "m").astype(float)
    y = adata.obs["Age"].astype(float)
    return X, y


def mirna_columns(X):
    return [col for col in X.columns if col not in META_COLS]


class PrevalenceFilterLog(BaseEstimator, TransformerMixin):
    """Keep miRNAs with RPMM > min_rpmm in >= min_fraction of training samples, then log2(1 + x)."""

    def __init__(self, min_rpmm=1.0, min_fraction=0.2):
        self.min_rpmm = min_rpmm
        self.min_fraction = min_fraction

    def fit(self, X, y=None):
        mirnas = mirna_columns(X)
        fraction = (X[mirnas] > self.min_rpmm).mean(axis=0)
        self.mirnas_ = fraction[fraction >= self.min_fraction].index.tolist()
        return self

    def transform(self, X):
        return np.log2(1 + X[self.mirnas_]).join(X[META_COLS])


class ProjectZScore(BaseEstimator, TransformerMixin):
    """Per-project z-score. Projects with >= min_samples training samples use their training
    mean/sd. A project not seen in training (leave-project-out) is standardized with its own
    mean/sd if it has >= min_samples samples; this uses no age labels. Everything else falls
    back to the global training mean/sd. Drops "Project"."""

    def __init__(self, min_samples=5):
        self.min_samples = min_samples

    def fit(self, X, y=None):
        self.mirnas_ = mirna_columns(X)
        grouped = X.groupby("Project")[self.mirnas_]
        large_enough = grouped.size() >= self.min_samples
        self.means_ = grouped.mean()[large_enough]
        self.stds_ = grouped.std(ddof=1)[large_enough]
        self.global_mean_ = X[self.mirnas_].mean()
        self.global_std_ = X[self.mirnas_].std(ddof=1)
        return self

    def transform(self, X):
        means = self.means_.copy()
        stds = self.stds_.copy()
        unseen = X.loc[~X["Project"].isin(means.index)].groupby("Project")[self.mirnas_]
        unseen_large = unseen.size() >= self.min_samples
        if unseen_large.any():
            means = pd.concat([means, unseen.mean()[unseen_large]])
            stds = pd.concat([stds, unseen.std(ddof=1)[unseen_large]])

        projects = X["Project"]
        row_means = means.reindex(projects).set_index(X.index).fillna(self.global_mean_)
        row_stds = stds.reindex(projects).set_index(X.index).fillna(self.global_std_)
        row_stds = row_stds.mask(row_stds == 0, 1.0).fillna(1.0)
        out = ((X[self.mirnas_] - row_means) / row_stds).fillna(0)
        return out.join(X[["Sex"]])


class TopKMirnas(BaseEstimator, TransformerMixin):
    """Keep the k miRNAs with the highest F-statistic against age; "Sex" is always kept."""

    def __init__(self, k="all"):
        self.k = k

    def fit(self, X, y):
        mirnas = [col for col in X.columns if col != "Sex"]
        if self.k == "all" or self.k >= len(mirnas):
            self.mirnas_ = mirnas
        else:
            f_stat, _ = f_regression(X[mirnas], y)
            order = np.argsort(np.nan_to_num(f_stat, nan=-1.0))[::-1]
            self.mirnas_ = [mirnas[i] for i in order[: self.k]]
        return self

    def transform(self, X):
        return X[self.mirnas_ + ["Sex"]]


def build_pipeline(model_name, preprocessing_cfg, random_state=42):
    if model_name == "mean":
        return Pipeline([("model", DummyRegressor(strategy="mean"))]), {}

    steps = [
        ("prevalence", PrevalenceFilterLog(preprocessing_cfg["min_rpmm"], preprocessing_cfg["min_fraction"])),
        ("zscore", ProjectZScore(preprocessing_cfg["zscore_min_samples"])),
        ("select", TopKMirnas()),
    ]
    if model_name == "linear":
        steps.append(("model", LinearRegression()))
        grid = {"select__k": [10, 25, 50, 100, 200]}
    elif model_name == "elastic_net":
        steps.append(("model", ElasticNet(max_iter=20000, tol=1e-3, random_state=random_state)))
        grid = {
            "select__k": [100, 200, "all"],
            "model__alpha": np.logspace(-2, 0.5, 8),
            "model__l1_ratio": [0.5, 1.0],
        }
    elif model_name == "hgb":
        steps.append(("model", HistGradientBoostingRegressor(
            max_iter=500, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
            min_samples_leaf=20, l2_regularization=1.0, random_state=random_state)))
        grid = {
            "select__k": [100, "all"],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [3, 6],
        }
    elif model_name == "mlp":
        network = MLPRegressor(
            solver="adam", learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, random_state=random_state)
        steps.append(("scale", StandardScaler()))
        # Networks train poorly on a raw target around 60 years, so age is standardized too
        steps.append(("model", TransformedTargetRegressor(regressor=network, transformer=StandardScaler())))
        grid = {
            "select__k": [100, "all"],
            "model__regressor__hidden_layer_sizes": [(32,), (64, 32)],
            "model__regressor__alpha": [1.0, 10.0, 30.0],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline(steps), grid


def age_strata(y, bins):
    return pd.cut(np.asarray(y), bins=bins, include_lowest=True, labels=False)


def fit_search(model_name, config, X, y, n_jobs=1):
    """Tune on (X, y) with GridSearchCV on the balanced MAE; inner folds stratified by age bin."""
    pipeline, grid = build_pipeline(model_name, config["preprocessing"], config["cv"]["random_state"])
    inner = StratifiedKFold(config["cv"]["inner_splits"], shuffle=True, random_state=config["cv"]["random_state"])
    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring=make_scorer(balanced_mae, greater_is_better=False, bins=config["age_bins"]),
        cv=list(inner.split(X, age_strata(y, config["age_bins"]))),
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)
    return search


def n_features_used(search):
    steps = search.best_estimator_.named_steps
    return len(steps["select"].mirnas_) if "select" in steps else 0


def balanced_mae(y_true, y_pred, bins):
    """Mean of the per-age-bin MAEs, so every age bin counts equally regardless of its size."""
    errors = pd.Series(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
    groups = pd.cut(np.asarray(y_true), bins=bins, include_lowest=True)
    return float(errors.groupby(groups, observed=True).mean().mean())


def regression_metrics(y_true, y_pred, bins):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "n": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "balanced_MAE": balanced_mae(y_true, y_pred, bins),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "MedianAE": median_absolute_error(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 and np.std(y_pred) > 0 else np.nan,
    }
