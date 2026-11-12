"""Shared helpers of the age models: data loading, metrics, evaluation report and plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error

AGE_GROUP_WIDTH = 10  # years, for the per-age-group errors
MIN_GROUP_SIZE = 10  # smallest group used in the within-group metrics


COHORT_PREFIX = "Cohort_"  # 0/1 cohort columns (scripts/add_cohort_features.py)


class SelectKBestKeep(SelectKBest):
    """SelectKBest that always keeps the columns at the indices in keep (e.g. the cohort
    columns) and picks the k best of the other columns."""

    def __init__(self, score_func=f_regression, *, k=10, keep=()):
        super().__init__(score_func=score_func, k=k)
        self.keep = keep

    def fit(self, X, y=None, **params):
        super().fit(X, y, **params)
        self.scores_ = self.scores_.astype(float)
        self.scores_[list(self.keep)] = -np.inf  # not counted among the k best
        return self

    def _get_support_mask(self):
        mask = super()._get_support_mask()
        mask[list(self.keep)] = True
        return mask


def feature_selector(X):
    """SelectKBest(f_regression), keeping the cohort columns of X (if any) in every model."""
    keep = tuple(i for i, column in enumerate(X.columns) if column.startswith(COHORT_PREFIX))
    return SelectKBestKeep(score_func=f_regression, keep=keep) if keep else SelectKBest(score_func=f_regression)


def load(path):
    """Features (Sex as Male = 1, Female = 0, plus the miRNAs) and age from a train/holdout table."""
    data = pd.read_csv(path, sep="\t", index_col=0)
    X = data.drop(columns=["Age"])
    X["Sex"] = (X["Sex"] == "Male").astype(int)
    return X, data["Age"].astype(float)


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "pearson_r": np.corrcoef(y_true, y_pred)[0, 1],
    }


def errors_by_age_group(y_true, y_pred) -> pd.DataFrame:
    errors = pd.DataFrame({"age": y_true, "error": np.abs(y_true - y_pred)})
    start = int(y_true.min() // AGE_GROUP_WIDTH * AGE_GROUP_WIDTH)
    # Last edge above the oldest age, so an age on a bin edge (e.g. 70) still has a bin
    stop = (y_true.max() // AGE_GROUP_WIDTH + 1) * AGE_GROUP_WIDTH
    edges = np.arange(start, stop + AGE_GROUP_WIDTH, AGE_GROUP_WIDTH)
    errors["age_group"] = pd.cut(errors["age"], bins=edges, right=False)
    return (errors.groupby("age_group", observed=True)["error"]
            .agg(n="count", MAE="mean", MedianAE="median").reset_index())


def load_groups(metadata_path, samples) -> pd.Series:
    """Disease_State of the samples (e.g. which control group each one belongs to)."""
    return pd.read_csv(metadata_path, sep="\t", index_col=0).loc[samples, "Disease_State"]


def errors_by_group(y_true, y_pred, groups) -> pd.DataFrame:
    y_pred = pd.Series(np.asarray(y_pred), index=y_true.index)
    rows = []
    for group, index in y_true.groupby(groups).groups.items():
        rows.append({"group": group, "n": len(index), "mean_age": y_true[index].mean(),
                     "bias": (y_pred[index] - y_true[index]).mean(),
                     **metrics(y_true[index], y_pred[index])})
    return pd.DataFrame(rows)


def within_group_metrics(y_true, y_pred, groups, min_size=MIN_GROUP_SIZE) -> dict:
    """Age signal within groups, ignoring differences between them (groups with fewer than
    min_size samples are left out):
    - within_r: correlation of actual and predicted age after subtracting each group's mean
      from both
    - offset_corrected_MAE: MAE after removing each group's mean error (its bias)
    - group_mean_baseline_MAE: MAE of predicting each sample's group mean age; the
      offset-corrected MAE must be below it for the model to carry age signal within groups"""
    y_pred = pd.Series(np.asarray(y_pred), index=y_true.index)
    groups = pd.Series(np.asarray(groups), index=y_true.index)
    sizes = groups.map(groups.value_counts())
    keep = sizes >= min_size
    y_true, y_pred, groups = y_true[keep], y_pred[keep], groups[keep]
    true_centred = y_true - y_true.groupby(groups).transform("mean")
    pred_centred = y_pred - y_pred.groupby(groups).transform("mean")
    error = y_pred - y_true
    offset_corrected = error - error.groupby(groups).transform("mean")
    return {
        "n": int(keep.sum()),
        "groups": int(groups.nunique()),
        "within_r": np.corrcoef(true_centred, pred_centred)[0, 1],
        "offset_corrected_MAE": offset_corrected.abs().mean(),
        "group_mean_baseline_MAE": true_centred.abs().mean(),
    }


def write_evaluation(path, header, cv_results, y_train, y_cv, y_holdout, y_pred, groups=None,
                     train_groups=None):
    """Evaluation report: tuning results, CV and holdout metrics, baseline, errors by age group
    and, with groups (Disease_State of the holdout samples; train_groups for the CV
    predictions), within-group metrics and errors by group."""
    cv_scores, holdout_scores = metrics(y_train, y_cv), metrics(y_holdout, y_pred)
    baseline_mae = mean_absolute_error(y_holdout, np.full(len(y_holdout), y_train.median()))
    with open(path, "w") as file:
        file.write(header + "\n\n")
        file.write(cv_results.to_string(index=False) + "\n\n")
        file.write("CV (training set): " + ", ".join(f"{k} = {v:.3f}" for k, v in cv_scores.items()) + "\n")
        file.write("Holdout: " + ", ".join(f"{k} = {v:.3f}" for k, v in holdout_scores.items()) + "\n")
        file.write(f"Holdout MAE of always predicting the training median age: {baseline_mae:.3f}\n\n")
        if groups is not None and groups.nunique() > 1:
            file.write(f"Within-group evaluation (groups with >= {MIN_GROUP_SIZE} samples; "
                       "see model_utils.within_group_metrics):\n")
            sets = [("Holdout", y_holdout, y_pred, groups)]
            if train_groups is not None:
                sets.insert(0, ("CV (training set)", y_train, y_cv, train_groups))
            for name, y_true, predicted, labels in sets:
                scores = within_group_metrics(y_true, predicted, labels)
                file.write(f"{name}: " + ", ".join(
                    f"{k} = {v:.3f}" if isinstance(v, float) else f"{k} = {v}"
                    for k, v in scores.items()) + "\n")
            file.write("\n")
        file.write("Holdout errors by age group:\n"
                   + errors_by_age_group(y_holdout, y_pred).to_string(index=False) + "\n")
        if groups is not None and groups.nunique() > 1:
            file.write("\nHoldout errors by group (bias = mean predicted - actual age):\n"
                       + errors_by_group(y_holdout, y_pred, groups).round(3).to_string(index=False) + "\n")
    with open(path) as file:
        print(file.read())


def write_predictions(path, y_holdout, y_pred):
    pd.DataFrame({"actual_age": y_holdout, "predicted_age": y_pred,
                  "age_gap": y_pred - y_holdout}).to_csv(path, sep="\t")


def plot_predictions(y_true, y_pred, title, path):
    """Predicted vs actual age with the identity line."""
    y_pred = pd.Series(np.asarray(y_pred), index=y_true.index)
    scores = metrics(y_true, y_pred)
    low = min(y_true.min(), y_pred.min()) - 2
    high = max(y_true.max(), y_pred.max()) + 2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=18, color="steelblue", edgecolors="none")
    ax.plot([low, high], [low, high], linestyle="--", color="darkorange", label="Predicted = actual")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Actual age (years)")
    ax.set_ylabel("Predicted age (years)")
    ax.set_title(f"{title}\nMAE = {scores['MAE']:.2f}, R² = {scores['R2']:.2f}, "
                 f"r = {scores['pearson_r']:.2f}, n = {len(y_true)}")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
