"""Applies an age model trained on the controls to the cancer samples (none were used in training).

z-scored models: the cancer samples have no training stats of their own, so each miRNA is
z-scored within each cancer type (Disease_State) with that type's own mean and SD (one batch,
like each control group). This removes any shift of a whole cancer type, including a real age
or disease offset, so only the within-type age signal (within_r, offset-corrected MAE vs the
group-mean baseline) is meaningful, not the bias.
Other models get the log2 values as deposited.
Cancer types with fewer than MIN_GROUP_SIZE samples are left out (too few to z-score)."""
import joblib
import numpy as np
import pandas as pd

from model_utils import (MIN_GROUP_SIZE, errors_by_age_group, errors_by_group, metrics,
                         plot_predictions, within_group_metrics, write_predictions)

MODEL = snakemake.input.model
EXPRESSION = snakemake.input.expression
METADATA = snakemake.input.metadata
ZSCORE = snakemake.params.zscore
TRAIN_AGE_RANGE = snakemake.params.train_age_range  # (min, max) of the training ages, or None
LABEL = snakemake.params.label


def zscore_within(X, groups, mirnas):
    X = X.copy()
    grouped = X[mirnas].groupby(groups)
    std = grouped.transform("std")
    X[mirnas] = ((X[mirnas] - grouped.transform("mean")) / std.mask(std == 0, 1.0)).fillna(0)
    return X


def prepare(expression, metadata, features):
    X = metadata[["Sex"]].join(expression)
    X["Sex"] = (X["Sex"] == "Male").astype(int)
    if ZSCORE:
        mirnas = [feature for feature in features if feature != "Sex"]
        X = zscore_within(X, metadata["Disease_State"], mirnas)
    return X[features]


def describe(name, y_true, y_pred, groups):
    scores = metrics(y_true, y_pred)
    gap = y_pred - y_true
    text = (f"{name} (n = {len(y_true)}, age {y_true.min():.0f}-{y_true.max():.0f}, "
            f"mean {y_true.mean():.1f}):\n  "
            + ", ".join(f"{k} = {v:.3f}" for k, v in scores.items())
            + f"\n  bias (mean predicted - actual) = {gap.mean():.3f}, "
              f"mean predicted age = {y_pred.mean():.2f}, SD of predictions = {y_pred.std():.2f}"
              f" (actual SD {y_true.std():.2f})\n")
    within = within_group_metrics(y_true, y_pred, groups)
    text += "  within cancer types: " + ", ".join(
        f"{k} = {v:.3f}" if isinstance(v, float) else f"{k} = {v}"
        for k, v in within.items()) + "\n"
    return text


def main():
    model = joblib.load(MODEL)
    expression = pd.read_csv(EXPRESSION, sep="\t", index_col=0)
    metadata = pd.read_csv(METADATA, sep="\t", index_col=0).loc[expression.index]
    sizes = metadata["Disease_State"].value_counts()
    small = sizes[sizes < MIN_GROUP_SIZE]
    if not small.empty:
        print(f"Left out (fewer than {MIN_GROUP_SIZE} samples): {small.to_dict()}")
    metadata = metadata[~metadata["Disease_State"].isin(small.index)]
    expression = expression.loc[metadata.index]

    X = prepare(expression, metadata, list(model.feature_names_in_))
    y = metadata["Age"].astype(float)
    groups = metadata["Disease_State"]
    y_pred = pd.Series(model.predict(X), index=y.index)
    write_predictions(snakemake.output.predictions, y, y_pred)

    with open(snakemake.output.evaluation, "w") as file:
        file.write(f"{LABEL}\nPreprocessing: "
                   + ("miRNAs z-scored within each cancer type (own mean and SD)\n" if ZSCORE
                      else "log2 values as deposited\n"))
        if not small.empty:
            file.write(f"Left out (fewer than {MIN_GROUP_SIZE} samples): {small.to_dict()}\n")
        file.write("\n" + describe("All cancer samples", y, y_pred, groups))
        if TRAIN_AGE_RANGE is not None:
            inside = y.between(*TRAIN_AGE_RANGE)
            file.write(describe(f"Within the training age range {TRAIN_AGE_RANGE}",
                                y[inside], y_pred[inside], groups[inside]))
        for sex in ["Male", "Female"]:
            keep = metadata["Sex"] == sex
            file.write(describe(sex, y[keep], y_pred[keep], groups[keep]))
        file.write("\nErrors by age group:\n" + errors_by_age_group(y, y_pred).to_string(index=False)
                   + "\n")
        table = errors_by_group(y, y_pred, groups)
        table["within_baseline_MAE"] = [np.abs(y[groups == g] - y[groups == g].mean()).mean()
                                        for g in table["group"]]
        file.write("\nErrors by cancer type (bias = mean predicted - actual age; "
                   "within_baseline_MAE = MAE of predicting the type's mean age):\n"
                   + table.sort_values("n", ascending=False).round(3).to_string(index=False) + "\n")
    with open(snakemake.output.evaluation) as file:
        print(file.read())
    plot_predictions(y, y_pred, LABEL, snakemake.output.plot)


if __name__ == "__main__":
    main()
