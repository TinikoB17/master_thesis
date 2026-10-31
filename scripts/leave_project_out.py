import numpy as np
import pandas as pd
from sklearn.utils.parallel import Parallel, delayed
from threadpoolctl import threadpool_limits

from age_models import fit_search, load_cohort, n_features_used, regression_metrics
from plots import plot_predictions

# Leave-project-out evaluation (tuning still by inner CV on the training projects).
#   ppmi_vs_rest: train on PPMI, test on all other projects
#   each_project: hold out each project in turn, train on the rest
# Held-out projects are z-scored with their own mean/sd (no age labels), see ProjectZScore.

config = snakemake.config
model_name = snakemake.wildcards.model
scheme = snakemake.wildcards.scheme
bins = config["age_bins"]
min_size = config["min_project_size"]


def run_split(held_out, train_mask, X, y, n_jobs):
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    with threadpool_limits(limits=1 if n_jobs == 1 else None):
        search = fit_search(model_name, config, X_train, y_train, n_jobs=n_jobs)
        y_pred = search.predict(X_test)
    predictions = pd.DataFrame({
        "Sample": X_test.index, "held_out": held_out, "Project": X_test["Project"].values,
        "actual_age": y_test.values, "predicted_age": y_pred, "train_mean_age": y_train.mean(),
    })
    info = {"n_train": int(train_mask.sum()), "n_mirnas_used": n_features_used(search),
            "best_params": str(search.best_params_)}
    return predictions, info


def metrics_row(name, predictions, info):
    row = {"scheme": scheme, "model": model_name, "test_set": name}
    row.update(regression_metrics(predictions["actual_age"], predictions["predicted_age"], bins))
    row["baseline_MAE"] = float(np.mean(np.abs(predictions["actual_age"] - predictions["train_mean_age"])))
    row["mean_actual_age"] = predictions["actual_age"].mean()
    row.update(info)
    return row


X, y = load_cohort(snakemake.input.cohort)
projects = X["Project"]

rows = []
if scheme == "ppmi_vs_rest":
    predictions, info = run_split("non-PPMI", (projects == "PPMI").to_numpy(), X, y, snakemake.threads)
    rows.append(metrics_row("non-PPMI pooled", predictions, info))
    for project, group in predictions.groupby("Project"):
        if len(group) >= min_size:
            rows.append(metrics_row(project, group, info))
elif scheme == "each_project":
    held_out_projects = projects.value_counts().index.tolist()
    results = Parallel(n_jobs=min(snakemake.threads, len(held_out_projects)), verbose=5)(
        delayed(run_split)(project, (projects != project).to_numpy(), X, y, 1)
        for project in held_out_projects
    )
    for project, (group, info) in zip(held_out_projects, results):
        rows.append(metrics_row(project, group, info))
    predictions = pd.concat([group for group, _ in results], ignore_index=True)
    rows.append(metrics_row("all held-out pooled", predictions, {"n_train": np.nan}))
else:
    raise ValueError(f"Unknown scheme: {scheme}")

metrics = pd.DataFrame(rows)
print(metrics.drop(columns=["best_params"], errors="ignore").to_string())
metrics.to_csv(snakemake.output.metrics, sep="\t", index=False)
predictions.to_csv(snakemake.output.predictions, sep="\t", index=False)
plot_predictions(predictions, bins, f"{model_name}: leave-project-out ({scheme})", snakemake.output.plot)
