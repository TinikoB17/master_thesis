import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Model comparison across the repeated nested CV folds and the leave-project-out test sets

models = snakemake.params.models

folds = pd.concat([pd.read_csv(path, sep="\t").assign(model=model)
                   for model, path in zip(models, snakemake.input.folds)], ignore_index=True)
metrics = ["MAE", "balanced_MAE", "RMSE", "R2", "MedianAE", "bias", "pearson_r"]
cv_table = folds.groupby("model")[metrics].agg(["mean", "std"]).reindex(models)
cv_table.columns = [f"{metric}_{stat}" for metric, stat in cv_table.columns]
cv_table = cv_table.sort_values("MAE_mean")
cv_table.to_csv(snakemake.output.cv_table, sep="\t")
print(cv_table.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, metric in zip(axes, ["MAE", "balanced_MAE", "R2"]):
    ax.boxplot([folds.loc[folds["model"] == model, metric] for model in models], tick_labels=models)
    ax.set_title(f"{metric} per outer fold")
    ax.tick_params(axis="x", labelrotation=30)
fig.suptitle("Repeated nested CV")
fig.tight_layout()
fig.savefig(snakemake.output.cv_plot)
plt.close(fig)

lopo = pd.concat([pd.read_csv(path, sep="\t") for path in snakemake.input.lopo], ignore_index=True)
lopo.to_csv(snakemake.output.lopo_table, sep="\t", index=False)

schemes = lopo["scheme"].unique().tolist()
fig, axes = plt.subplots(len(schemes), 2, figsize=(16, 4.5 * len(schemes)), squeeze=False)
width = 0.8 / len(models)
for row, scheme in enumerate(schemes):
    sub = lopo[lopo["scheme"] == scheme]
    test_sets = sub["test_set"].drop_duplicates().tolist()
    sizes = sub.drop_duplicates("test_set").set_index("test_set")["n"]
    x = np.arange(len(test_sets))
    for col, metric in enumerate(["MAE", "bias"]):
        ax = axes[row, col]
        for i, model in enumerate(models):
            values = sub[sub["model"] == model].set_index("test_set").reindex(test_sets)[metric]
            ax.bar(x + i * width, values, width, label=model)
        if metric == "MAE":
            baseline = sub[sub["model"] == models[0]].set_index("test_set").reindex(test_sets)["baseline_MAE"]
            ax.scatter(x + 0.4 - width / 2, baseline, marker="_", s=400, color="black", label="train-mean baseline")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels([f"{t}\n(n={sizes[t]})" for t in test_sets], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric if metric == "MAE" else "bias (predicted - actual)")
        ax.set_title(f"{scheme}: {metric}")
    axes[row, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(snakemake.output.lopo_plot)
plt.close(fig)
