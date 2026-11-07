import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Overview of the projects among healthy blood samples: number of samples per project
# and the oldest sample in each project. Projects are ordered by sample count.

metadata_path = snakemake.input.blood_healthy_metadata
counts_out = snakemake.output.project_counts
max_age_out = snakemake.output.project_max_age

md = pd.read_csv(metadata_path, sep="\t")
order = md["Project"].value_counts().index.tolist()

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)


def finish(ax, title, ylabel, path):
    ax.set_title(title)
    ax.set_xlabel("Project")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    sns.despine(ax=ax)
    ax.figure.tight_layout()
    ax.figure.savefig(path)
    plt.close(ax.figure)


fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=md, x="Project", order=order, color="tab:blue", ax=ax)
ax.bar_label(ax.containers[0], padding=2, fontsize=10)
finish(ax, f"Project Sample Counts - Healthy Blood (n = {len(md)})", "Number of samples", counts_out)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=md, x="Project", y="Age", order=order, estimator="max", errorbar=None, color="tab:blue", ax=ax)
ax.bar_label(ax.containers[0], fmt="%.0f", padding=2, fontsize=10)
finish(ax, "Oldest Sample per Project - Healthy Blood", "Maximum age (years)", max_age_out)
