from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Expression of individual miRNAs against age, one figure per miRNA with a female and a male
# panel, points coloured by project, with a linear fit and the Pearson/Spearman correlation. Settings come from
# config.yaml (expression_plots): the h5ad file, the projects to use (empty = all), and the
# miRNAs. With an empty miRNA list, top_n > 0 plots the top_n miRNAs most correlated with age
# (|Spearman rho| on the selected samples) and top_n = 0 plots every miRNA.

h5ad_path = snakemake.input.h5ad
output_dir = Path(snakemake.output.plot_dir)
projects_to_keep = snakemake.params.projects
requested = snakemake.params.mirnas
top_n = snakemake.params.top_n
log2 = snakemake.params.log2

adata = sc.read_h5ad(h5ad_path)
if projects_to_keep:
    adata = adata[adata.obs["Project"].astype(str).isin(projects_to_keep)].copy()
    if adata.n_obs == 0:
        raise ValueError(f"No samples from projects {projects_to_keep} in {h5ad_path}")
print(f"{adata.n_obs} samples from {adata.obs['Project'].astype(str).unique().tolist()}")

age = adata.obs["Age"].astype(float).to_numpy()
output_dir.mkdir(parents=True, exist_ok=True)

if requested:
    missing = [mirna for mirna in requested if mirna not in adata.var_names]
    if missing:
        print(f"Warning: miRNAs not in {h5ad_path}, skipped: {missing}")
    mirnas = [mirna for mirna in requested if mirna in adata.var_names]
    if not mirnas:
        raise ValueError(f"None of the configured miRNAs are in {h5ad_path}")
elif top_n > 0:
    # Spearman is rank-based, so the ranking is the same with or without the log2 transform
    all_expression = adata.to_df()
    ranking = []
    for mirna in all_expression.columns:
        values = all_expression[mirna].to_numpy()
        if np.ptp(values) == 0:
            continue
        rho, pval = spearmanr(age, values)
        ranking.append((mirna, rho, pval))
    ranking = pd.DataFrame(ranking, columns=["miRNA", "Spearman_rho", "Spearman_pval"])
    ranking = ranking.reindex(ranking["Spearman_rho"].abs().sort_values(ascending=False).index)
    ranking.to_csv(output_dir / "correlation_ranking.tsv", sep="\t", index=False)
    mirnas = ranking["miRNA"].head(top_n).tolist()
    print(ranking.head(top_n).round(4).to_string(index=False))
else:
    mirnas = adata.var_names.tolist()
print(f"Plotting {len(mirnas)} miRNAs from {h5ad_path}")

expression = adata[:, mirnas].to_df()
if log2:
    expression = np.log2(1 + expression)
projects = adata.obs["Project"].astype(str).to_numpy()
sex = adata.obs["Sex"].astype(str).to_numpy()
y_label = "log2(1 + expression)" if log2 else "Expression"
# One panel per sex, female left and male right
SEXES = [("f", "Female"), ("m", "Male")]

project_order = adata.obs["Project"].astype(str).value_counts().index.tolist()
palette = dict(zip(project_order, sns.color_palette("tab10", len(project_order))))

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def plot_panel(ax, age, values, projects, title):
    for project in project_order[::-1]:  # largest project drawn first, small ones stay visible
        mask = projects == project
        if mask.any():
            ax.scatter(age[mask], values[mask], s=12, alpha=0.5, color=palette[project])

    if len(values) > 2 and np.ptp(values) > 0 and np.ptp(age) > 0:
        slope, intercept = np.polyfit(age, values, 1)
        ages = np.array([age.min(), age.max()])
        ax.plot(ages, intercept + slope * ages, color="black", linewidth=1.5)
        pearson_r, pearson_p = pearsonr(age, values)
        spearman_r, spearman_p = spearmanr(age, values)
        stats = (f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.1e})\n"
                 f"Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.1e})")
    elif len(values) == 0:
        stats = "no samples"
    else:
        stats = "too few samples or constant expression"
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="lightgray"))
    ax.set_title(f"{title} (n = {len(values)})")
    ax.set_xlabel("Age (years)")
    sns.despine(ax=ax)


# Legend: projects with their sample counts (both sexes), largest first, then the fit
legend_handles = [plt.Line2D([], [], marker="o", linestyle="", alpha=0.5, color=palette[project])
                  for project in project_order]
legend_labels = [f"{project} (n={(projects == project).sum()})" for project in project_order]
legend_handles.append(plt.Line2D([], [], color="black", linewidth=1.5))
legend_labels.append("linear fit")

for mirna in mirnas:
    values = expression[mirna].to_numpy()

    fig, axes = plt.subplots(1, len(SEXES), figsize=(14, 5), sharex=True, sharey=True)
    for ax, (code, name) in zip(axes, SEXES):
        mask = sex == code
        plot_panel(ax, age[mask], values[mask], projects[mask], name)
    axes[0].set_ylabel(y_label)

    fig.legend(legend_handles, legend_labels, fontsize=8, markerscale=1.5,
               loc="upper left", bbox_to_anchor=(0.86, 0.9), frameon=False)
    fig.suptitle(f"{mirna} (n = {np.isin(sex, [code for code, _ in SEXES]).sum()} samples)")
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(output_dir / f"{mirna}.svg")
    plt.close(fig)
