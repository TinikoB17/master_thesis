import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Distribution of the Pearson and Spearman correlations of every miRNA with age
# (training set, from calculate_correlations.py), one histogram per method.
# Design follows src/histogram_correlations.py; the threshold lines use
# min_abs_corr from config.yaml.

correlations_path = snakemake.input.correlations
min_abs_corr = snakemake.params.min_abs_corr
max_pval = snakemake.params.max_pval

corrs = pd.read_csv(correlations_path, sep="\t")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

panels = [
    ("Pearson", "Pearson_corr", "Pearson_pval", snakemake.output.pearson),
    ("Spearman", "Spearman_Corr", "Spearman_pval", snakemake.output.spearman),
]

for name, corr_col, pval_col, output_path in panels:
    n_passing = ((corrs[pval_col] < max_pval) & (corrs[corr_col].abs() >= min_abs_corr)).sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(data=corrs, x=corr_col, kde=True, bins=np.arange(-1, 1.01, 0.02),
                 color="tab:blue", edgecolor="white", linewidth=0.5, ax=ax)
    ax.axvline(-min_abs_corr, color="purple", ls="--", lw=1.2)
    ax.axvline(min_abs_corr, color="purple", ls="--", lw=1.2, label=f"|r| = {min_abs_corr}")

    ax.set_xlim(-1, 1)
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_xlabel(f"{name} Correlation Coefficient with Age")
    ax.set_ylabel("Number of miRNAs")
    ax.set_title(f"{name} Correlation Distribution")
    ax.legend(loc="upper right", frameon=True)
    ax.text(0.02, 0.97,
            f"n = {len(corrs)} miRNAs\nmedian r = {corrs[corr_col].median():.3f}\n"
            f"|r| ≥ {min_abs_corr}, p < {max_pval}: {n_passing}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="lightgray"))
    sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
