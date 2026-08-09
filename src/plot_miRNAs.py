import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.stats import linregress
from statsmodels.nonparametric.smoothers_lowess import lowess

exp = pd.read_csv("test_output/rpmm_tissueatlas_blood_healthy.csv", sep='\t')
# eligible = pd.read_csv("correlations/highest_spearman.csv", sep='\t')["miRNA"].to_list()
exp = exp.set_index("miRNA").T
exp.index.name ="Sample"

expression_cols = exp.columns.to_list()

# metadata = pd.read_csv("PPMI_mod.csv", sep='\t').set_index("Sample")
metadata = pd.read_csv("test_output/blood_metadata_tissueatlas.csv", sep='\t').set_index("Sample")
metadata = metadata[(metadata["Disease_Condition"].isin(["Healthy"]))]

df = exp.merge(
    metadata,
    left_index=True,
    right_index=True,
    how='inner'
)


df[expression_cols] = np.log2(df[expression_cols] + 1)

output_dir = "gene_plots"
os.makedirs(output_dir, exist_ok=True)

for mir in expression_cols:
    if mir == 'hsa-miR-181a-5p':

        sns.set_theme(style="whitegrid", context="notebook")

        plt.figure(figsize=(6.5, 4.5))
        x = df["Age"]
        y = df[mir]
        mask = x.notna() & y.notna()

        r, p = pearsonr(x[mask], y[mask])

        ax = sns.regplot(
            data=df,
            x="Age",
            y=mir,
            lowess=True,
            scatter_kws={"s": 35, "alpha": 0.6, "color": "#2a6f97"},
            line_kws={"color": "#d1495b", "lw": 2.5}
        )

        # labels
        ax.set_title(f"{mir} vs Age", fontsize=13, pad=10)
        ax.set_xlabel("Age", fontsize=11)
        ax.set_ylabel("log2(RPMM + 1)", fontsize=11)

        # annotation (clean, journal style)
        ax.text(
            0.05, 0.95,
            f"Pearson r = {r:.3f}\np = {p:.2e}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.8
            )
        )

        # remove top/right spines (journal style)
        sns.despine()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{mir}.png", dpi=300, bbox_inches="tight")
        plt.close()
