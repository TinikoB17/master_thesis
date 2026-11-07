import pandas as pd
from scipy.stats import spearmanr, pearsonr
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

expression = snakemake.input.rpmm_filtered_train
changing_mirnas = snakemake.output.changing_mirnas
all_correlations = snakemake.output.correlations

# Correlation thresholds come from config.yaml (see rule calculate_corr)
min_abs_corr = snakemake.params.min_abs_corr
max_pval = snakemake.params.max_pval


adata = sc.read_h5ad(expression)
exp = adata.to_df()
exp.index.name = "Sample"

# Age from the h5ad's own sample metadata (same training samples)
merged = exp.join(adata.obs["Age"].astype(float))


results = []

X = merged.drop(columns='Age')
y = merged['Age']

for mirna in X.columns:
    pearson_corr, pearson_p = pearsonr(X[mirna], y)
    spearman_corr, spearman_p = spearmanr(X[mirna], y)

    results.append([mirna, pearson_corr, pearson_p, spearman_corr, spearman_p])

corr_df = pd.DataFrame(results, columns=['miRNA', "Pearson_corr", "Pearson_pval", "Spearman_Corr", "Spearman_pval"])

corr_df["Spearman_Corr"] = pd.to_numeric(corr_df["Spearman_Corr"], errors="coerce")

print(corr_df["Spearman_Corr"].max(), corr_df["Spearman_Corr"].min(), corr_df["Pearson_corr"].max(), corr_df["Pearson_corr"].min())

corr_df = corr_df.dropna()

pearson_condition = (corr_df["Pearson_pval"] < max_pval) & (
    (corr_df["Pearson_corr"] <= -min_abs_corr) | (corr_df["Pearson_corr"] >= min_abs_corr)
)
spearman_condition = (corr_df["Spearman_pval"] < max_pval) & (
    (corr_df["Spearman_Corr"] <= -min_abs_corr) | (corr_df["Spearman_Corr"] >= min_abs_corr)
)

pearson_changing = corr_df[pearson_condition | spearman_condition]

print(pearson_changing, len(pearson_changing))




corr_df.to_csv(all_correlations, sep='\t', index=False)
pearson_changing["miRNA"].to_csv(changing_mirnas, sep='\t', index=False)
