import pandas as pd
from scipy.stats import spearmanr, pearsonr
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

expression = snakemake.input.rpmm_filtered_train
metadata_train = snakemake.input.metadata_train
changing_mirnas = snakemake.output.changing_mirnas
linearly_changing_mirnas = snakemake.output.linearly_changing_mirnas


exp = sc.read_h5ad(expression).to_df()
exp.index.name = "Sample"

md = pd.read_csv(metadata_train, sep='\t').set_index("Sample")

merged = exp.merge(md["Age"], how="right", left_index=True, right_index=True)


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

pearson_condition = (corr_df["Pearson_pval"] < 0.05) & (
    (corr_df["Pearson_corr"] <= -0.15) | (corr_df["Pearson_corr"] >= 0.15)
)
spearman_condition = (corr_df["Spearman_pval"] < 0.05) & (
    (corr_df["Spearman_Corr"] <= -0.15) | (corr_df["Spearman_Corr"] >= 0.15)
)

pearson_changing = corr_df[pearson_condition | spearman_condition]

print(pearson_changing, len(pearson_changing))

linearly_changing = corr_df[spearman_condition]
print(linearly_changing)



pearson_changing["miRNA"].to_csv(changing_mirnas, sep='\t', index=False)
linearly_changing["miRNA"].to_csv(linearly_changing_mirnas, sep='\t', index=False)
