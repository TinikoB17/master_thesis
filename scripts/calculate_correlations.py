import pandas as pd
from scipy.stats import spearmanr, pearsonr
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
expression = snakemake.input.blood_h5ad_healthy
metadata_train = snakemake.input.metadata_train
changing_mirnas = snakemake.output.changing_mirnas
cluster_hist = snakemake.output.cluster_hist

exp = sc.read_h5ad(expression).to_df()
exp.index.name = "Sample"
print(exp)

md = pd.read_csv(metadata_train, sep='\t').set_index("Sample")

merged = exp.merge(md["Age"], how="right", left_index=True, right_index=True)


results = []

X = merged.drop(columns='Age')
y = merged['Age']

for mirna in X.columns:
    pearson_corr, pearson_p = pearsonr(X[mirna], y)
    spearman_corr, spearman_p = spearmanr(X[mirna], y)

    results.append([mirna, pearson_corr, spearman_corr])

corr_df = pd.DataFrame(results, columns=['miRNA', "Pearson_corr", "Spearman_Corr"])
corr_df.to_csv("test_output/correlations.csv", sep='\t', index=False)
conditions =  [corr_df["Spearman_Corr"] < -0.2,
    (corr_df["Spearman_Corr"] < -0.1) & (corr_df["Spearman_Corr"] > -0.2),
    (corr_df["Spearman_Corr"] > -0.1) & (corr_df["Spearman_Corr"] < 0.1),
    (corr_df["Spearman_Corr"] > 0.1) & (corr_df["Spearman_Corr"] < 0.2),
    corr_df["Spearman_Corr"] > 0.2]

values = [1, 2, 3, 4, 5]
corr_df["Cluster_TB"] = np.select(conditions, values)

print(corr_df["Spearman_Corr"].max(), corr_df["Spearman_Corr"].min(), corr_df["Pearson_corr"].max(), corr_df["Pearson_corr"].min())

corr_df = corr_df.dropna()

pearson_changing = corr_df[(corr_df["Pearson_corr"] < -0.1) | (corr_df["Pearson_corr"] > 0.1) | (corr_df["Spearman_Corr"] > 0.1) | (corr_df["Spearman_Corr"] < -0.1)]

fig, ax = plt.subplots()
counts = corr_df["Cluster_TB"].value_counts()
ax.bar(counts.index, counts.values,  color="#4DD1A9")
plt.xticks([1, 2, 3, 4, 5])
ax.set_title("Number of rpmm-filtered miRNAs in each cluster", fontdict={"fontsize": 14, "fontweight": "bold"})
ax.set_xlabel("Clusters")
ax.set_ylabel("Number of miRNAs in a cluster")

fig.savefig(cluster_hist)

print(corr_df)
# mirnas_change = corr_df[corr_df["Cluster_TB"].isin([1, 2, 4, 5])]
# mirnas_change = corr_df[corr_df["Cluster_TB"].isin([1, 2, 4, 5])]
# print(mirnas_change["miRNA"])
pearson_changing["miRNA"].to_csv(changing_mirnas, sep='\t', index=False)
# mirnas_change["miRNA"].to_csv("correlations/highest_spearman.csv", sep='\t')

# cluster_members_tob = pd.read_csv("test_output/cluster_members.csv", sep=';')

# merged = corr_df.merge(cluster_members_tob, how="inner", left_on="miRNA", right_on="miRNA")
