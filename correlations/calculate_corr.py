import pandas as pd
from scipy.stats import spearmanr, pearsonr
import dcor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

exp = pd.read_csv("test_output/rpmm_tissueatlas_blood_healthy.csv", sep='\t').set_index("miRNA").T
exp.index.name = "Sample"
exp.columns.name = None
md = pd.read_csv("test_output/blood_metadata_tissueatlas.csv", sep='\t')
md = md[(md["Disease_Condition"].isin(["Healthy"]))].set_index("Sample")
merged = exp.merge(md["Age"], how="right", left_index=True, right_index=True)

results = []

X = merged.drop(columns='Age')
y = merged['Age']

for mirna in X.columns:
    pearson_corr, pearson_p = pearsonr(X[mirna], merged["Age"])
    spearman_corr, spearman_p = spearmanr(X[mirna], y)
    results.append([mirna, dcor.distance_correlation(X[mirna], y, method="naive"), pearson_corr, spearman_corr])

corr_df = pd.DataFrame(results, columns=['miRNA', 'DC', "Pearson_corr", "Spearman_Corr"])

conditions =  [corr_df["Spearman_Corr"] < -0.2,
    (corr_df["Spearman_Corr"] < -0.1) & (corr_df["Spearman_Corr"] > -0.2),
    (corr_df["Spearman_Corr"] > -0.1) & (corr_df["Spearman_Corr"] < 0.1),
    (corr_df["Spearman_Corr"] > 0.1) & (corr_df["Spearman_Corr"] < 0.2),
    corr_df["Spearman_Corr"] > 0.2]

values = [1, 2, 3, 4, 5]
corr_df["Cluster_TB"] = np.select(conditions, values)

corr_df = corr_df.dropna()

mirnas_change = corr_df[corr_df["Cluster_TB"].isin([1, 2, 4, 5])]
mirnas_change["miRNA"].to_csv("correlations/highest_spearman.csv", sep='\t', index=False)

cluster_members_tob = pd.read_csv("test_output/cluster_members.csv", sep=';')

merged = corr_df.merge(cluster_members_tob, how="inner", left_on="miRNA", right_on="miRNA")

fig, ax = plt.subplots()
counts = merged["Cluster_TB"].value_counts()
ax.bar(counts.index, counts.values,  color="#4DD1A9")
ax.set_title("Number of miRNAs in each cluster", fontdict={"fontsize": 14, "fontweight": "bold"})
ax.set_xlabel("Clusters")
ax.set_ylabel("Number of rpmm-filtered miRNAs in a cluster")

fig.savefig("correlations/cluster_miRNAs.svg")

breakpoint()

pivot = merged.pivot_table(
    index="Cluster_TB",
    columns="Cluster",
    aggfunc="size",
    fill_value=0
)

hm = sns.heatmap(pivot, annot=True, fmt="d")
plt.title("Cluster assignment comparison")
plt.ylabel("Cluster assigned by TB")
plt.xlabel("Cluster assigned by TF")
fig = hm.get_figure()
fig.savefig("correlations/cluster_corr.svg")
plt.close(fig)
dc_corr = pd.read_csv("test_output/non-linear-dc.csv", sep=';')

merged_dc = corr_df.merge(dc_corr, how="inner", left_on="miRNA", right_on="miRNA")

sc = sns.scatterplot(merged_dc, x="DC", y="non-linear DC")
plt.title("Distance Correlation ")
plt.ylabel("Distance Correlation - TF")
plt.xlabel("Distance Correlation - TB")
f = sc.get_figure()
f.savefig("correlations/distance_correlations.svg")
plt.close(f)

sc_pearson = sns.scatterplot(merged_dc, x="Pearson_corr", y="linear PC")
plt.title("Pearson Correlation ")
plt.ylabel("Pearson Correlation - TF")
plt.xlabel("Pearson Correlation - TB")
pears = sc_pearson.get_figure()
pears.savefig("correlations/pearson_correlations.svg")
plt.close(pears)

pearson_vs_distance_TB = sns.scatterplot(merged_dc, x="Pearson_corr", y="DC")
plt.title("Pearson Correlation vs Distance Correlation ")
plt.ylabel("Pearson Correlation")
plt.xlabel("Distance Correlation")
p_vs_dist = pearson_vs_distance_TB.get_figure()
p_vs_dist.savefig("correlations/pearson_correlations_vs_dist_TB.svg")
plt.close(p_vs_dist)

pearson_vs_distance_TF = sns.scatterplot(merged_dc, x="linear PC", y="non-linear DC")
plt.title("Pearson Correlation vs Distance Correlation ")
plt.ylabel("Pearson Correlation")
plt.xlabel("Distance Correlation")
p_vs_dist_tobias = pearson_vs_distance_TF.get_figure()
p_vs_dist_tobias.savefig("correlations/pearson_correlations_vs_dist_TF.svg")
plt.close(p_vs_dist_tobias)


breakpoint()

