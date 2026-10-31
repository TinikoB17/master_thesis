import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Cohort description: per-project counts, age and sex, plus an overview figure

obs = ad.read_h5ad(snakemake.input.cohort, backed="r").obs
bins = snakemake.config["age_bins"]

table = obs.groupby("Project", observed=True).agg(
    n=("Age", "size"),
    age_mean=("Age", "mean"),
    age_sd=("Age", "std"),
    age_min=("Age", "min"),
    age_max=("Age", "max"),
    female=("Sex", lambda s: (s == "f").sum()),
    male=("Sex", lambda s: (s == "m").sum()),
    sequencing=("Sequencing_Method", lambda s: ", ".join(sorted(s.astype(str).unique()))),
    disease_label=("Disease_Condition_original", lambda s: ", ".join(sorted(s.astype(str).unique()))),
).sort_values("n", ascending=False)
total = pd.DataFrame([{
    "n": len(obs), "age_mean": obs["Age"].mean(), "age_sd": obs["Age"].std(), "age_min": obs["Age"].min(),
    "age_max": obs["Age"].max(), "female": (obs["Sex"] == "f").sum(), "male": (obs["Sex"] == "m").sum(),
}], index=["total"])
table = pd.concat([table, total])
table.to_csv(snakemake.output.table, sep="\t", index_label="Project")
print(table.round(1).to_string())

projects = table.index[:-1].tolist()
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

axes[0].hist([obs.loc[obs["Project"] == p, "Age"] for p in projects], bins=range(bins[0], bins[-1] + 1, 5),
             stacked=True, label=projects)
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Samples")
axes[0].set_title("Age distribution (stacked by project)")
axes[0].legend(fontsize=7)

age_bin = pd.cut(obs["Age"], bins=bins, include_lowest=True)
counts = pd.crosstab(age_bin, obs["Sex"])
counts.plot.bar(ax=axes[1], rot=0)
axes[1].set_xlabel("Age bin")
axes[1].set_title("Sex by age bin")

axes[2].boxplot([obs.loc[obs["Project"] == p, "Age"] for p in projects],
                tick_labels=[f"{p}\n(n={table.loc[p, 'n']})" for p in projects])
axes[2].tick_params(axis="x", labelrotation=45, labelsize=7)
axes[2].set_ylabel("Age")
axes[2].set_title("Age by project")

fig.tight_layout()
fig.savefig(snakemake.output.figure)
plt.close(fig)
