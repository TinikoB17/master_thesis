from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Differential expression of miRNAs by age (continuous), Bonferroni-corrected, as in
# Huan et al. 2018 (Nat Commun 9:2015: 127 of the whole-blood miRNAs associated with age at
# P < 3.3e-4 in 5,221 adults).
# Input: prevalence-filtered log2(1 + RPMM) healthy blood samples (training set, or training
# and test set together; config.yaml, differential_expression_age). Every miRNA is fitted with
#   expression ~ age + covariates (Project, Sex)
# by ordinary least squares, so the age effect is estimated within projects. The age
# coefficient (log2 change per year) is tested with a t-test; a miRNA is significant if
# P < alpha / number of miRNAs tested (Bonferroni).

inputs = snakemake.input.expression
output_table = snakemake.output.table
output_volcano = snakemake.output.volcano
covariates = snakemake.params.covariates
alpha = snakemake.params.alpha
sample_label = snakemake.params.sample_label

adata = ad.concat([ad.read_h5ad(path) for path in inputs], join="inner")
expression = adata.to_df()
obs = adata.obs.copy()
age = obs["Age"].astype(float)
n_mirnas = expression.shape[1]
threshold = alpha / n_mirnas


# --- Design matrix: intercept, age (years), covariates ------------------------------------
design = pd.DataFrame({"intercept": 1.0, "age": age}, index=obs.index)
for covariate in covariates:
    dummies = pd.get_dummies(obs[covariate].astype(str), prefix=covariate, drop_first=True, dtype=float)
    design = design.join(dummies)
X = design.to_numpy()
Y = expression.to_numpy(dtype=float)
n, p = X.shape
if np.linalg.matrix_rank(X) < p:
    raise ValueError("Design matrix is rank deficient (age confounded with the covariates)")


# --- OLS for all miRNAs at once (same design for every miRNA) -------------------------------
XtX_inv = np.linalg.inv(X.T @ X)
beta = XtX_inv @ X.T @ Y
residuals = Y - X @ beta
df_resid = n - p
sigma2 = (residuals ** 2).sum(axis=0) / df_resid
age_idx = design.columns.get_loc("age")
slope = beta[age_idx]
se = np.sqrt(sigma2 * XtX_inv[age_idx, age_idx])
with np.errstate(divide="ignore", invalid="ignore"):
    t_stat = slope / se
pval = 2 * stats.t.sf(np.abs(t_stat), df_resid)

# Partial correlation of expression and age given the covariates (from the t statistic)
partial_r = t_stat / np.sqrt(t_stat ** 2 + df_resid)

table = pd.DataFrame({
    "miRNA": expression.columns,
    "log2FC_per_year": slope,
    "log2FC_per_decade": slope * 10,
    "se": se,
    "t": t_stat,
    "partial_r": partial_r,
    "pval": pval,
    "p_bonferroni": np.minimum(np.nan_to_num(pval, nan=1.0) * n_mirnas, 1.0),
    "mean_log2": expression.mean().values,
})
table["significant"] = table["pval"] < threshold
table["direction"] = np.where(~table["significant"], "",
                              np.where(table["log2FC_per_year"] > 0, "up with age", "down with age"))
table = table.sort_values("pval").reset_index(drop=True)
Path(output_table).parent.mkdir(parents=True, exist_ok=True)
table.to_csv(output_table, sep="\t", index=False)

significant = table[table["significant"]]
print(f"{n_mirnas} miRNAs, {n} samples ({sample_label}), age {age.min():.0f}-{age.max():.0f}, "
      f"covariates: {covariates}")
print(f"Bonferroni threshold: P < {alpha} / {n_mirnas} = {threshold:.2e}")
print(f"{len(significant)} miRNAs associated with age "
      f"({(significant['log2FC_per_year'] > 0).sum()} up, {(significant['log2FC_per_year'] < 0).sum()} down)")
print(significant.head(20)[["miRNA", "log2FC_per_decade", "partial_r", "pval", "p_bonferroni"]]
      .to_string(index=False))


# --- Volcano plot: change per decade vs -log10 p --------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.5))
colors = {"up with age": "#c0392b", "down with age": "#2e6da4", "": "lightgray"}
labels = {"up with age": "higher with age", "down with age": "lower with age", "": "not significant"}
for key in ["", "down with age", "up with age"]:
    d = table[table["direction"] == key]
    ax.scatter(d["log2FC_per_decade"], -np.log10(d["pval"]), s=12, alpha=0.75, color=colors[key],
               edgecolors="none", label=f"{labels[key]} ({len(d)})")
ax.axhline(-np.log10(threshold), color="grey", linestyle=":", linewidth=0.9)
ax.annotate(f"Bonferroni P = {threshold:.1e}", (ax.get_xlim()[0], -np.log10(threshold)),
            fontsize=8, color="grey", xytext=(4, 3), textcoords="offset points")
for _, row in significant.head(10).iterrows():
    ax.annotate(row["miRNA"].replace("hsa-", ""), (row["log2FC_per_decade"], -np.log10(row["pval"])),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axvline(0, color="grey", linewidth=0.6)
ax.set_xlabel("log2 fold change per decade of age (adjusted for " + ", ".join(covariates) + ")")
ax.set_ylabel("-log10 p-value")
ax.set_title(f"miRNA expression by age, healthy blood ({sample_label}, n = {n})\n"
             f"{len(significant)} of {n_mirnas} miRNAs significant (Bonferroni, alpha = {alpha})")
ax.legend(fontsize=8, loc="upper left", frameon=False)
fig.tight_layout()
Path(output_volcano).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_volcano)
plt.close(fig)
