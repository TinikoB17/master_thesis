import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

# Variance partitioning per miRNA with a linear mixed model:
#   log2(1 + RPMM) ~ Age + Sex + (1 | Project)
# Fixed-effect variance = variance of that term's contribution to the fitted values,
# Project variance = random intercept variance, Residual = model scale.

INPUT_H5AD = snakemake.input.rpmm_filtered_train
OUTPUT_VARIANCE = snakemake.output.variance
OUTPUT_PLOT = snakemake.output.lmm_plot
N_JOBS = snakemake.threads
# Optional age window (inclusive); without it all training samples are used
MIN_AGE = float(snakemake.wildcards.get("min_age", "-inf"))
MAX_AGE = float(snakemake.wildcards.get("max_age", "inf"))

FORMULA = "Expression ~ Age + C(Sex)"
FIXED_TERMS = {"Age": "Age", "Sex": "C(Sex)"}
# lbfgs is fast but on some miRNAs it diverges to a non-finite likelihood with the
# project variance stuck at 0 while still reporting convergence; the derivative-free
# optimizers find the proper optimum there.
OPTIMIZERS = ["lbfgs", "powell", "nm"]


def fit_lmm(df: pd.DataFrame):
    model = smf.mixedlm(FORMULA, df, groups=df["Project"])
    for method in OPTIMIZERS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = model.fit(method=method, reml=True)
            except (np.linalg.LinAlgError, ValueError):
                continue
        if result.converged and np.isfinite(result.llf):
            return result, method
    return None, None


def term_variance(result, term: str) -> float:
    """Variance of the fitted contribution of one fixed-effect term (all its columns)."""
    exog = pd.DataFrame(result.model.exog, columns=result.model.exog_names)
    columns = [col for col in exog.columns if col == term or col.startswith(f"{term}[")]
    return float((exog[columns] @ result.fe_params[columns]).var())


def lmm_mirna(mirna: str, expression: np.ndarray, metadata: pd.DataFrame):
    df = metadata.assign(Expression=expression)
    if df["Expression"].var() == 0:
        return None

    result, method = fit_lmm(df)
    if result is None:
        print(f"LMM did not converge for {mirna}")
        return None

    variances = {name: term_variance(result, term) for name, term in FIXED_TERMS.items()}
    variances["Project"] = float(result.cov_re.iloc[0, 0])
    variances["Residual"] = float(result.scale)
    total = sum(variances.values())

    return {
        "miRNA": mirna,
        **{f"Var_{name}": value / total * 100 for name, value in variances.items()},
        "Age_Coef": result.fe_params["Age"],
        "Age_pval": result.pvalues["Age"],
        "Optimizer": method,
    }


def create_boxplot(variance_df: pd.DataFrame):
    var_cols = [col for col in variance_df.columns if col.startswith("Var_")]
    df_melt = variance_df.melt(value_vars=var_cols, var_name="Variable", value_name="Variance_Percentage")
    df_melt["Variable"] = df_melt["Variable"].str.replace("Var_", "")

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df_melt, x="Variable", y="Variance_Percentage", color="white",
                width=0.5, fliersize=0, zorder=5)
    sns.stripplot(data=df_melt, x="Variable", y="Variance_Percentage", hue="Variable",
                  palette="Set2", legend=False, alpha=0.4, jitter=0.25, size=4, zorder=0)

    age_range = "" if np.isinf([MIN_AGE, MAX_AGE]).all() else f" (age {MIN_AGE:g}–{MAX_AGE:g})"
    plt.title(f"Variance Explained by LMM Variables Across miRNAs{age_range}", fontsize=14, pad=15)
    plt.ylabel("Variance Explained (%)", fontsize=12)
    plt.xlabel("Variables", fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, format="svg", bbox_inches="tight")
    plt.close()


def main():
    adata = sc.read_h5ad(INPUT_H5AD)
    age = adata.obs["Age"].astype(float)
    adata = adata[(age >= MIN_AGE) & (age <= MAX_AGE)].copy()
    metadata = adata.obs[["Age", "Sex", "Project"]].astype({"Age": float, "Sex": str, "Project": str})
    expression = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    mirna_names = adata.var_names.tolist()

    print(f"Fitting {len(mirna_names)} miRNAs on {adata.n_obs} samples aged "
          f"{metadata['Age'].min():.0f}-{metadata['Age'].max():.0f} from {metadata['Project'].nunique()} projects")
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(lmm_mirna)(mirna, expression[:, i], metadata) for i, mirna in enumerate(mirna_names)
    )

    results_df = pd.DataFrame([r for r in results if r is not None]).set_index("miRNA")
    results_df.insert(results_df.columns.get_loc("Age_pval") + 1, "Age_padj",
                      multipletests(results_df["Age_pval"], method="fdr_bh")[1])
    results_df = results_df.sort_values(by="Var_Age", ascending=False)

    print(results_df["Optimizer"].value_counts())
    print(results_df.describe())
    results_df.to_csv(OUTPUT_VARIANCE)

    create_boxplot(results_df)


main()
