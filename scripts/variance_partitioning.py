import pandas as pd
import numpy as np
import scanpy as sc
import statsmodels.formula.api as smf
import warnings
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt


INPUT_H5AD = "test_output/blood_healthy_rpmm_filt_train.h5ad"
INPUT_MD = "test_output/metadata_train.csv"
OUTPUT_VARIANCE = "variance.csv"


def lmm_mirna(mirna, exp, metadata):
    df = metadata.copy()
    df["Expression"] = exp

    df = df.dropna(subset=["Expression", "Age", "Sex", "Project"])

    if df["Expression"].var() == 0:
        return None

    model = smf.mixedlm("Expression ~ Age + C(Sex)", df, groups=df["Project"])
    result = model.fit(method="lbfgs", disp=False)

    var_project = result.cov_re.iloc[0, 0]

    var_residual = result.scale

    age_eff = df["Age"] * result.params["Age"]

    sex_coef = [e for e in result.params.index if "Sex" in e][0]
    sex_effect = (df["Sex"] == sex_coef.split("[T.")[1].strip("]")).astype(int) * result.params[sex_coef]

    var_age = age_eff.var()
    var_sex = sex_effect.var()

    total_var = var_age + var_sex + var_project + var_residual

    return {
        "miRNA": mirna,
        "Var_Age": (var_age / total_var) * 100,
        "Var_Sex": (var_sex / total_var) * 100,
        "Var_Project": (var_project / total_var) * 100,
        "Var_Residual": (var_residual / total_var) * 100,
        "Age_Coef": result.params["Age"],
        "Age_pval": result.pvalues["Age"]
    }

def create_boxplot(variance_df):
    df_melt = variance_df.melt(value_vars=["Var_Age", "Var_Sex", "Var_Project", "Var_Residual"], var_name="Variable", value_name="Variance_Percentage")

    df_melt["Variable"] = df_melt["Variable"].str.replace("Var_", "")

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df_melt,
        x="Variable",
        y="Variance_Percentage",
        color="white",
        width=0.5,
        fliersize=0,
        zorder=5
    )

    sns.stripplot(
        data=df_melt,
        x="Variable",
        y="Variance_Percentage",
        alpha=0.4,
        jitter=0.25,
        size=4,
        palette="Set2",
        zorder=0
    )

    plt.title("Variance Explained by LMM Variables Across miRNAs", fontsize=14, pad=15)
    plt.ylabel("Variance Explained (%)", fontsize=12)
    plt.xlabel("Variables", fontsize=12)

    plt.ylim(-5, 105)

    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    plt.savefig("figures/lmm.svg", format="svg", bbox_inches="tight")

def main():
    adata = sc.read_h5ad(INPUT_H5AD)
    meta = pd.read_csv(INPUT_MD, sep='\t', index_col=0)
    print(adata.X)
    print(meta)
    
    common_samples = adata.obs_names.intersection(meta.index)
    adata = adata[common_samples]
    meta = meta.loc[common_samples]
    print(meta)
    exp = adata.X.toarray()

    mirna_names = adata.var_names.tolist()

    print(f"Running LMMs for {len(mirna_names)} miRNAs")

    results = Parallel(n_jobs=-1, verbose=10)(delayed(lmm_mirna)(mirna_names[i], exp[:, i], meta) for i in range(len(mirna_names)))

    results = [r for r in results if r is not None]

    results_df = pd.DataFrame(results).set_index("miRNA").sort_values(by="Var_Age", ascending=False)
    print(results_df, sep='\t')
    results_df.to_csv(OUTPUT_VARIANCE)

    create_boxplot(results_df)

main()
