"""Differentially expressed miRNAs in blood: SDAI (all classes) vs healthy, using the
miRNATissueAtlas blood data (../mirnatissueatlas/test_output/...cleaned_blood.h5ad, RPMM).

Exploratory side analysis, separate from the mirnatissueatlas pipeline.

Model per miRNA, on log2(1 + RPMM), by ordinary least squares:
    expression ~ group + age + sex
group coefficient = log2 fold change (case - reference), t-test, Benjamini-Hochberg FDR and
Bonferroni. Significant: FDR < 0.05 and |log2FC| >= 1.

CAUTION - confounding: all SDAI samples come from one project (PRJNA381937, Illumina HiSeq
2000) that has no healthy samples, so group and project (lab, protocol, sequencer) cannot be
separated and a project difference shows up as "differential expression". Two extra comparisons
put the main result in context:
  sdai_vs_healthy_hiseq2000: SDAI vs the healthy samples from the same sequencer type only
                             (removes the platform, not the lab)
  batch_control:             healthy PPMI vs healthy from the other projects; no disease
                             difference at all, so every hit here is a batch effect
Also, without the confounding:
  pd_vs_healthy_ppmi:        Parkinson's disease vs healthy, both from PPMI (one study, one
                             lab). PPMI samples have no subject/visit IDs in the metadata, so
                             repeated visits of one person (if any) cannot be accounted for

Usage: python de_sdai_vs_healthy.py [--h5ad PATH] [--out DIR]
"""
import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
DEFAULT_H5AD = HERE.parent / "mirnatissueatlas/test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad"

MIN_RPMM = 1.0           # a miRNA counts as expressed in a sample at RPMM >= MIN_RPMM
MIN_PREVALENCE = 0.5     # ... and is tested if expressed in >= 50% of the samples of either group
MAX_FDR = 0.05
MIN_ABS_LOG2FC = 1.0
ALPHA = 0.05             # Bonferroni: p < ALPHA / number of miRNAs tested


def comparisons(obs):
    healthy = obs["Disease_Condition"] == "Healthy"
    sdai = obs["Disease_Condition"].astype(str).str.startswith("SDAI")
    hiseq2000 = obs["Sequencing_Method"].astype(str) == "Illumina HiSeq 2000"
    ppmi = obs["Project"].astype(str) == "PPMI"
    # name -> (case mask, reference mask, case label, reference label)
    return {
        "sdai_vs_healthy": (sdai, healthy, "SDAI", "Healthy"),
        "sdai_vs_healthy_hiseq2000": (sdai, healthy & hiseq2000, "SDAI", "Healthy (HiSeq 2000)"),
        "batch_control": (healthy & ~ppmi, healthy & ppmi, "Healthy, other projects", "Healthy, PPMI"),
        "pd_vs_healthy_ppmi": ((obs["Disease_Condition"] == "PD") & ppmi, healthy & ppmi, "PD (PPMI)", "Healthy (PPMI)"),
    }


def differential_expression(rpmm, obs, case, reference):
    """OLS per miRNA: log2(1 + RPMM) ~ group + age + sex; returns the results table."""
    samples = case | reference
    rpmm, obs = rpmm.loc[samples], obs.loc[samples]
    is_case = case[samples].astype(float)

    # Overlapping age range, so age is not extrapolated between the groups
    low = max(obs.loc[is_case == 1, "Age"].min(), obs.loc[is_case == 0, "Age"].min())
    high = min(obs.loc[is_case == 1, "Age"].max(), obs.loc[is_case == 0, "Age"].max())
    keep = obs["Age"].between(low, high) & obs["Sex"].isin(["m", "f"])
    rpmm, obs, is_case = rpmm.loc[keep], obs.loc[keep], is_case[keep]

    expressed = rpmm >= MIN_RPMM
    prevalence_case = expressed[is_case == 1].mean()
    prevalence_ref = expressed[is_case == 0].mean()
    tested = (prevalence_case >= MIN_PREVALENCE) | (prevalence_ref >= MIN_PREVALENCE)
    Y = np.log2(1 + rpmm.loc[:, tested].to_numpy(dtype=float))

    X = np.column_stack([np.ones(len(obs)), is_case, obs["Age"].to_numpy(float),
                         (obs["Sex"] == "m").to_numpy(float)])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    df = len(obs) - X.shape[1]
    sigma2 = ((Y - X @ beta) ** 2).sum(axis=0) / df
    se = np.sqrt(sigma2 * XtX_inv[1, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta[1] / se
    p = 2 * stats.t.sf(np.abs(t), df)
    p = np.nan_to_num(p, nan=1.0)
    n_tested = int(tested.sum())

    table = pd.DataFrame({
        "miRNA": rpmm.columns[tested],
        "log2FC": beta[1],
        "se": se,
        "t": t,
        "pval": p,
        "padj_bh": multipletests(p, method="fdr_bh")[1],
        "p_bonferroni": np.minimum(p * n_tested, 1.0),
        "mean_log2_case": Y[is_case.to_numpy() == 1].mean(axis=0),
        "mean_log2_reference": Y[is_case.to_numpy() == 0].mean(axis=0),
        "prevalence_case": prevalence_case[tested].values,
        "prevalence_reference": prevalence_ref[tested].values,
    })
    table["significant"] = (table["padj_bh"] < MAX_FDR) & (table["log2FC"].abs() >= MIN_ABS_LOG2FC)
    table["bonferroni_significant"] = table["pval"] < ALPHA / n_tested
    info = {"n_case": int((is_case == 1).sum()), "n_reference": int((is_case == 0).sum()),
            "age_range": f"{low:.0f}-{high:.0f}", "n_tested": n_tested,
            "case_projects": sorted(obs.loc[is_case == 1, "Project"].astype(str).unique()),
            "reference_projects": sorted(obs.loc[is_case == 0, "Project"].astype(str).unique())}
    return table.sort_values("pval").reset_index(drop=True), info


def volcano(table, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    up = table["significant"] & (table["log2FC"] > 0)
    down = table["significant"] & (table["log2FC"] < 0)
    ns = ~table["significant"]
    neg_log_p = -np.log10(table["pval"].clip(lower=1e-300))
    for mask, color, label in [(ns, "lightgray", "not significant"),
                               (down, "#2a78d6", "lower in case"), (up, "#e34948", "higher in case")]:
        ax.scatter(table.loc[mask, "log2FC"], neg_log_p[mask], s=10, alpha=0.75, color=color,
                   edgecolors="none", label=f"{label} ({int(mask.sum())})")
    for x in (-MIN_ABS_LOG2FC, MIN_ABS_LOG2FC):
        ax.axvline(x, color="grey", linestyle="--", linewidth=0.8)
    for _, row in table[table["significant"]].nsmallest(8, "pval").iterrows():
        ax.annotate(row["miRNA"].replace("hsa-", ""), (row["log2FC"], -np.log10(max(row["pval"], 1e-300))),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("log2 fold change (case vs reference, adjusted for age and sex)")
    ax.set_ylabel("-log10 p-value")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5ad", default=str(DEFAULT_H5AD))
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)
    obs = adata.obs.copy()
    obs["Age"] = pd.to_numeric(obs["Age"], errors="coerce")
    rpmm = adata.to_df()

    summary = []
    for name, (case, reference, case_label, ref_label) in comparisons(obs).items():
        table, info = differential_expression(rpmm, obs, case, reference)
        table.to_csv(out / f"{name}.tsv", sep="\t", index=False)
        sig = table[table["significant"]]
        summary.append({
            "comparison": name, "case": case_label, "reference": ref_label, **info,
            "significant": len(sig), "up": int((sig["log2FC"] > 0).sum()), "down": int((sig["log2FC"] < 0).sum()),
            "bonferroni": int(table["bonferroni_significant"].sum()),
            "fdr_only": int((table["padj_bh"] < MAX_FDR).sum()),
            "median_abs_log2FC": table["log2FC"].abs().median(),
        })
        volcano(table, f"{case_label} (n = {info['n_case']}) vs {ref_label} (n = {info['n_reference']})\n"
                       f"{len(sig)} of {info['n_tested']} miRNAs: FDR < {MAX_FDR}, |log2FC| >= {MIN_ABS_LOG2FC}",
                out / f"{name}_volcano.svg")
        print(f"\n== {name}: {case_label} vs {ref_label}")
        print(f"   {info}")
        print(sig.head(15)[["miRNA", "log2FC", "padj_bh", "mean_log2_case", "mean_log2_reference"]]
              .round(4).to_string(index=False))

    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print("\nSummary:")
    print(summary.drop(columns=["case_projects", "reference_projects"]).round(3).to_string(index=False))

    # Overlap of the main hits with the platform-matched comparison and with the batch control
    main_hits = set(pd.read_csv(out / "sdai_vs_healthy.tsv", sep="\t").query("significant")["miRNA"])
    for other in ["sdai_vs_healthy_hiseq2000", "batch_control"]:
        hits = set(pd.read_csv(out / f"{other}.tsv", sep="\t").query("significant")["miRNA"])
        print(f"sdai_vs_healthy hits also significant in {other}: {len(main_hits & hits)} of {len(main_hits)}")


if __name__ == "__main__":
    main()
