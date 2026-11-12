"""Differentially expressed serum miRNAs in GSE113740 (Toray 3D-Gene microarray): gastric
cancer vs non-cancer, using the geo_serum pipeline data.

Exploratory side analysis, separate from the geo_serum pipeline. Same model as
de_sdai_vs_healthy.py, per miRNA by ordinary least squares:
    expression ~ group + age + sex
on the pipeline's expression values (log2, quantile normalised across all samples, miRBase
high-confidence miRNAs: ../geo_serum/data/GSE113740/expression.tsv). group coefficient =
log2 fold change (case - reference), t-test, Benjamini-Hochberg FDR and Bonferroni.
Significant: FDR < 0.05 and |log2FC| >= 1.
A miRNA is tested if detected (above its sample's undetected floor, scripts/undetected.py) in
>= 50% of the samples of either group.

Batches: the cancer panel (25 per cancer type) and Non-cancer 1 share the deposited scaling
(per-sample floor about -0.3) and cluster together in the UMAP; Non-cancer 2 and 3 have a floor
about -3.5 and form their own cluster. Comparisons:
  gastric_vs_noncancer:       Gastric cancer vs Non-cancer 1-3 (mixes in the batch difference)
  gastric_vs_noncancer1:      Gastric cancer vs Non-cancer 1 (same batch: the main comparison)
  batch_control:              Non-cancer 2+3 vs Non-cancer 1; no disease difference, so every
                              hit is a batch effect
  gastric_vs_other_cancers:   Gastric cancer vs the other cancer types of the panel
                              (gastric-specific vs shared cancer changes)

Usage: python de_gse113740_gastric.py [--data DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
GEO_SERUM = HERE.parent / "geo_serum"
sys.path.insert(0, str(GEO_SERUM / "scripts"))
from undetected import undetected_values  # noqa: E402

MIN_PREVALENCE = 0.5
MAX_FDR = 0.05
MIN_ABS_LOG2FC = 1.0
ALPHA = 0.05

PANEL_CANCERS = ["Biliary Tract Cancer", "Bladder Cancer", "Breast Cancer", "Colorectal Cancer",
                 "Esophageal Cancer", "Glioma", "Hepatocellular Carcinoma", "Lung Cancer",
                 "Ovarian Cancer", "Pancreatic Cancer", "Prostate Cancer", "Sarcoma"]


def comparisons(meta):
    state = meta["Disease_State"]
    gastric = state == "Gastric Cancer"
    nc1 = state == "Non-cancer 1"
    nc23 = state.isin(["Non-cancer 2", "Non-cancer 3"])
    return {
        "gastric_vs_noncancer": (gastric, nc1 | nc23, "Gastric cancer", "Non-cancer 1-3"),
        "gastric_vs_noncancer1": (gastric, nc1, "Gastric cancer", "Non-cancer 1"),
        "batch_control": (nc23, nc1, "Non-cancer 2+3", "Non-cancer 1"),
        "gastric_vs_other_cancers": (gastric, state.isin(PANEL_CANCERS), "Gastric cancer", "Other panel cancers"),
    }


def differential_expression(expression, detected, meta, case, reference):
    samples = case | reference
    keep = samples & meta["Age"].notna() & meta["Sex"].isin(["Male", "Female"])
    is_case = case[keep].astype(float)
    low = max(meta.loc[keep & case, "Age"].min(), meta.loc[keep & reference, "Age"].min())
    high = min(meta.loc[keep & case, "Age"].max(), meta.loc[keep & reference, "Age"].max())
    keep &= meta["Age"].between(low, high)
    is_case = case[keep].astype(float)
    data, det, m = expression.loc[keep], detected.loc[keep], meta.loc[keep]

    prevalence_case = det[is_case == 1].mean()
    prevalence_ref = det[is_case == 0].mean()
    tested = (prevalence_case >= MIN_PREVALENCE) | (prevalence_ref >= MIN_PREVALENCE)
    Y = data.loc[:, tested].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(m)), is_case, m["Age"].to_numpy(float),
                         (m["Sex"] == "Male").to_numpy(float)])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        X = X[:, :3]  # one sex only (e.g. a single-sex group): no sex term
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    df = len(m) - X.shape[1]
    sigma2 = ((Y - X @ beta) ** 2).sum(axis=0) / df
    se = np.sqrt(sigma2 * XtX_inv[1, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta[1] / se
    p = np.nan_to_num(2 * stats.t.sf(np.abs(t), df), nan=1.0)
    n_tested = int(tested.sum())

    table = pd.DataFrame({
        "miRNA": data.columns[tested], "log2FC": beta[1], "se": se, "t": t, "pval": p,
        "padj_bh": multipletests(p, method="fdr_bh")[1],
        "p_bonferroni": np.minimum(p * n_tested, 1.0),
        "mean_case": Y[is_case.to_numpy() == 1].mean(axis=0),
        "mean_reference": Y[is_case.to_numpy() == 0].mean(axis=0),
        "detected_case": prevalence_case[tested].values,
        "detected_reference": prevalence_ref[tested].values,
    })
    table["significant"] = (table["padj_bh"] < MAX_FDR) & (table["log2FC"].abs() >= MIN_ABS_LOG2FC)
    table["bonferroni_significant"] = table["pval"] < ALPHA / n_tested
    info = {"n_case": int((is_case == 1).sum()), "n_reference": int((is_case == 0).sum()),
            "age_range": f"{low:.0f}-{high:.0f}", "n_tested": n_tested}
    return table.sort_values("pval").reset_index(drop=True), info


def volcano(table, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    up = table["significant"] & (table["log2FC"] > 0)
    down = table["significant"] & (table["log2FC"] < 0)
    neg_log_p = -np.log10(table["pval"].clip(lower=1e-300))
    for mask, color, label in [(~table["significant"], "lightgray", "not significant"),
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
    parser.add_argument("--data", default=str(GEO_SERUM / "data/GSE113740"))
    parser.add_argument("--out", default=str(HERE / "results/gse113740_gastric"))
    args = parser.parse_args()
    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    expression = pd.read_csv(data_dir / "expression.tsv", sep="\t", index_col=0)
    meta = pd.read_csv(data_dir / "metadata.tsv", sep="\t", index_col=0).loc[expression.index]
    # Detection from the values as deposited (before normalisation): above the sample's floor
    named = pd.read_csv(data_dir / "expression_named.tsv", sep="\t", index_col=0).loc[expression.index]
    undetected, _ = undetected_values(named)
    detected = ~undetected[expression.columns]

    summary, hits = [], {}
    for name, (case, reference, case_label, ref_label) in comparisons(meta).items():
        table, info = differential_expression(expression, detected, meta, case, reference)
        table.to_csv(out / f"{name}.tsv", sep="\t", index=False)
        sig = table[table["significant"]]
        hits[name] = sig.set_index("miRNA")["log2FC"]
        summary.append({"comparison": name, "case": case_label, "reference": ref_label, **info,
                        "significant": len(sig), "up": int((sig["log2FC"] > 0).sum()),
                        "down": int((sig["log2FC"] < 0).sum()),
                        "bonferroni": int(table["bonferroni_significant"].sum()),
                        "fdr_only": int((table["padj_bh"] < MAX_FDR).sum()),
                        "median_abs_log2FC": table["log2FC"].abs().median()})
        volcano(table, f"GSE113740: {case_label} (n = {info['n_case']}) vs {ref_label} (n = {info['n_reference']})\n"
                       f"{len(sig)} of {info['n_tested']} miRNAs: FDR < {MAX_FDR}, |log2FC| >= {MIN_ABS_LOG2FC}",
                out / f"{name}_volcano.svg")
        print(f"\n== {name}: {case_label} vs {ref_label} {info}")
        print(sig.head(15)[["miRNA", "log2FC", "padj_bh", "mean_case", "mean_reference"]].round(4).to_string(index=False))

    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print("\nSummary:")
    print(summary.round(3).to_string(index=False))

    main_hits = hits["gastric_vs_noncancer1"]
    for other in ["gastric_vs_noncancer", "batch_control", "gastric_vs_other_cancers"]:
        shared = main_hits.index.intersection(hits[other].index)
        print(f"gastric_vs_noncancer1 hits also significant in {other}: {len(shared)} of {len(main_hits)}")


if __name__ == "__main__":
    main()
