
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd

def plot_umap_from_anndata(annotated_object: sc.AnnData, color_by: str, blood_or_all_tissues: str, saveas=None):

    tissue = "blood" if blood_or_all_tissues == "Blood" else "all"
    save_as = f'_{tissue}_{color_by}.svg'
    print(save_as)
    title = f'UMAP Projection of Samples Colored by {color_by}'
    if saveas:
        save_as = saveas
    sc.pp.scale(annotated_object)
    sc.pp.pca(annotated_object, n_comps=50)
    sc.pp.neighbors(annotated_object)
    sc.tl.umap(annotated_object)
    pl = sc.pl.umap(annotated_object, color=color_by, save=save_as, title=title, cmap="plasma")


def plot_highlight_blood(annotated_object: sc.AnnData):
    tissues = annotated_object.obs["Tissue"].unique().to_list()
    palette = {t: "tab:red" if t == "blood" else "lightgray" for t in tissues}
    sc.pp.neighbors(annotated_object)
    sc.tl.umap(annotated_object)
    tissue = sc.pl.umap(annotated_object, color="Tissue", save='_highlight_blood.svg', palette=palette, title="UMAP Projection of Samples Colored by Tissue - Highlighting Blood")

def plot_age_histograms(annotated_object: sc.AnnData,
                        condition: str = None,
                        tissue: str = None,
                        train_test: str = None):

    obs = annotated_object.obs

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.3
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.histplot(
        data=obs,
        x="Age",
        bins=20,
        kde=True,
        stat="count",
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
        ax=ax
    )

    ax.set_xlim(0, 100)

    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel("Number of samples", fontsize=12)

    n_samples = len(obs)

    if train_test:
        title = (
            f"Age Distribution\n in "
            f"{train_test} set"
            f"(n = {n_samples})"
        )
    else:
        if tissue == "All" and condition == "All":
            title = f"Age Distribution of All Samples (n = {n_samples})"
        else:
            title = (
                f"Age Distribution\n"
                f"Condition: {condition}, Tissue: {tissue}\n"
                f"(n = {n_samples})"
            )

    ax.set_title(title, fontsize=14)

    sns.despine()

    fig.tight_layout()
    if train_test:
        fig.savefig(
        f"figures/{train_test}_age_histogram.svg",
        bbox_inches="tight"
        )
    else:
        fig.savefig(
            f"figures/{condition}_{tissue}_age_histogram.svg",
            bbox_inches="tight"
        )

    plt.close(fig)


def plot_count_of_miRNAs_at_threshold(annotated_object: sc.AnnData, saveas: str, mirna_list_saveas: str):
    expressions = annotated_object.to_df()
    exp_transposed = expressions.T
    exp_transposed['count_exceeds_1'] = (exp_transposed > 1).sum(axis=1)
    exp_transposed['count_exceeds_one_in_fraction'] = exp_transposed['count_exceeds_1'] / (len(exp_transposed.columns.to_list()) - 1)
    exp_transposed.index.name = "miRNA"
    print(exp_transposed)
    mirnas_that_pass_threshold = exp_transposed[exp_transposed["count_exceeds_one_in_fraction"] >= 0.2].index.tolist()
    other_mirnas = len(exp_transposed[exp_transposed["count_exceeds_one_in_fraction"] >= 0.2].index.tolist())
    print(other_mirnas)
    print(len(mirnas_that_pass_threshold), "0.5")
    mirnas_df = pd.DataFrame(mirnas_that_pass_threshold)

    mirnas_df.to_csv(mirna_list_saveas, sep='\t', index=False)

    fractions = np.linspace(0, 1, 101)

    num_miRNAs = [
        (exp_transposed['count_exceeds_one_in_fraction'] >= f).sum()
        for f in fractions
    ]

    plt.figure()
    plt.plot(fractions * 100, num_miRNAs)
    plt.xlabel('% of samples with expression > 1')
    plt.ylabel('Number of miRNAs')
    plt.title('miRNAs expressed above rpmm 1 across samples')

    plt.savefig(saveas, dpi=300, bbox_inches='tight')
    plt.close()
    return mirnas_that_pass_threshold
