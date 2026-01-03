
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd

def plot_umap_from_anndata(annotated_object: sc.AnnData, color_by: str, blood_or_all_tissues: str):

    tissue = "blood" if blood_or_all_tissues == "Blood" else "all"
    save_as = f'_{tissue}_{color_by}.svg'
    print(save_as)
    title = f'UMAP Projection of Samples Colored by {color_by}'

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

def plot_age_histograms(annotated_object: sc.AnnData, condition: str, tissue: str):
    plt.figure()
    sns.set_theme(style="whitegrid")

    obs = annotated_object.obs
    histogram = sns.histplot(data=obs, x="Age", kde=True)
    if tissue == "All" and condition == "All":
        plt.title("Age Distribution in All Samples")
    else:
        plt.title(f"Age Distribution in Samples: Condition - {condition}, Tissue - {tissue}")
    hist = histogram.get_figure()
    hist.savefig(f"figures/{condition}_{tissue}_age_histogram.svg")


def plot_count_of_miRNAs_at_threshold(annotated_object: sc.AnnData, saveas: str, mirna_list_saveas: str):
    expressions = annotated_object.to_df()
    exp_transposed = expressions.T
    exp_transposed['count_exceeds_1'] = (exp_transposed > 1).sum(axis=1)
    exp_transposed['count_exceeds_one_in_fraction'] = exp_transposed['count_exceeds_1'] / (len(exp_transposed.columns.to_list()) - 1)
    exp_transposed.index.name = "miRNA"

    mirnas_that_pass_threshold = exp_transposed[exp_transposed["count_exceeds_one_in_fraction"] >= 0.5].index.tolist()
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
