
import scanpy as sc
from pathlib import Path


def plot_umap_from_anndata(annotated_object: sc.AnnData, color_by: str, blood_or_all_tissues: str):

    tissue = "blood" if blood_or_all_tissues == "Blood" else "all"
    save_as = f'_{tissue}_{color_by}.svg'

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

