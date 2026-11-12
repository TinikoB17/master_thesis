"""UMAP of all samples of a series, coloured by age, disease state, sex and each sample's
original floor value (the per-sample scaling of the deposited data, which drove the batch
structure in the pooled GEO data). Same steps as the miRNATissueAtlas pipeline
(plot_umap_from_anndata): scale each miRNA, PCA, neighbour graph, UMAP."""
import matplotlib

matplotlib.use("Agg")
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

N_PCS = snakemake.params.n_pcs
N_NEIGHBORS = snakemake.params.n_neighbors
SEED = snakemake.params.seed
NAME = snakemake.params.name  # dataset name in the titles
DISEASE_ORDER = snakemake.params.disease_order  # fixed colour order of the disease groups
PLOT_GROUPS = snakemake.params.plot_groups  # disease state -> disease group
SEX_ORDER = ["Male", "Female"]

# Categorical colours, assigned in a fixed order (dataviz reference palette, light mode)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
               "#e34948"]
# Sequential single-hue ramp (blue, steps 250 -> 700) for age and floor
BLUE = LinearSegmentedColormap.from_list(
    "blue", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
TEXT, MUTED, EDGE = "#3d3d3a", "#73726c", "#d6d5ce"
# Samples with a missing category (e.g. sex not given)
UNKNOWN, UNKNOWN_COLOR = "Unknown", "#b4b2a9"


def embed(expression: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    adata = ad.AnnData(X=expression.to_numpy(dtype=float), obs=metadata)
    adata.obs_names = expression.index
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=N_PCS, random_state=SEED)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    variance = adata.uns["pca"]["variance_ratio"]
    print(f"PCA: first 5 components explain {np.round(variance[:5] * 100, 1).tolist()} % of the variance")
    return pd.DataFrame(adata.obsm["X_umap"], index=expression.index, columns=["UMAP1", "UMAP2"])


def new_axes(title, n):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.set_title(f"{title}\n{NAME}, all samples (n = {n})", fontsize=12, color=TEXT)
    ax.set_xlabel("UMAP 1", color=MUTED)
    ax.set_ylabel("UMAP 2", color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    return fig, ax


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_categorical(data, column, order, title, path):
    colors = dict(zip([group for group in order if group != UNKNOWN], CATEGORICAL))
    colors[UNKNOWN] = UNKNOWN_COLOR
    fig, ax = new_axes(title, len(data))
    # Random drawing order, so the largest group does not hide the others
    shuffled = data.sample(frac=1, random_state=SEED)
    ax.scatter(shuffled["UMAP1"], shuffled["UMAP2"], c=shuffled[column].map(colors), s=7,
               alpha=0.75, linewidths=0.15, edgecolors="white", rasterized=True)
    counts = data[column].value_counts()
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=colors[group],
                      markeredgecolor="white") for group in order]
    ax.legend(handles, [f"{group} (n = {counts.get(group, 0)})" for group in order],
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, labelcolor=TEXT, fontsize=9)
    save(fig, path)


def plot_continuous(data, column, label, title, path):
    fig, ax = new_axes(title, len(data))
    shuffled = data.sample(frac=1, random_state=SEED)
    points = ax.scatter(shuffled["UMAP1"], shuffled["UMAP2"], c=shuffled[column], cmap=BLUE, s=7,
                        alpha=0.8, linewidths=0.15, edgecolors="white", rasterized=True)
    colorbar = fig.colorbar(points, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label(label, color=TEXT)
    colorbar.outline.set_visible(False)
    save(fig, path)


def main():
    expression = pd.read_csv(snakemake.input.expression, sep="\t", index_col=0)
    metadata = pd.read_csv(snakemake.input.metadata, sep="\t", index_col=0).loc[expression.index]
    floors = pd.read_csv(snakemake.input.floors, sep="\t", index_col=0)
    unknown = set(metadata["Disease_State"]) - set(PLOT_GROUPS)
    if unknown:
        raise ValueError(f"Disease states without a disease group: {unknown}")
    if len(DISEASE_ORDER) > len(CATEGORICAL):
        raise ValueError(f"{len(DISEASE_ORDER)} disease groups but {len(CATEGORICAL)} colours")

    data = embed(expression, metadata).join(metadata[["Age", "Disease_State", "Sex"]])
    data["Disease_Group"] = data["Disease_State"].map(PLOT_GROUPS)
    data["Original_Floor"] = floors.loc[data.index, "original_floor"]
    data.to_csv(snakemake.output.embedding, sep="\t")

    plot_continuous(data, "Age", "Age (years)", "UMAP coloured by age", snakemake.output.age)
    plot_categorical(data, "Disease_Group", DISEASE_ORDER, "UMAP coloured by disease state",
                     snakemake.output.disease_state)
    sex = data.assign(Sex=data["Sex"].fillna(UNKNOWN))
    sex_order = SEX_ORDER + ([UNKNOWN] if (sex["Sex"] == UNKNOWN).any() else [])
    plot_categorical(sex, "Sex", sex_order, "UMAP coloured by sex", snakemake.output.sex)
    plot_continuous(data, "Original_Floor", "Original floor (log2)",
                    "UMAP coloured by original floor value", snakemake.output.floor)
    print(f"UMAP of {len(data)} samples x {expression.shape[1]} miRNAs")


if __name__ == "__main__":
    main()
