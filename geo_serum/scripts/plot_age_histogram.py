import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

METADATA = snakemake.input.metadata
GROUP = snakemake.params.group
# Disease_State values in the group, e.g. all three non-cancer control groups
DISEASE_STATES = snakemake.params.disease_states
OUTPUT = snakemake.output.histogram

# Same colours as the age histograms of the miRNATissueAtlas pipeline (mirnatissueatlas/scripts/visualize_input_data.py)
SEX_COLORS = {"Male": "mediumspringgreen", "Female": "salmon"}


def main():
    metadata = pd.read_csv(METADATA, sep="\t", index_col=0)
    samples = metadata[metadata["Disease_State"].isin(DISEASE_STATES)]
    if samples["Age"].isna().any():
        print(f"{GROUP}: {samples['Age'].isna().sum()} samples without age left out")
    samples = samples.dropna(subset=["Age"])
    if samples.empty:
        raise ValueError(f"No samples with Disease_State in {DISEASE_STATES}; "
                         f"groups: {sorted(metadata['Disease_State'].unique())}")
    sexes = [sex for sex in SEX_COLORS if (samples["Sex"] == sex).any()]
    n_samples = len(samples)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    fig, ax = plt.subplots(figsize=(7, 5))

    # One bin per year would leave gaps for integer ages, so the bins are whole years wide
    # at a width that gives about 20 bins
    width = max(1, int(np.ceil((samples["Age"].max() - samples["Age"].min() + 1) / 20)))
    bins = np.arange(samples["Age"].min() - 0.5, samples["Age"].max() + width, width)
    sns.histplot(data=samples, x="Age", hue="Sex", hue_order=sexes, palette=SEX_COLORS,
                 multiple="stack", bins=bins, stat="count", edgecolor="black", alpha=0.8,
                 ax=ax, legend=False)

    # Density of all samples, scaled to the number of samples per bin
    ages = np.linspace(bins[0], bins[-1], 300)
    density = gaussian_kde(samples["Age"].astype(float))(ages)
    ax.plot(ages, density * n_samples * width, color="black", linewidth=1.5)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel("Number of samples", fontsize=12)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SEX_COLORS[sex], edgecolor="black", alpha=0.8)
               for sex in sexes]
    labels = [f"{sex} (n = {(samples['Sex'] == sex).sum()})" for sex in sexes]
    handles.append(plt.Line2D([], [], color="black", linewidth=1.5))
    labels.append("All samples (density)")
    ax.legend(handles, labels, fontsize=10, loc="upper left", frameon=False)

    ax.set_title(f"Age Distribution\n{GROUP} (n = {n_samples})", fontsize=14)
    sns.despine()
    fig.tight_layout()
    fig.savefig(OUTPUT)
    plt.close(fig)

    ages = samples["Age"]
    print(f"{GROUP}: n = {n_samples}, age {ages.min()}-{ages.max()}, "
          f"median {ages.median()}, mean {ages.mean():.1f} -> {OUTPUT}")


if __name__ == "__main__":
    main()
