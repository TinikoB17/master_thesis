from itertools import cycle, islice

import matplotlib.pyplot as plt
import pandas as pd

# Number of blood samples (all disease conditions) per sequencing method and per
# disease condition, as horizontal bar charts with the most common category on top.

metadata_path = snakemake.input.blood_metadata

COLORS = ["olivedrab", "salmon", "lightgreen", "sandybrown", "crimson", "mediumslateblue", "darkseagreen", "tan",
          "maroon", "rosybrown", "steelblue", "lightcoral", "mediumaquamarine", "peru", "indianred", "lightsteelblue"]

data_blood = pd.read_csv(metadata_path, sep="\t")


def plot_counts(column, title, path):
    counts = data_blood[column].value_counts().sort_values()
    colors = list(islice(cycle(COLORS), len(counts)))[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(counts) + 1.5)))
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.bar_label(bars, padding=3, fontsize=12)
    ax.set_xlabel("Number of samples", fontsize=16)
    fig.suptitle(f"{title} (n = {len(data_blood)})", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlim(0, counts.max() * 1.12)  # room for the bar labels
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


plot_counts("Sequencing_Method", "Blood Sample Count by Sequencing Method", snakemake.output.sequencing_method)
plot_counts("Disease_Condition", "Blood Sample Count by Disease Condition", snakemake.output.disease_condition)
