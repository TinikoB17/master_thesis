import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictions(predictions: pd.DataFrame, bins, title: str, path: str):
    """Predicted vs. actual age coloured by project, and residuals by age bin and by project.
    Expects one row per sample with columns Project, actual_age, predicted_age."""
    residual = predictions["predicted_age"] - predictions["actual_age"]
    projects = predictions["Project"].value_counts().index.tolist()
    colors = dict(zip(projects, plt.cm.tab10.colors))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"width_ratios": [1.1, 1, 1]})

    ax = axes[0]
    for project in projects[::-1]:
        sub = predictions[predictions["Project"] == project]
        ax.scatter(sub["actual_age"], sub["predicted_age"], s=12, alpha=0.6, color=colors[project],
                   label=f"{project} (n={len(sub)})")
    low, high = bins[0], bins[-1]
    ax.plot([low, high], [low, high], linestyle="--", color="grey", linewidth=1)
    ax.set_xlim(low - 2, high + 2)
    ax.set_ylim(low - 2, high + 2)
    ax.set_xlabel("Actual age")
    ax.set_ylabel("Predicted age")
    ax.set_title("Predicted vs. actual")
    ax.legend(fontsize=7, loc="upper left")

    age_bin = pd.cut(predictions["actual_age"], bins=bins, include_lowest=True)
    labels = [str(interval) for interval in age_bin.cat.categories]
    axes[1].boxplot([residual[age_bin == interval] for interval in age_bin.cat.categories],
                    tick_labels=labels, showfliers=False)
    axes[1].axhline(0, linestyle="--", color="grey", linewidth=1)
    axes[1].set_xlabel("Actual age bin")
    axes[1].set_ylabel("Predicted - actual age")
    axes[1].set_title("Residuals by age")

    axes[2].boxplot([residual[predictions["Project"] == project] for project in projects],
                    tick_labels=[f"{p}\n(n={(predictions['Project'] == p).sum()})" for p in projects],
                    showfliers=False)
    axes[2].axhline(0, linestyle="--", color="grey", linewidth=1)
    axes[2].tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[2].set_ylabel("Predicted - actual age")
    axes[2].set_title("Residuals by project")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
