import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import false_discovery_control, linregress

EXPRESSION = snakemake.input.expression
METADATA = snakemake.input.metadata
GROUP = snakemake.params.group
MIN_DETECTION = snakemake.params.min_detection
N_TOP = snakemake.params.n_top
COMMON_FLOOR = snakemake.params.common_floor

STATS = snakemake.output.stats
TOP_PLOT = snakemake.output.top_plot
ALL_PLOTS = snakemake.output.all_plots

PER_PAGE_COLUMNS, PER_PAGE_ROWS = 4, 3


def detected_values(expression: pd.DataFrame) -> pd.DataFrame:
    """Undetected miRNAs all have the common floor value (set_common_floor rule); those values
    become NaN so they are left out of the fits."""
    return expression.mask(expression <= COMMON_FLOOR)


def fit_lines(expression: pd.DataFrame, age: pd.Series) -> pd.DataFrame:
    rows = []
    for mirna in expression.columns:
        values = expression[mirna]
        keep = values.notna()
        fit = linregress(age[keep], values[keep])
        rows.append({
            "miRNA": mirna,
            "n_detected": int(keep.sum()),
            "detection_rate": keep.mean(),
            "slope_per_year": fit.slope,
            "intercept": fit.intercept,
            "pearson_r": fit.rvalue,
            "r2": fit.rvalue ** 2,
            "p_value": fit.pvalue,
        })
    stats = pd.DataFrame(rows)
    stats["fdr"] = false_discovery_control(stats["p_value"], method="bh")
    stats["abs_r"] = stats["pearson_r"].abs()
    return stats.sort_values("abs_r", ascending=False).drop(columns="abs_r").reset_index(drop=True)


def plot_mirna(ax, mirna, values, age, fit):
    keep = values.notna()
    # Smaller, lighter points for large cohorts; rasterised so the PDF stays small
    size, alpha = (8, 0.5) if keep.sum() < 2000 else (2, 0.25)
    ax.scatter(age[keep], values[keep], s=size, alpha=alpha, color="steelblue", edgecolors="none",
               rasterized=True)
    ages = np.array([age.min(), age.max()])
    ax.plot(ages, fit["intercept"] + fit["slope_per_year"] * ages, color="darkorange", linewidth=1.8)
    ax.set_title(mirna, fontsize=10)
    ax.text(0.03, 0.97,
            f"r = {fit['pearson_r']:.2f}, FDR = {fit['fdr']:.1e}\n"
            f"slope = {fit['slope_per_year']:.3f}/yr, n = {fit['n_detected']}",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    ax.tick_params(labelsize=8)


def plot_grid(stats, expression, age, n_columns, title):
    n_rows = int(np.ceil(len(stats) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(3.6 * n_columns, 3.0 * n_rows),
                             squeeze=False)
    for ax, (_, fit) in zip(axes.flat, stats.iterrows()):
        plot_mirna(ax, fit["miRNA"], expression[fit["miRNA"]], age, fit)
    for ax in axes.flat[len(stats):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Age (years)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Expression (log2)", fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def main():
    expression = pd.read_csv(EXPRESSION, sep="\t", index_col=0)
    age = pd.read_csv(METADATA, sep="\t", index_col=0).loc[expression.index, "Age"].astype(float)

    detected = detected_values(expression)
    detection_rate = detected.notna().mean()
    tested = detected.loc[:, detection_rate >= MIN_DETECTION]
    print(f"{tested.shape[1]} of {expression.shape[1]} miRNAs detected in "
          f">= {MIN_DETECTION:.0%} of the {len(expression)} {GROUP} samples")

    stats = fit_lines(tested, age)
    stats.to_csv(STATS, sep="\t", index=False)
    print(f"FDR < 0.05: {(stats['fdr'] < 0.05).sum()} miRNAs "
          f"({(stats.loc[stats['fdr'] < 0.05, 'slope_per_year'] > 0).sum()} increasing with age)")
    print(stats.head(10).to_string(index=False))

    sns.set_theme(style="whitegrid", context="paper")

    fig = plot_grid(stats.head(N_TOP), tested, age, n_columns=6,
                    title=f"{GROUP}: {N_TOP} miRNAs most correlated with age "
                          f"(n = {len(expression)} samples)")
    fig.savefig(TOP_PLOT)
    plt.close(fig)

    # Every tested miRNA, strongest correlation first
    per_page = PER_PAGE_COLUMNS * PER_PAGE_ROWS
    with PdfPages(ALL_PLOTS) as pdf:
        for start in range(0, len(stats), per_page):
            page = stats.iloc[start:start + per_page]
            fig = plot_grid(page, tested, age, n_columns=PER_PAGE_COLUMNS,
                            title=f"{GROUP}: miRNA expression vs age "
                                  f"({start + 1}-{start + len(page)} of {len(stats)}, by |r|)")
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
