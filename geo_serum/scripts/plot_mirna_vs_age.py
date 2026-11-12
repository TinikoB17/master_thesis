"""miRNA expression vs age in the non-cancer control cohorts (the controls dataset, so the
CONTROLS_AGE_RANGE filter applies), one panel per miRNA, points and fitted lines coloured by
cohort.

Undetected values (at or below each sample's original floor) are left out of the fits and plots.
The cohorts differ in mean age and in expression level (batch), so a correlation pooled over all
samples can come from the cohort difference alone. The miRNAs are therefore ranked by the
within-cohort correlation: age and expression are centred within each cohort first (partial
correlation with age, adjusted for cohort). The pooled and per-cohort correlations are reported
too.

Outputs: a stats table (all tested miRNAs), a grid of the top miRNAs (SVG) and every tested
miRNA (PDF, one page per 12)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as st

EXPRESSION = snakemake.input.expression
METADATA = snakemake.input.metadata
NAME = snakemake.params.name  # dataset name in the titles
COHORT_ORDER = snakemake.params.cohort_order  # fixed colour order of the cohorts
MIN_DETECTION = snakemake.params.min_detection  # minimum detection rate in every cohort
MIN_DETECTED = 10  # minimum detected samples for a per-cohort fit
N_TOP = snakemake.params.n_top

# Same categorical colours and inks as the UMAP plots (scripts/plot_umap.py)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, MUTED, EDGE = "#3d3d3a", "#73726c", "#d6d5ce"
PER_PAGE_COLUMNS, PER_PAGE_ROWS = 4, 3


def detected_values(expression: pd.DataFrame, floors: pd.Series) -> pd.DataFrame:
    """Values at or below the sample's original floor (undetected) become NaN."""
    return expression.mask(expression.round(9).le(floors.round(9), axis=0))


def within_cohort_r(values: pd.Series, age: pd.Series, cohort: pd.Series):
    """Correlation of age and expression after centring both within each cohort, and its
    p-value (degrees of freedom reduced by one per cohort mean)."""
    frame = pd.DataFrame({"x": age, "y": values, "g": cohort}).dropna()
    frame = frame[frame["g"].map(frame["g"].value_counts()) >= MIN_DETECTED]
    centred = frame[["x", "y"]] - frame.groupby("g")[["x", "y"]].transform("mean")
    r = np.corrcoef(centred["x"], centred["y"])[0, 1]
    df = len(frame) - frame["g"].nunique() - 1
    t = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    return r, 2 * st.t.sf(abs(t), df), len(frame)


def correlation_table(detected: pd.DataFrame, age: pd.Series, cohort: pd.Series) -> pd.DataFrame:
    rows = []
    for mirna in detected.columns:
        values = detected[mirna]
        keep = values.notna()
        row = {"miRNA": mirna, "n_detected": int(keep.sum()), "detection_rate": keep.mean()}
        row["pooled_r"], row["pooled_p"] = st.pearsonr(age[keep], values[keep])
        row["within_r"], row["within_p"], _ = within_cohort_r(values, age, cohort)
        for name in COHORT_ORDER:
            in_cohort = keep & (cohort == name)
            if in_cohort.sum() >= MIN_DETECTED:
                fit = st.linregress(age[in_cohort], values[in_cohort])
                row[f"r_{name}"], row[f"slope_{name}"] = fit.rvalue, fit.slope
            row[f"n_{name}"] = int(in_cohort.sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    table["within_fdr"] = st.false_discovery_control(table["within_p"], method="bh")
    table["pooled_fdr"] = st.false_discovery_control(table["pooled_p"], method="bh")
    order = table["within_r"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def plot_mirna(ax, row, values, age, cohort, colors):
    for name in COHORT_ORDER:
        keep = values.notna() & (cohort == name)
        if not keep.any():
            continue
        ax.scatter(age[keep], values[keep], s=3, alpha=0.3, color=colors[name],
                   edgecolors="none", rasterized=True)
    for name in COHORT_ORDER:  # lines on top of every cohort's points
        if pd.notna(row.get(f"slope_{name}")):
            keep = values.notna() & (cohort == name)
            fit = st.linregress(age[keep], values[keep])
            ages = np.array([age[keep].min(), age[keep].max()])
            ax.plot(ages, fit.intercept + fit.slope * ages, color="white", linewidth=4)
            ax.plot(ages, fit.intercept + fit.slope * ages, color=colors[name], linewidth=2)
    ax.set_title(row["miRNA"], fontsize=10, color=TEXT)
    ax.text(0.03, 0.97,
            f"within-cohort r = {row['within_r']:.2f} (FDR {row['within_fdr']:.1e})\n"
            f"pooled r = {row['pooled_r']:.2f}, n = {row['n_detected']}",
            transform=ax.transAxes, va="top", fontsize=7.5, color=TEXT,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.grid(color=EDGE, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(EDGE)


def plot_grid(table, detected, age, cohort, colors, n_columns, title):
    n_rows = int(np.ceil(len(table) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(3.6 * n_columns, 3.0 * n_rows + 0.6),
                             squeeze=False)
    for ax, (_, row) in zip(axes.flat, table.iterrows()):
        plot_mirna(ax, row, detected[row["miRNA"]], age, cohort, colors)
    for ax in axes.flat[len(table):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Age (years)", fontsize=9, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("Expression (log2)", fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], color=colors[name], marker="o", markersize=8, linewidth=2)
               for name in COHORT_ORDER]
    labels = [f"{name} (n = {(cohort == name).sum()})" for name in COHORT_ORDER]
    fig.suptitle(title, fontsize=13, color=TEXT)
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1 - 0.45 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    return fig


def main():
    expression = pd.read_csv(EXPRESSION, sep="\t", index_col=0)
    metadata = pd.read_csv(METADATA, sep="\t", index_col=0).loc[expression.index]
    age = metadata["Age"].astype(float)
    cohort = metadata["Disease_State"]
    unknown = set(cohort) - set(COHORT_ORDER)
    if unknown:
        raise ValueError(f"Cohorts missing from cohort_order: {sorted(unknown)}")
    colors = dict(zip(COHORT_ORDER, CATEGORICAL))

    detected = detected_values(expression, metadata["Original_Floor"])
    # Detected in at least MIN_DETECTION of the samples of every cohort, so no cohort's fit
    # rests on a few detected values
    rates = detected.notna().groupby(cohort).mean()
    tested = detected.loc[:, (rates >= MIN_DETECTION).all()]
    print(f"{tested.shape[1]} of {expression.shape[1]} miRNAs detected in >= {MIN_DETECTION:.0%} "
          f"of the samples of every cohort ({len(expression)} samples, age "
          f"{int(age.min())}-{int(age.max())})")

    table = correlation_table(tested, age, cohort)
    table.to_csv(snakemake.output.stats, sep="\t", index=False)
    significant = table["within_fdr"] < 0.05
    print(f"Within-cohort FDR < 0.05: {significant.sum()} miRNAs "
          f"({(table.loc[significant, 'within_r'] > 0).sum()} increasing with age); "
          f"pooled FDR < 0.05: {(table['pooled_fdr'] < 0.05).sum()}")
    print(table.head(10)[["miRNA", "within_r", "within_fdr", "pooled_r"]
                         + [f"r_{name}" for name in COHORT_ORDER]].to_string(index=False))

    plt.rcParams["svg.fonttype"] = "none"
    subtitle = f"{NAME} non-cancer cohorts, age {int(age.min())}-{int(age.max())}"
    fig = plot_grid(table.head(N_TOP), tested, age, cohort, colors, n_columns=6,
                    title=f"{subtitle}: {N_TOP} miRNAs most correlated with age within cohorts")
    fig.savefig(snakemake.output.top_plot, dpi=200)
    plt.close(fig)

    per_page = PER_PAGE_COLUMNS * PER_PAGE_ROWS
    with PdfPages(snakemake.output.all_plots) as pdf:
        for start in range(0, len(table), per_page):
            page = table.iloc[start:start + per_page]
            fig = plot_grid(page, tested, age, cohort, colors, n_columns=PER_PAGE_COLUMNS,
                            title=f"{subtitle}: miRNA expression vs age ({start + 1}-"
                                  f"{start + len(page)} of {len(table)}, by within-cohort |r|)")
            pdf.savefig(fig, dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()
