import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Workflow figure of the thesis pipeline, from the raw miRNA atlas to the age predictions.
# Sample/miRNA counts are read from the intermediate h5ad files and the metrics from the
# model evaluation files, so the figure stays in sync with the pipeline outputs.

RAW = snakemake.input.raw
CLEANED = snakemake.input.cleaned
BLOOD = snakemake.input.blood
BLOOD_HEALTHY = snakemake.input.blood_healthy
AGE_FILTERED = snakemake.input.age_filtered
TRAIN = snakemake.input.train
TEST = snakemake.input.test
PREVALENCE_FILTERED = snakemake.input.prevalence_filtered
HIGH_CONF = snakemake.input.high_confidence_mirnas
CHANGING = snakemake.input.changing_mirnas
CV_DIR = Path(snakemake.params.cv_dir)
EVAL_DIR = Path(snakemake.params.eval_dir)
OUTPUT = snakemake.output.workflow

# model name -> (label, feature set, hold-out evaluation file)
MODELS = {
    "linear_regression": ("Linear regression", "age-correlated", "linear_regression_eval_train_stats.txt"),
    "linear_regression_pol2": ("Linear regression, poly. deg. 2", "age-correlated", "linear_regression_pol2_eval_train_stats.txt"),
    "lasso": ("Lasso", "age-correlated", "lasso_eval_train_stats.txt"),
    "linear_regression_all": ("Linear regression", "all", "linear_regression_eval_all_train_stats.txt"),
    "lasso_all": ("Lasso", "all", "lasso_all_eval_train_stats.txt"),
    "hgb": ("Hist. gradient boosting", "all", "hgb_eval_train_stats.txt"),
}

COLORS = {
    "input": "#dbe4f0",
    "filter": "#d6ecf3",
    "split": "#fbe3c7",
    "features": "#d9efd8",
    "model": "#e6dcf1",
    "output": "#f6d5d5",
    "explore": "#eeeeee",
}
EDGE = "#4a4a4a"


def shape(path, with_projects=True):
    adata = ad.read_h5ad(path, backed="r")
    n_projects = adata.obs["Project"].nunique() if with_projects else None
    shape = adata.shape
    adata.file.close()
    return shape, n_projects


def describe(path, with_projects=True):
    (n_samples, n_mirnas), n_projects = shape(path, with_projects)
    text = f"{n_samples:,} samples × {n_mirnas:,} miRNAs"
    return text + (f" · {n_projects} projects" if with_projects else "")


def count_lines(path):
    with open(path) as file:
        return sum(1 for line in file if line.strip())


def read_metric(text, pattern):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else float("nan")


def model_metrics(model):
    holdout = (EVAL_DIR / MODELS[model][2]).read_text()
    cv = (CV_DIR / f"{model}_summary.txt").read_text()
    return {
        "test_mae": read_metric(holdout, r"Test MAE:\s*([-\d.]+)"),
        "test_r2": read_metric(holdout, r"Test R2:\s*([-\d.]+)"),
        "cv_mae": read_metric(cv, r"\n\s+MAE:\s*([-\d.]+)"),
        "cv_mae_sd": read_metric(cv, r"\n\s+MAE:\s*[-\d.]+ \+- ([\d.]+)"),
        "cv_r2": read_metric(cv, r"\n\s+R2:\s*([-\d.]+)"),
    }


def box(ax, x, y, w, h, title, body, kind, body_size=9, family=None):
    """Box with top-left corner (x, y); y grows downwards."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor=COLORS[kind], edgecolor=EDGE, linewidth=1))
    ax.text(x + w / 2, y + 0.18, title, ha="center", va="top", fontsize=10.5, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + 0.55, body, ha="center", va="top", multialignment="left" if family else "center",
                fontsize=body_size, linespacing=1.45, family=family)
    return {"top": (x + w / 2, y), "bottom": (x + w / 2, y + h), "left": (x, y + h / 2),
            "right": (x + w, y + h / 2)}


def arrow(ax, start, end, style="-|>", dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, color=EDGE,
                                 linewidth=1.2, linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def results_table(metrics):
    header = f"{'Model':<34}{'Features':<16}{'Hold-out MAE':>13}{'R²':>7}   {'Nested CV MAE':>15}{'R²':>7}"
    lines = [header, "─" * len(header)]
    best = min(metrics, key=lambda m: metrics[m]["cv_mae"])
    for model, (label, features, _) in MODELS.items():
        m = metrics[model]
        marker = " ◀" if model == best else ""
        lines.append(f"{label:<34}{features:<16}{m['test_mae']:>13.2f}{m['test_r2']:>7.2f}   "
                     f"{m['cv_mae']:>8.2f} ± {m['cv_mae_sd']:.2f}{m['cv_r2']:>7.2f}{marker}")
    return "\n".join(lines)


def main():
    (raw_samples, raw_mirnas), _ = shape(RAW, with_projects=False)
    (n_train, _), _ = shape(TRAIN)
    (n_test, _), _ = shape(TEST)
    (_, n_prevalent), _ = shape(PREVALENCE_FILTERED)
    n_age_filtered = shape(AGE_FILTERED)[0][0]
    n_changing = count_lines(CHANGING) - 1
    metrics = {model: model_metrics(model) for model in MODELS}

    fig, ax = plt.subplots(figsize=(14, 21.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(21.5, 0)
    ax.axis("off")

    ax.text(7, 0.2, "Workflow: predicting age from blood miRNA expression", ha="center", va="top",
            fontsize=15, fontweight="bold")

    # --- Inputs ---
    meta = box(ax, 0.3, 0.9, 3.6, 1.3, "Sample metadata", "human_combined.tsv\nAge, disease, project, seq. method", "input")
    atlas = box(ax, 4.4, 0.9, 5.2, 1.3, "miRNA expression atlas",
                f"Harmonized RPMM\n{raw_samples:,} samples × {raw_mirnas:,} miRNAs", "input")
    mirbase = box(ax, 10.1, 0.9, 3.6, 1.3, "miRBase high confidence",
                  f"{count_lines(HIGH_CONF):,} mature miRNAs", "input")

    # --- Main filtering column ---
    clean = box(ax, 3.5, 2.9, 7.0, 1.6, "1  Cleaning",
                "Keep high-confidence miRNAs · join metadata\ndrop missing annotations and unknown sex\n"
                + describe(CLEANED), "filter")
    for source, rad in [(meta["bottom"], 0.15), (atlas["bottom"], 0), (mirbase["bottom"], -0.15)]:
        arrow(ax, source, clean["top"], rad=rad)

    blood = box(ax, 3.5, 5.1, 7.0, 1.6, "2  Healthy blood samples",
                f"Tissue = blood ({describe(BLOOD, with_projects=False).split(' ×')[0]})"
                " · disease condition = healthy\ndrop miRNAs never detected in blood\n" + describe(BLOOD_HEALTHY),
                "filter")
    arrow(ax, clean["bottom"], blood["top"])

    age = box(ax, 3.5, 7.3, 7.0, 1.3, "3  Age range 30–80 years", describe(AGE_FILTERED), "filter")
    arrow(ax, blood["bottom"], age["top"])

    explore = box(ax, 10.9, 5.1, 2.8, 3.5, "Exploratory analyses",
                  "UMAPs (tissue, project,\nseq. method, age)\nAge & sex distributions\nmiRNA prevalence\n"
                  "Variance partitioning\n(LMM: age, sex, project)", "explore", body_size=8.5)
    arrow(ax, blood["right"], explore["left"], dashed=True)

    # --- Legend ---
    legend = [("input", "Input data"), ("filter", "Sample filtering"), ("split", "Data splitting"),
              ("features", "Feature processing"), ("model", "Models"), ("output", "Predictions"),
              ("explore", "Exploration")]
    for i, (kind, label) in enumerate(legend):
        y = 5.15 + i * 0.48
        ax.add_patch(FancyBboxPatch((0.4, y), 0.4, 0.3, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=COLORS[kind], edgecolor=EDGE, linewidth=0.8))
        ax.text(0.95, y + 0.15, label, va="center", fontsize=9)

    # --- Track A: hold-out evaluation ---
    split = box(ax, 0.5, 9.2, 6.6, 1.3, "A  Hold-out: stratified train/test split (90/10)",
                f"strata = project × age bin\ntrain: {n_train} samples · test: {n_test} samples", "split")
    arrow(ax, age["bottom"], split["top"], rad=0.1)

    prevalence = box(ax, 0.5, 11.0, 6.6, 1.3, "Prevalence filter + log transform",
                     f"RPMM > 1 in ≥ 20 % of training samples\nlog2(1 + RPMM) → {n_prevalent} miRNAs", "features")
    arrow(ax, split["bottom"], prevalence["top"])

    feature_sets = box(ax, 0.5, 12.8, 6.6, 1.6, "Two feature sets",
                       f"age-correlated: {n_changing} miRNAs\n(Pearson or Spearman p < 0.05, |r| ≥ 0.15)\n"
                       f"all: {n_prevalent} miRNAs", "features")
    arrow(ax, prevalence["bottom"], feature_sets["top"])

    scaling = box(ax, 0.5, 14.9, 6.6, 1.4, "Scaling & tuning",
                  "per-project z-score (training-set stats)\nYeo–Johnson on skewed miRNAs · one-hot sex\n"
                  "SelectKBest + GridSearchCV (5-fold, MAE)", "features", body_size=8.5)
    arrow(ax, feature_sets["bottom"], scaling["top"])

    # --- Track B: repeated nested CV ---
    outer = box(ax, 7.6, 9.2, 5.6, 1.3, "B  Repeated nested CV: outer loop",
                f"all {n_age_filtered} samples · RepeatedKFold 5 folds × 10 repeats\n→ performance estimate", "split")
    arrow(ax, age["bottom"], outer["top"], rad=-0.1)

    inner = box(ax, 7.6, 11.0, 5.6, 5.3, "Refit inside every training fold",
                "prevalence filter + log2\n↓\nper-project z-score\n↓\nage-correlation filter\n"
                "(correlated feature set only)\n↓\nYeo–Johnson · one-hot sex\n↓\nSelectKBest\n\n"
                "Inner loop: GridSearchCV (5-fold)\n→ hyperparameters", "features")
    arrow(ax, outer["bottom"], inner["top"])

    # --- Models ---
    models = box(ax, 0.5, 16.8, 12.7, 1.3, "Models",
                 "age-correlated features: linear regression · polynomial (deg. 2) regression · Lasso\n"
                 "all features: linear regression · Lasso · histogram gradient boosting", "model")
    arrow(ax, scaling["bottom"], (3.8, 16.8))
    arrow(ax, inner["bottom"], (10.4, 16.8))

    # --- Predictions ---
    predictions = box(ax, 0.5, 18.6, 12.7, 2.7, "Predicted age · error in years (MAE, lower is better)",
                      results_table(metrics), "output", body_size=8.5, family="DejaVu Sans Mono")
    arrow(ax, models["bottom"], predictions["top"])

    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)


main()
