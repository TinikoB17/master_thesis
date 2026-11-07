import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from analysis_config import CONFIG

# Preprocessing steps for cross-validation pipelines. Each step learns from the training
# fold only (fit) and is then applied to the held-out fold (transform).
# Used by repeated_cv.py and by the standalone experiments (e.g. neural_network_cv.py).

META_COLS = ["Project", "Sex"]


class PrevalenceFilterLog(BaseEstimator, TransformerMixin):
    """Keep miRNAs with RPMM > min_rpmm in >= min_fraction of samples, then log2(1 + x).
    Same filter as visualize_input_data.plot_count_of_miRNAs_at_threshold."""

    def __init__(self, min_rpmm=1.0, min_fraction=0.2):
        self.min_rpmm = min_rpmm
        self.min_fraction = min_fraction

    def fit(self, X, y=None):
        mirna_cols = [col for col in X.columns if col not in META_COLS]
        fraction = (X[mirna_cols] > self.min_rpmm).mean(axis=0)
        self.mirnas_ = fraction[fraction >= self.min_fraction].index.tolist()
        return self

    def transform(self, X):
        out = np.log2(1 + X[self.mirnas_])
        return out.join(X[META_COLS])

    def set_output(self, *, transform=None):
        return self


class ProjectZScore(BaseEstimator, TransformerMixin):
    """Per-project z-score with mean/std learned on the training data.
    Projects unseen in training, or with fewer than min_samples training samples, fall back
    to the global training mean/std (a std from 2-3 samples can be ~0 and blow up the
    z-scores). Drops "Project"."""

    def __init__(self, min_samples=5):
        self.min_samples = min_samples

    def fit(self, X, y=None):
        self.mirnas_ = [col for col in X.columns if col not in META_COLS]
        grouped = X.groupby("Project", observed=True)[self.mirnas_]
        large_enough = grouped.size() >= self.min_samples
        self.means_ = grouped.mean()[large_enough]
        self.stds_ = grouped.std(ddof=1)[large_enough]
        self.global_mean_ = X[self.mirnas_].mean()
        self.global_std_ = X[self.mirnas_].std(ddof=1)
        return self

    def transform(self, X):
        projects = X["Project"].astype(str)
        means = self.means_.reindex(projects).set_index(X.index).fillna(self.global_mean_)
        stds = self.stds_.reindex(projects).set_index(X.index).fillna(self.global_std_)
        stds = stds.mask(stds == 0, 1.0).fillna(1.0)
        out = ((X[self.mirnas_] - means) / stds).fillna(0)
        return out.join(X[["Sex"]])

    def set_output(self, *, transform=None):
        return self


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Keep miRNAs whose Pearson or Spearman correlation with age has p < max_p and
    |r| >= min_abs_corr. Same criteria as calculate_correlations.py (changing_mirnas);
    the default thresholds come from config.yaml."""

    def __init__(self, min_abs_corr=CONFIG["correlation"]["min_abs_corr"],
                 max_p=CONFIG["correlation"]["max_pval"]):
        self.min_abs_corr = min_abs_corr
        self.max_p = max_p

    def fit(self, X, y):
        y = np.asarray(y)
        keep = []
        for mirna in [col for col in X.columns if col != "Sex"]:
            values = X[mirna].to_numpy()
            if np.all(values == values[0]):
                continue
            p_corr, p_pval = pearsonr(values, y)
            s_corr, s_pval = spearmanr(values, y)
            if (p_pval < self.max_p and abs(p_corr) >= self.min_abs_corr) or \
               (s_pval < self.max_p and abs(s_corr) >= self.min_abs_corr):
                keep.append(mirna)
        self.mirnas_ = keep
        return self

    def transform(self, X):
        return X[self.mirnas_ + ["Sex"]]

    def set_output(self, *, transform=None):
        return self


class SkewedPowerTransform(BaseEstimator, TransformerMixin):
    """Yeo-Johnson on miRNAs with skew > 1 (decided on training data), one-hot Sex,
    passthrough for the rest. Same as build_preprocessor() in the model scripts."""

    def __init__(self, power_transform=True):
        self.power_transform = power_transform

    def fit(self, X, y=None):
        numeric = [col for col in X.columns if col != "Sex"]
        skew = X[numeric].skew()
        skewed = skew[skew > 1].index.tolist() if self.power_transform else []
        symmetric = [col for col in numeric if col not in skewed]
        self.transformer_ = ColumnTransformer(
            transformers=[
                ("power_transformer", PowerTransformer(method="yeo-johnson", standardize=False), skewed),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Sex"]),
                ("num", "passthrough", symmetric),
            ],
            remainder="drop",
        ).fit(X, y)
        return self

    def transform(self, X):
        return self.transformer_.transform(X)

    def set_output(self, *, transform=None):
        return self
