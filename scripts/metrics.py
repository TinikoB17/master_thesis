import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

# Age bins (years) used for the balanced MAE; same bins as the per-age-group evaluation
AGE_BINS = [30, 40, 50, 60, 70, 80]


def balanced_mae(y_true, y_pred, bins=AGE_BINS) -> float:
    """Mean of the per-age-bin MAEs, so every age bin counts equally regardless of its size.
    Empty bins are skipped (small validation folds may lack the youngest or oldest samples)."""
    errors = pd.Series(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
    age_bins = pd.cut(np.asarray(y_true), bins=bins, include_lowest=True)
    return float(errors.groupby(age_bins, observed=True).mean().mean())


# For GridSearchCV: scorers are maximized, so the balanced MAE is negated
balanced_mae_scorer = make_scorer(balanced_mae, greater_is_better=False)
