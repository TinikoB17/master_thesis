import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_utils import load, load_groups, plot_predictions, write_evaluation, write_predictions

TRAIN = snakemake.input.train
HOLDOUT = snakemake.input.holdout
METADATA = snakemake.input.metadata
GROUP = snakemake.params.group
ALPHAS = np.logspace(*snakemake.params.alpha_range, snakemake.params.n_alphas)

MODEL = snakemake.output.model
EVALUATION = snakemake.output.evaluation
SELECTED = snakemake.output.selected_features
PREDICTIONS = snakemake.output.predictions
CV_PLOT = snakemake.output.cv_plot
HOLDOUT_PLOT = snakemake.output.holdout_plot


def main():
    X, y = load(TRAIN)
    X_holdout, y_holdout = load(HOLDOUT)

    # LASSO selects the features itself: the L1 penalty sets the coefficients of the
    # uninformative miRNAs to exactly 0; alpha (the penalty strength) is chosen by CV
    pipeline = Pipeline(steps=[
        ("scale", StandardScaler()),
        ("model", Lasso(max_iter=100000, random_state=42)),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, param_grid={"model__alpha": ALPHAS},
                          scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, refit=True)
    search.fit(X, y)
    best_model = search.best_estimator_
    joblib.dump(best_model, MODEL)

    # Number of non-zero coefficients for each alpha, fitted on the whole training set
    scaled = StandardScaler().fit_transform(X)
    n_features = [int((Lasso(alpha=alpha, max_iter=100000).fit(scaled, y).coef_ != 0).sum())
                  for alpha in ALPHAS]
    cv_results = pd.DataFrame({
        "alpha": ALPHAS,
        "n_features": n_features,
        "CV_MAE": -search.cv_results_["mean_test_score"],
        "CV_MAE_sd": search.cv_results_["std_test_score"],
    })

    # Non-zero coefficients (per standard deviation of the feature), largest first
    coefficients = pd.Series(best_model.named_steps["model"].coef_, index=X.columns)
    selected = (coefficients[coefficients != 0].rename("coefficient").rename_axis("feature")
                .reset_index())
    selected = selected.reindex(selected["coefficient"].abs().sort_values(ascending=False).index)
    selected.to_csv(SELECTED, sep="\t", index=False)

    y_cv = cross_val_predict(best_model, X, y, cv=cv)
    y_pred = best_model.predict(X_holdout)
    write_predictions(PREDICTIONS, y_holdout, y_pred)

    best_alpha = search.best_params_["model__alpha"]
    header = (f"{GROUP}: StandardScaler -> Lasso\n"
              f"Train: {len(y)} samples, holdout: {len(y_holdout)} samples, "
              f"{X.shape[1]} candidate features\n\n"
              f"Best alpha: {best_alpha:.4g} ({len(selected)} non-zero coefficients)")
    write_evaluation(EVALUATION, header, cv_results, y, y_cv, y_holdout, y_pred,
                     groups=load_groups(METADATA, y_holdout.index),
                     train_groups=load_groups(METADATA, y.index))

    plot_predictions(y, y_cv, f"LASSO, {GROUP} - CV predictions", CV_PLOT)
    plot_predictions(y_holdout, y_pred, f"LASSO, {GROUP} - holdout", HOLDOUT_PLOT)


if __name__ == "__main__":
    main()
