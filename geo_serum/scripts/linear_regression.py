import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_utils import feature_selector, load, load_groups, plot_predictions, write_evaluation, write_predictions

TRAIN = snakemake.input.train
HOLDOUT = snakemake.input.holdout
METADATA = snakemake.input.metadata
GROUP = snakemake.params.group
K_VALUES = snakemake.params.k_values

MODEL = snakemake.output.model
EVALUATION = snakemake.output.evaluation
SELECTED = snakemake.output.selected_features
PREDICTIONS = snakemake.output.predictions
CV_PLOT = snakemake.output.cv_plot
HOLDOUT_PLOT = snakemake.output.holdout_plot


def main():
    X, y = load(TRAIN)
    X_holdout, y_holdout = load(HOLDOUT)

    pipeline = Pipeline(steps=[
        # ("scale", StandardScaler()),
        ("feature_selection", feature_selector(X)),  # cohort columns always kept
        ("model", LinearRegression()),
    ])

    # The number of features is chosen by cross-validation on the training set only, from the
    # k values set in the Snakefile (those below the number of candidate features)
    k_values = [k for k in K_VALUES if k < X.shape[1]]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, param_grid={"feature_selection__k": k_values},
                          scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, refit=True)
    search.fit(X, y)
    best_model = search.best_estimator_
    joblib.dump(best_model, MODEL)

    cv_results = pd.DataFrame({
        "k": search.cv_results_["param_feature_selection__k"],
        "CV_MAE": -search.cv_results_["mean_test_score"],
        "CV_MAE_sd": search.cv_results_["std_test_score"],
    })

    # Selected features with their coefficients (per standard deviation of the feature)
    selector = best_model.named_steps["feature_selection"]
    selected = pd.DataFrame({
        "feature": X.columns[selector.get_support()],
        "coefficient": best_model.named_steps["model"].coef_,
        "f_score": selector.scores_[selector.get_support()],
    }).sort_values("f_score", ascending=False)
    selected.to_csv(SELECTED, sep="\t", index=False)

    y_cv = cross_val_predict(best_model, X, y, cv=cv)
    y_pred = best_model.predict(X_holdout)
    write_predictions(PREDICTIONS, y_holdout, y_pred)

    header = (f"{GROUP}: StandardScaler -> SelectKBest(f_regression) -> LinearRegression\n"
              f"Train: {len(y)} samples, holdout: {len(y_holdout)} samples, "
              f"{X.shape[1]} candidate features\n\n"
              f"Best k: {search.best_params_['feature_selection__k']} ({len(selected)} features)")
    write_evaluation(EVALUATION, header, cv_results, y, y_cv, y_holdout, y_pred,
                     groups=load_groups(METADATA, y_holdout.index),
                     train_groups=load_groups(METADATA, y.index))

    plot_predictions(y, y_cv, f"Linear regression, {GROUP} - CV predictions", CV_PLOT)
    plot_predictions(y_holdout, y_pred, f"Linear regression, {GROUP} - holdout", HOLDOUT_PLOT)


if __name__ == "__main__":
    main()
