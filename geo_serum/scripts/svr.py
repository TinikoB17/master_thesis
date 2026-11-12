import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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

N_BEST_SETTINGS = 15  # rows of the tuning results in the evaluation report


def main():
    X, y = load(TRAIN)
    X_holdout, y_holdout = load(HOLDOUT)

    # SVR is distance-based, so the features are standardised; it has no built-in feature
    # selection, so the number of miRNAs (k) is tuned with the SVR settings
    pipeline = Pipeline(steps=[
        ("scale", StandardScaler()),
        ("feature_selection", feature_selector(X)),  # cohort columns always kept
        ("model", SVR()),
    ])

    # Same grid as the miRNATissueAtlas pipeline (mirnatissueatlas/scripts/svr_tuned.py); epsilon is in years:
    # errors smaller than epsilon are not penalised
    # k values set in the Snakefile (those below the number of candidate features)
    k_values = [k for k in K_VALUES if k < X.shape[1]]
    param_grid = [
        {
            "feature_selection__k": k_values,
            "model__kernel": ["rbf"],
            "model__C": [30, 100, 300, 1000, 3000],
            "model__gamma": ["scale", 3e-3, 1e-3, 3e-4],
            "model__epsilon": [0.1, 0.5, 1.0],
        },
        {
            "feature_selection__k": k_values,
            "model__kernel": ["linear"],
            "model__C": [0.001, 0.01, 0.1],
            "model__epsilon": [0.1, 0.5, 1.0],
        },
    ]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, param_grid=param_grid, scoring="neg_mean_absolute_error",
                          cv=cv, n_jobs=-1, refit=True)
    search.fit(X, y)
    best_model = search.best_estimator_
    joblib.dump(best_model, MODEL)

    params = pd.DataFrame(search.cv_results_["params"])
    params.columns = [column.split("__")[1] for column in params.columns]
    cv_results = (params.assign(CV_MAE=-search.cv_results_["mean_test_score"],
                                CV_MAE_sd=search.cv_results_["std_test_score"])
                  .sort_values("CV_MAE").head(N_BEST_SETTINGS))
    cv_results["gamma"] = cv_results["gamma"].fillna("-")

    # Features kept by SelectKBest; a linear kernel also has one coefficient per feature
    selector = best_model.named_steps["feature_selection"]
    selected = pd.DataFrame({
        "feature": X.columns[selector.get_support()],
        "f_score": selector.scores_[selector.get_support()],
    })
    svr = best_model.named_steps["model"]
    if svr.kernel == "linear":
        selected["coefficient"] = svr.coef_.ravel()
    selected.sort_values("f_score", ascending=False).to_csv(SELECTED, sep="\t", index=False)

    y_cv = cross_val_predict(best_model, X, y, cv=cv)
    y_pred = best_model.predict(X_holdout)
    write_predictions(PREDICTIONS, y_holdout, y_pred)

    best = {key.split("__")[1]: value for key, value in search.best_params_.items()}
    header = (f"{GROUP}: StandardScaler -> SelectKBest(f_regression) -> SVR\n"
              f"Train: {len(y)} samples, holdout: {len(y_holdout)} samples, "
              f"{X.shape[1]} candidate features\n\n"
              f"Best settings: {best} ({len(selected)} features), "
              f"{svr.support_.size} support vectors\n"
              f"Best {N_BEST_SETTINGS} of {len(params)} settings by CV MAE:")
    write_evaluation(EVALUATION, header, cv_results, y, y_cv, y_holdout, y_pred,
                     groups=load_groups(METADATA, y_holdout.index),
                     train_groups=load_groups(METADATA, y.index))

    plot_predictions(y, y_cv, f"SVR, {GROUP} - CV predictions", CV_PLOT)
    plot_predictions(y_holdout, y_pred, f"SVR, {GROUP} - holdout", HOLDOUT_PLOT)


if __name__ == "__main__":
    main()
