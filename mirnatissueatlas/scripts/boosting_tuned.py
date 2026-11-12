import numpy as np
import pandas as pd
import joblib
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.utils.class_weight import compute_class_weight

from metrics import AGE_BINS, balanced_mae, balanced_mae_scorer


DATA_PATH = snakemake.input.model_input
TEST_DATA_PATH = snakemake.input.model_test

EVALUATION = snakemake.output.hgb_eval
MODEL = snakemake.output.hgb_model

def load_training_data():
    data = pd.read_csv(DATA_PATH, sep="\t").set_index("Sample")
    metadata_cols = ["Age", "Sex"]

    mirna_cols = [col for col in data.columns if col not in metadata_cols]
    print(len(mirna_cols))
    n_top_features = 512
    top_mirs = (data[mirna_cols].var(numeric_only=True).nlargest(n_top_features).index.tolist())
    to_keep = top_mirs + metadata_cols
    X = data[to_keep].drop(columns=["Age"])
    # X = data[to_keep]
    y = data["Age"]
    return X, y, to_keep


def build_preprocessor(X: pd.DataFrame):
    categorical_columns = ["Sex"]
    numeric_columns = [col for col in X.columns if col not in categorical_columns]
    skew_values = X[numeric_columns].skew()
    skewed_columns = skew_values[skew_values > 1].index.tolist()
    symmetric_columns = [col for col in numeric_columns if col not in skewed_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            # ("power_transformer", PowerTransformer(method="yeo-johnson", standardize=False), skewed_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
            ("num_passthrough", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )

    return preprocessor


def main():
    X, y, to_keep = load_training_data()

    age_bins = pd.cut(y, bins=6, labels=False)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(age_bins), y=age_bins)
    sample_weights = np.array([class_weights[b] for b in age_bins])
    print(age_bins)
    print(class_weights)
    preprocessor = build_preprocessor(X)
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            ("model", HistGradientBoostingRegressor(
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=42)),
        ]
    )

    param_distributions = {
        "feature_selection__k": [50, 100, 200, 'all'],
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_leaf_nodes": randint(7, 32),
        "model__max_features": uniform(0.3, 0.7),
        "model__min_samples_leaf": randint(10, 51),
        "model__l2_regularization": loguniform(1e-3, 10),
        "model__loss": ["absolute_error", "squared_error"],
        "model__max_depth": [3, 4, 5, 6, None]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=150,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        random_state=42
    )

    # search.fit(X, y, model__sample_weight=sample_weights)
    search.fit(X, y)

    best_model = search.best_estimator_

    best_params = f"Best Params: {search.best_params_}, Best CV balanced MAE: {-search.best_score_} \n"

    joblib.dump(best_model, MODEL)
    print(f"Model saved at {MODEL}")

    with open(EVALUATION, "a") as file:
        file.write(best_params)

    print("Best params:", search.best_params_)
    print("Best CV balanced MAE:", -search.best_score_)


    test_data = pd.read_csv(TEST_DATA_PATH, sep="\t").set_index("Sample")
    X_test = test_data[to_keep].drop(columns=["Age"])
    y_test = test_data["Age"]


    y_pred = best_model.predict(X_test)
    print("Test R2:", r2_score(y_test, y_pred))
    print("Test MAE:", mean_absolute_error(y_test, y_pred))
    print("Test balanced MAE:", balanced_mae(y_test, y_pred))
    print("Test RMSE:", root_mean_squared_error(y_test, y_pred))
    print("Test MedianAE:", median_absolute_error(y_test, y_pred))

    eval_df = pd.DataFrame({
        "actual_age": y_test,
        "predicted_age": y_pred,
        "error": np.abs(y_test - y_pred)
    })

    bins = AGE_BINS

    eval_df["age_group"] = pd.cut(eval_df["actual_age"], bins=bins)
    perf_by_group = (eval_df.groupby("age_group", observed=False)
                     .agg(Sample_Count=("error", "count"),
                          MAE=("error", "mean"),
                          Median_AE=("error", "median"),)
                          .reset_index())

    print("Performance By Age Groups:", perf_by_group)

    with open(EVALUATION, "a") as file:
        file.write(
            f"Test R2: {r2_score(y_test, y_pred)}, \n Test MAE: {mean_absolute_error(y_test, y_pred)}, \n Test RMSE: {root_mean_squared_error(y_test, y_pred)}, \n Test balanced MAE: {balanced_mae(y_test, y_pred)}")


if __name__ == "__main__":
    from sklearn.model_selection import cross_val_predict

    main()
