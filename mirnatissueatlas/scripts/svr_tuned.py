import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from metrics import AGE_BINS, balanced_mae, balanced_mae_scorer

DATA_PATH = snakemake.input.model_input
TEST_DATA_PATH = snakemake.input.model_test

EVALUATION = snakemake.output.svr_eval
MODEL = snakemake.output.svr_model

# Optional: restrict the features to the miRNAs in a table with a "miRNA" column, e.g. the
# LASSO selection; if the table has a "significant" column (the age-group differential
# expression results), only the rows flagged significant are used
MIRNA_TABLE = snakemake.input.get("mirna_table")


def select_features(data: pd.DataFrame) -> pd.DataFrame:
    if MIRNA_TABLE is None:
        return data
    table = pd.read_csv(MIRNA_TABLE, sep=None, engine="python")
    if "significant" in table.columns:
        table = table[table["significant"]]
    mirnas = table["miRNA"].tolist()
    missing = [mirna for mirna in mirnas if mirna not in data.columns]
    if missing:
        raise ValueError(f"miRNAs from {MIRNA_TABLE} not in the model input: {missing}")
    print(f"Using {len(mirnas)} miRNAs from {MIRNA_TABLE}")
    return data[["Age", "Sex"] + mirnas]


def load_training_data():
    data = select_features(pd.read_csv(DATA_PATH, sep="\t").set_index("Sample"))
    X = data.drop(columns=["Age"])
    y = data["Age"]
    return X, y


def build_preprocessor(X: pd.DataFrame):
    categorical_columns = ["Sex"]
    numeric_columns = [col for col in X.columns if col not in categorical_columns]
    skew_values = X[numeric_columns].skew()
    skewed_columns = skew_values[skew_values > 1].index.tolist()
    symmetric_columns = [col for col in numeric_columns if col not in skewed_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("power_transformer", PowerTransformer(method="yeo-johnson", standardize=False), skewed_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
            ("num_passthrough", "passthrough", symmetric_columns),
        ],
        remainder="drop",
    )

    return preprocessor


def main():
    X, y = load_training_data()

    preprocessor = build_preprocessor(X)

    # SVR is distance-based, so every feature is rescaled after the power transform
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            ("scale", StandardScaler()),
            ("model", SVR()),
        ]
    )

    # SVR has no built-in feature selection, so k is tuned as well (smaller values when the
    # miRNAs are already pre-selected).
    # epsilon is in years: errors smaller than epsilon are not penalized.
    k_values = [25, 50, "all"] if MIRNA_TABLE else [100, 200, "all"]
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
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=balanced_mae_scorer,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(X, y)
    best_model = search.best_estimator_
    best_params = f"Best Params: {search.best_params_}, Best CV balanced MAE: {-search.best_score_} \n"

    joblib.dump(best_model, MODEL)
    print(f"Model saved at {MODEL}")

    with open(EVALUATION, "a") as file:
        file.write(best_params)
    print("Best params:", search.best_params_)
    print("Best CV balanced MAE:", -search.best_score_)

    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred_cv, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--", color="darkorange")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("SVR - CV predictions")
    plt.tight_layout()
    plt.savefig(snakemake.output.cv_plot)
    plt.close()

    test_data = select_features(pd.read_csv(TEST_DATA_PATH, sep="\t").set_index("Sample"))
    X_test = test_data.drop(columns=["Age"])
    y_test = test_data["Age"]

    y_pred = best_model.predict(X_test)
    print("Test R2:", r2_score(y_test, y_pred))
    print("Test MAE:", mean_absolute_error(y_test, y_pred))
    print("Test balanced MAE:", balanced_mae(y_test, y_pred))
    print("Test RMSE:", root_mean_squared_error(y_test, y_pred))
    print("Test MedianAE:", median_absolute_error(y_test, y_pred))

    low = min(y_test.min(), y_pred.min())
    high = max(y_test.max(), y_pred.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([low, high], [low, high], linestyle="--", color="darkorange")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title(
        f"SVR - Test predictions\n"
        f"MAE = {mean_absolute_error(y_test, y_pred):.2f}, R² = {r2_score(y_test, y_pred):.2f}"
    )
    plt.tight_layout()
    plt.savefig(snakemake.output.test_plot)
    plt.close()

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

    print(perf_by_group)

    with open(EVALUATION, "a") as file:
        file.write(
            f"Test R2: {r2_score(y_test, y_pred)}, \n Test MAE: {mean_absolute_error(y_test, y_pred)}, \n Test RMSE: {root_mean_squared_error(y_test, y_pred)}, \n Test balanced MAE: {balanced_mae(y_test, y_pred)}")


if __name__ == "__main__":
    main()
