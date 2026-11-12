import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.utils.class_weight import compute_class_weight

from metrics import AGE_BINS, balanced_mae, balanced_mae_scorer


DATA_PATH = snakemake.input.model_input
TEST_DATA_PATH = snakemake.input.model_test

EVALUATION = snakemake.output.linear_regression_eval_all
MODEL = snakemake.output.linear_reg_model_all


# Optional: restrict the features to the miRNAs in a table with a "miRNA" column (e.g. the
# LASSO selection); if it has a "significant" column, only the rows flagged significant
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

    skewed_cols = skew_values[skew_values > 1].index.tolist()
    symmetric_cols = [col for col in numeric_columns if col not in skewed_cols]

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("power_transformer", pt, skewed_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
            ("num", "passthrough", symmetric_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def main():
    X, y = load_training_data()

    age_bins = pd.cut(y, bins=6, labels=False)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(age_bins), y=age_bins)
    sample_weights = np.array([class_weights[b] for b in age_bins])
    preprocessor = build_preprocessor(X)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_regression, k=500)),
            ("model", LinearRegression()),
        ]
    )

    # Only k values below the number of features (Sex becomes two one-hot columns)
    n_features = X.shape[1] + 1
    param_grid = {
        "feature_selection__k": [k for k in [50, 70, 100, 200] if k < n_features] + ["all"],
        "model": [LinearRegression()],
    }

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

    # search.fit(X, y, model__sample_weight=sample_weights)
    search.fit(X, y)

    best_model = search.best_estimator_
    joblib.dump(best_model, MODEL)
    print(f"Model saved at {MODEL}")

    best_params = f"Best Params: {search.best_params_}, Best CV balanced MAE: {-search.best_score_} \n"

    with open(EVALUATION, "a") as file:
        file.write(best_params)

    
    print("Best params:", search.best_params_)
    print("Best CV balanced MAE:", -search.best_score_)


    test_data = select_features(pd.read_csv(TEST_DATA_PATH, sep="\t").set_index("Sample"))
    X_test = test_data.drop(columns=["Age"])
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

    print(perf_by_group)
    with open(EVALUATION, "a") as file:
        file.write(
            f"Test R2: {r2_score(y_test, y_pred)}, Test MAE: {mean_absolute_error(y_test, y_pred)}, Test RMSE: {root_mean_squared_error(y_test, y_pred)}, \n Test balanced MAE: {balanced_mae(y_test, y_pred)}")

if __name__ == "__main__":
    main()
