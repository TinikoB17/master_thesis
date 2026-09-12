import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.utils.class_weight import compute_class_weight


DATA_PATH = snakemake.input.model_input
TEST_DATA_PATH = snakemake.input.model_test

EVALUATION = snakemake.output.linear_regression_eval

def load_training_data():
    data = pd.read_csv(DATA_PATH, sep="\t").set_index("Sample")
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

    param_grid = {
        "feature_selection__k": [50, 100, "all"],
        "model": [LinearRegression()],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    # search.fit(X, y, model__sample_weight=sample_weights)
    search.fit(X, y)

    best_model = search.best_estimator_
    best_params = f"Best Params: {search.best_params_}, Best CV MAE: {-search.best_score_} \n"

    with open(EVALUATION, "a") as file:
        file.write(best_params)

    
    print("Best params:", search.best_params_)
    print("Best CV MAE:", -search.best_score_)

    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred_cv, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--", color="darkorange")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Linear Regression / Ridge - CV predictions")
    plt.tight_layout()
    plt.savefig("linear_regression_tuned_cv_prediction.svg")
    plt.close()

    test_data = pd.read_csv(TEST_DATA_PATH, sep="\t").set_index("Sample")
    X_test = test_data.drop(columns=["Age"])
    y_test = test_data["Age"]

    y_pred = best_model.predict(X_test)
    print("Test R2:", r2_score(y_test, y_pred))
    print("Test MAE:", mean_absolute_error(y_test, y_pred))
    print("Test RMSE:", root_mean_squared_error(y_test, y_pred))
    print("Test MedianAE:", median_absolute_error(y_test, y_pred))

    eval_df = pd.DataFrame({
        "actual_age": y_test,
        "predicted_age": y_pred,
        "error": np.abs(y_test - y_pred)
    })

    bins = [30, 40, 50, 60, 70, 80]

    eval_df["age_group"] = pd.cut(eval_df["actual_age"], bins=bins)
    perf_by_group = (eval_df.groupby("age_group", observed=False)
                     .agg(Sample_Count=("error", "count"),
                          MAE=("error", "mean"),
                          Median_AE=("error", "median"),)
                          .reset_index())

    print(perf_by_group)
    with open(EVALUATION, "a") as file:
        file.write(
            f"Test R2: {r2_score(y_test, y_pred)}, Test MAE: {mean_absolute_error(y_test, y_pred)}, Test RMSE: {root_mean_squared_error(y_test, y_pred)}")

if __name__ == "__main__":
    main()
