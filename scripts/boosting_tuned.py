import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.utils.class_weight import compute_class_weight


DATA_PATH = snakemake.input.model_input
TEST_DATA_PATH = snakemake.input.model_test

EVALUATION = snakemake.output.hgb_eval

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
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]
    )

    param_grid = {
        "feature_selection__k": ['all'],
        "model__max_iter": [1000],
        "model__learning_rate": [0.05],
        "model__max_leaf_nodes": [31], 
        "model__min_samples_leaf": [2],
        "model__l2_regularization": [0.1],
        "model__loss": ["absolute_error"],
        # "model__max_bins": [127, 255],
        "model__max_depth": [8]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=model,
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
    plt.title("HistGradientBoostingRegressor - CV predictions")
    plt.tight_layout()
    plt.savefig("boosting_tuned_cv_prediction.svg")
    plt.close()

    test_data = pd.read_csv(TEST_DATA_PATH, sep="\t").set_index("Sample")
    X_test = test_data[to_keep].drop(columns=["Age"])
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

    print("Performance By Age Groups:", perf_by_group)
if __name__ == "__main__":
    from sklearn.model_selection import cross_val_predict

    main()
