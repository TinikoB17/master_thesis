import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "model_input_data/blood_healthy_age_scaled_train.csv"
TEST_DATA_PATH = "model_input_data/blood_healthy_age_scaled_test.csv"


def load_training_data():
    data = pd.read_csv(DATA_PATH, sep="\t").set_index("Sample")
    X = data.drop(columns=["Age"])
    y = data["Age"]
    return X, y


def build_preprocessor(X: pd.DataFrame):
    categorical_columns = ["Sequencing_Method", "Project", "Sex"]
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
            ("num", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )
    return preprocessor


def main():
    X, y = load_training_data()
    preprocessor = build_preprocessor(X)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("variance_filter", VarianceThreshold(threshold=0.0)),
            ("feature_selection", SelectKBest(f_regression, k=500)),
            ("model", Ridge(random_state=42)),
        ]
    )

    param_grid = {
        "model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "model__fit_intercept": [True, False],
        "feature_selection__k": [200, 400, 500, 800, "all"],
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

    search.fit(X, y)
    best_model = search.best_estimator_

    print("Best params:", search.best_params_)
    print("Best CV MAE:", -search.best_score_)

    test_data = pd.read_csv(TEST_DATA_PATH, sep="\t").set_index("Sample")
    X_test = test_data.drop(columns=["Age"])
    y_test = test_data["Age"]

    y_pred = best_model.predict(X_test)
    print("Test R2:", r2_score(y_test, y_pred))
    print("Test MAE:", mean_absolute_error(y_test, y_pred))
    print("Test RMSE:", root_mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
