import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.metrics import make_scorer, mean_absolute_error

data = pd.read_csv("model_input_data/blood_healthy_age_scaled.csv", sep='\t').set_index("Sample")
data = data[(data["Age"] >= 20) & (data["Age"] <= 86)]

X = data.drop(columns=["Age"])
y = data["Age"]

cat_columns = ["Project", "Sequencing_Method"]

num_cols = [col for col in X.columns if col not in cat_columns]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_columns),
        ("num", "passthrough", num_cols)
    ]
)

model = LinearRegression()
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ('model', model)
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Metrics
mae_scores = -cross_val_score(pipeline, X, y, cv=kf,
                              scoring=make_scorer(mean_absolute_error, greater_is_better=False))
rmse_scores = np.sqrt(-cross_val_score(pipeline, X, y, cv=kf,
                                       scoring="neg_mean_squared_error"))
r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")

print("MAE:", mae_scores.mean())

print("RMSE:", rmse_scores.mean(), "+/-", rmse_scores.std())
print("R2:", r2_scores.mean(), "+/-", r2_scores.std())
# print(data)

