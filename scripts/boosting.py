import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, TargetEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.linear_model import LassoCV
from sklearn.ensemble import HistGradientBoostingRegressor
# from catboost import CatBoostRegressor

data = pd.read_csv("model_input_data/blood_healthy_age_scaled.csv", sep='\t').set_index("Sample")
data = data[(data["Age"] >= 20) & (data["Age"] <= 86)]

X = data.drop(columns=["Age"])
y = data["Age"]

cat_columns = ["Project", "Sequencing_Method"]

num_cols = [col for col in X.columns if col not in cat_columns]

skew_values = X[num_cols].skew()

skewed_cols = skew_values[skew_values > 1].index.tolist()
symmetric_cols = [col for col in num_cols if col not in skewed_cols]

pt = PowerTransformer(method="yeo-johnson", standardize=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("power_transformer", pt, skewed_cols),
        ("cat", TargetEncoder(smooth="auto", cv=5, random_state=42), cat_columns),
        ("num", "passthrough", symmetric_cols),
        #  ("cat", "passthrough", cat_columns)
    ]
)

# cat_indices = [X.columns.get_loc(c) for c in cat_columns]

hgb = HistGradientBoostingRegressor(
    max_iter=1200,
    learning_rate=0.05,
    max_depth=9,
    min_samples_leaf=50,
    l2_regularization=3.0,
    max_bins=128,
    random_state=42
)
model = Pipeline([
    ("preprocess", preprocessor),
    ("histgradientboost", hgb)
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Metrics
mae_scores = -cross_val_score(model, X, y, cv=kf,
                              scoring=make_scorer(mean_absolute_error, greater_is_better=False))
rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf,
                                       scoring="neg_mean_squared_error"))
r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

print("MAE:", mae_scores.mean(), "+/-", mae_scores.std() )

print("RMSE:", rmse_scores.mean(), "+/-", rmse_scores.std())
print("R2:", r2_scores.mean(), "+/-", r2_scores.std())

