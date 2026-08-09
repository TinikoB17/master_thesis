from sklearn.svm import SVR
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.linear_model import Ridge


data = pd.read_csv("model_input_data/blood_healthy_age_scaled.csv", sep='\t').set_index("Sample")
# data = pd.read_csv("model_input_data/filtered_tsi.csv", sep='\t').set_index("Sample")


# deconvoluted = pd.read_csv("deconvoluted_tissueatlas.csv", sep='\t')

# new_cols = deconvoluted.columns.to_list()

# data[new_cols] = deconvoluted[new_cols]

data = data[(data["Age"] >= 20) & (data["Age"] <= 86)]

X = data.drop(columns=["Age", "Project"])
y = data.drop(columns=["Project"])["Age"]

cat_columns = ["Sex", "Sequencing_Method"]

num_cols = [col for col in X.columns if col not in cat_columns]
# num_cols = [col for col in X.columns if col not in cat_columns and col not in new_cols]

skew_values = X[num_cols].skew()
skewed_cols = skew_values[skew_values > 1].index.tolist()

symmetric_cols = [col for col in num_cols if col not in skewed_cols]

pt = PowerTransformer(method="yeo-johnson", standardize=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("power_transformer", pt, skewed_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_columns),
        ("num", "passthrough", symmetric_cols),
        #  ("cat", "passthrough", cat_columns)
    ]
)
# cat_indices = [X.columns.get_loc(c) for c in cat_columns]
ridge = Ridge(alpha=0.7)
model = Pipeline([
    ("preprocess", preprocessor),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),

    ("ridge", ridge)
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

