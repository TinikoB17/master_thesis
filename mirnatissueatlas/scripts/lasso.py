import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, TargetEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LassoCV
# import shap
import matplotlib.pyplot as plt

data = pd.read_csv("model_input_data/blood_healthy_age_scaled_train.csv", sep='\t').set_index("Sample")
data = data.drop(columns=["Project"])

X = data.drop(columns=["Age"])
y = data["Age"]

print(len(X.columns))

cat_columns = ["Sequencing_Method", "Sex"]

num_cols = [col for col in X.columns if col not in cat_columns]

skew_values = X[num_cols].skew()

skewed_cols = skew_values[skew_values > 1].index.tolist()
symmetric_cols = [col for col in num_cols if col not in skewed_cols]

pt = PowerTransformer(method="yeo-johnson", standardize=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("power_transformer", pt, skewed_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_columns),

        # ("cat", TargetEncoder(smooth="auto", cv=5, random_state=42), cat_columns),
        # ("num", "passthrough", symmetric_cols)
    ]
)


alphas = np.logspace(-1, 1, 20)

lasso = LassoCV(alphas=alphas,
                cv=5,
                random_state=42)


pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ('model', lasso)
])


kf = KFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(pipeline, X, y, cv=kf)
print(y_pred.min(), )
plt.scatter(y, y_pred),
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--")

plt.savefig("lasso_pred.svg")
# Metrics
mae_scores = -cross_val_score(pipeline, X, y, cv=kf,
                              scoring=make_scorer(mean_absolute_error, greater_is_better=False))
rmse_scores = np.sqrt(-cross_val_score(pipeline, X, y, cv=kf,
                                       scoring="neg_mean_squared_error"))
r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")

print("MAE:", mae_scores.mean(), "+/-", mae_scores.std() )

print("RMSE:", rmse_scores.mean(), "+/-", rmse_scores.std())
print("R2:", r2_scores.mean(), "+/-", r2_scores.std())

pipeline.fit(X, y)
print(pipeline.named_steps["model"].alpha_)
print(pipeline.named_steps["model"].n_iter_)

#Test data
test = pd.read_csv("model_input_data/blood_healthy_age_scaled_test.csv", sep='\t').set_index("Sample")
X_test = test.drop(columns=["Age", "Project"])
y_test = test.drop(columns=["Project"])["Age"]
print(test)
y_pred = pipeline.predict(X_test)

print(pipeline.score(X_test, y_test))
print(mean_absolute_error(y_test, y_pred))
print(root_mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
