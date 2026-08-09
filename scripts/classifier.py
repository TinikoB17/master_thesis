from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
data = pd.read_csv("model_input_data/blood_healthy_age_scaled.csv", sep='\t').set_index("Sample")

data = data[(data["Age"] >= 20) & (data["Age"] <= 86)]

data = data.drop(columns=["Age", "Project"])

print(data)

X = data.drop(columns=["Decade"])
y = data["Decade"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_columns = ["Sequencing_Method", "Sex"]

num_cols = [col for col in X.columns if col not in cat_columns]

skew_values = X[num_cols].skew()

skewed_cols = skew_values[skew_values > 1].index.tolist()
symmetric_cols = [col for col in num_cols if col not in skewed_cols]

pt = PowerTransformer(method="yeo-johnson", standardize=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("power_transformer", pt, skewed_cols),
        ("cat", OneHotEncoder(), cat_columns),
        ("num", "passthrough", symmetric_cols)
    ]
)

svm_ovo =  SVC(kernel="rbf", C=1.0, gamma="scale")

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ('model', svm_ovo)
])
pipeline.fit(X_train, y_train)
# Predict and evaluate the One-vs-One model
y_pred_ovo = pipeline.predict(X_test)
print("One-vs-One Accuracy:", accuracy_score(y_test, y_pred_ovo))
print("One-vs-One Precision:", precision_score(y_test, y_pred_ovo, average='weighted'))

