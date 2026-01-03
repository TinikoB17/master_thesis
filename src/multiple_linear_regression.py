from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.pipeline import Pipeline
import scanpy as sc
import numpy as np


import scripts.model_preprocessing as model_preprocessing

anndata = sc.read_h5ad("./filtered_input/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad")
dataset = model_preprocessing.preprocess_data(anndata, ["Project", "Sequencing_Method", "Age"])

X = dataset.drop(columns=["Age"])
y = dataset["Age"]

categorical_columns = ["Sequencing_Method", "Project"]

model = LinearRegression()

column_transformer = ColumnTransformer([('encode_cats', OneHotEncoder(handle_unknown='ignore'), categorical_columns)], remainder="passthrough")
print(X)
print(y)
pipeline = Pipeline(
    [("preprocessing", column_transformer),
     ('var', VarianceThreshold(threshold=0.3)),
     ("feature_selection", SelectKBest(f_regression, k=500)),
     ("model", model)]
)

scores = cross_val_score(pipeline, X, y, cv=5, scoring="explained_variance")
print(np.mean(scores))

print(scores)
