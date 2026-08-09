import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

exp = pd.read_csv("test_output/rpmm_tissueatlas_blood_healthy.csv", sep='\t').set_index('miRNA')

exp = exp.drop(columns = ['SRR8839813'])

metadata = pd.read_csv("test_output/blood_metadata_tissueatlas.csv", sep='\t').set_index("Sample")

metadata = metadata[(metadata["Disease_Condition"] == "Healthy") & (metadata["Project"] == "PPMI")]

exp = exp[metadata.index]

metadata["Sex"] = metadata["Sex"].astype("category")
metadata["Disease"] = metadata["Sequencing_Method"].astype("category")

X = exp.T
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pcs = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_

cmvar = np.cumsum(explained)
n_pc = np.argmax(cmvar >= 0.7) + 1

print(n_pc)

res = []
for i in range(n_pc):
    df = metadata.copy()
    df["PC"] = pcs[:, i]

    model = smf.ols(
        "PC ~ Age + C(Sex)", data=df
    ).fit()

    anova = anova_lm(model, typ=2)

    ss = anova["sum_sq"]
    total = ss.sum()
    frac = ss /total

    frac["PC"] = i + 1
    frac["Weight"] = explained[i]
    res.append(frac)

pvca = pd.concat(res, axis=1).T


variables = ["Age", "C(Sex)", "Residual"]

weighted = {}

for var in variables:
    vals = []
    for i in range(n_pc):
        df = metadata.copy()
        df["PC"] = pcs[:, i]
        model = smf.ols(
            "PC ~ Age + C(Sex)", data=df
        ).fit()

        anova = anova_lm(model, typ=2)

        ss = anova["sum_sq"]
        total = ss.sum()
        frac = ss /total

        if var == "Residual":
            vals.append((1 - frac.sum()) * explained[i])
        else:
            vals.append(frac[var] * explained[i])
    weighted[var] = np.sum(vals)

pvca_result = pd.Series(weighted)
pvca_result /= pvca_result.sum()
breakpoint()
