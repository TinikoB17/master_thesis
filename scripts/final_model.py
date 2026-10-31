import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from age_models import fit_search, load_cohort, n_features_used

# Refit the model with the lowest repeated-CV MAE (the mean baseline excluded) on all samples,
# tuning its hyperparameters with the same inner CV. Report the miRNAs it uses: coefficients
# for linear models, permutation importance (on the training data) otherwise.

config = snakemake.config
cv_table = pd.read_csv(snakemake.input.cv_table, sep="\t", index_col=0)
model_name = cv_table.drop(index="mean", errors="ignore")["MAE_mean"].idxmin()
print(f"Best model by repeated-CV MAE: {model_name}")

X, y = load_cohort(snakemake.input.cohort)
search = fit_search(model_name, config, X, y, n_jobs=snakemake.threads)
pipeline = search.best_estimator_
joblib.dump(pipeline, snakemake.output.model)

features = pipeline.named_steps["select"].mirnas_ + ["Sex"]
estimator = pipeline.named_steps["model"]
if hasattr(estimator, "coef_"):
    # Inputs are z-scored, so coefficients are years per sd of log2 expression (Sex: m vs f)
    table = pd.DataFrame({"feature": features, "coefficient": estimator.coef_})
    table = table[table["coefficient"] != 0]
    table = table.reindex(table["coefficient"].abs().sort_values(ascending=False).index)
else:
    # Permute the selected features (output of prevalence -> zscore -> select), not raw RPMM
    inputs = pipeline[:3].transform(X)
    result = permutation_importance(pipeline[3:], inputs, y, scoring="neg_mean_absolute_error",
                                    n_repeats=5, random_state=config["cv"]["random_state"],
                                    n_jobs=snakemake.threads)
    table = pd.DataFrame({"feature": features, "importance_mean": result.importances_mean,
                          "importance_std": result.importances_std})
    table = table.sort_values("importance_mean", ascending=False)
table.to_csv(snakemake.output.features, sep="\t", index=False)

in_sample_mae = float(np.mean(np.abs(pipeline.predict(X) - y)))
info = [
    f"Model: {model_name}",
    f"Repeated-CV MAE: {cv_table.loc[model_name, 'MAE_mean']:.3f} +- {cv_table.loc[model_name, 'MAE_std']:.3f}",
    f"Trained on: {len(y)} samples",
    f"Best hyperparameters (inner {config['cv']['inner_splits']}-fold CV): {search.best_params_}",
    f"Inner-CV balanced MAE: {-search.best_score_:.3f}",
    f"miRNAs given to the model: {n_features_used(search)}",
    f"Features with non-zero coefficient: {len(table)}" if hasattr(estimator, "coef_") else
    "Feature ranking: permutation importance on the training data (optimistic)",
    f"In-sample MAE (optimistic, for reference only): {in_sample_mae:.3f}",
]
with open(snakemake.output.info, "w") as file:
    file.write("\n".join(info) + "\n")
print("\n".join(info))
