import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

# Builds the modelling cohort: healthy blood samples with known age and sex, restricted to
# high-confidence miRNAs. Expression stays as raw RPMM; filtering by prevalence, log transform
# and scaling happen inside the CV pipelines (see age_models.py).

cfg = snakemake.config["cohort"]
steps = []


def record(step: str, obs: pd.DataFrame, n_mirnas: int):
    steps.append({"step": step, "samples": len(obs), "projects": obs["Project"].nunique(), "miRNAs": n_mirnas})
    print(f"{step}: {len(obs)} samples, {n_mirnas} miRNAs")


adata = ad.read_h5ad(snakemake.input.expression, backed="r")
record("all samples", adata.obs, adata.n_vars)

with open(snakemake.input.high_confidence_mirnas) as file:
    high_confidence = {line.strip() for line in file if line.strip()}
mirnas = [mirna for mirna in adata.var_names if mirna in high_confidence]
print(f"{len(high_confidence)} high-confidence miRNAs, {len(mirnas)} present in the expression data")

obs = adata.obs
obs = obs[obs["Tissue"] == cfg["tissue"]]
record(f"Tissue == {cfg['tissue']}", obs, len(mirnas))
obs = obs[obs["Biotype"] == cfg["biotype"]]
record(f"Biotype == {cfg['biotype']}", obs, len(mirnas))

# Age, disease and project come from the combined metadata table (the h5ad "Project" is coarser)
metadata = pd.read_csv(snakemake.input.metadata, sep="\t").set_index("Sample")
obs = obs.drop(columns=["Project"]).join(metadata[["Age", "Disease_Condition", "Project", "Sequencing_Method"]], how="inner")
obs = obs[obs["Age"].notna()]
record("in metadata with age", obs, len(mirnas))

healthy_labels = {label.lower() for label in cfg["healthy_labels"]}
obs = obs[obs["Disease_Condition"].astype(str).str.strip().str.lower().isin(healthy_labels)]
record("healthy", obs, len(mirnas))

obs = obs[obs["Age"].between(cfg["min_age"], cfg["max_age"])]
record(f"age {cfg['min_age']}-{cfg['max_age']}", obs, len(mirnas))

obs = obs[obs["Sex"].isin(cfg["sexes"])]
record("sex " + "/".join(cfg["sexes"]), obs, len(mirnas))

cohort = adata[obs.index, mirnas].to_memory()
cohort.obs = obs.copy()
cohort.obs["Disease_Condition_original"] = cohort.obs["Disease_Condition"]
cohort.obs["Disease_Condition"] = "Healthy"
for col in ["Sex", "Project", "Sequencing_Method", "Tissue", "Biotype"]:
    cohort.obs[col] = cohort.obs[col].astype(str).astype("category")
if sparse.issparse(cohort.X):
    cohort.X = cohort.X.toarray()
cohort.X = np.asarray(cohort.X, dtype=np.float32)

cohort = cohort[:, cohort.X.sum(axis=0) > 0].copy()
record("drop miRNAs that are zero in every sample", cohort.obs, cohort.n_vars)

cohort.write_h5ad(snakemake.output.cohort)
pd.DataFrame(steps).to_csv(snakemake.output.steps, sep="\t", index=False)
