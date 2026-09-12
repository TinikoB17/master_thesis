import pandas as pd
from neuroCombat import neuroCombat
import numpy as np
import scanpy as sc
import anndata as ad

exp = pd.read_csv("test_output/rpmm_tissueatlas_blood_healthy.csv", sep='\t')
print(exp)
# eligible = pd.read_csv("correlations/highest_spearman.csv", sep='\t')["miRNA"].to_list()
exp = exp.set_index("miRNA")
exp = exp.drop(columns=['SRR8839813'])
exp = np.log2(exp + 1)
print(exp)
# metadata = pd.read_csv("PPMI_mod.csv", sep='\t').set_index("Sample")
metadata = pd.read_csv("test_output/blood_metadata_tissueatlas_healthy.csv", sep='\t').set_index("Sample")


metadata = metadata.drop(columns=["Biotype", "Tissue", "Organ_system", "Species", "Disease_Condition"])

categotical_cols = []
continuous_cols = ["Age"]
batch_col = "Project"
print(pd.crosstab(metadata['Project'], metadata['Sex']))
combat_outp = neuroCombat(dat=exp, covars=metadata, batch_col=batch_col, categorical_cols=categotical_cols, continuous_cols=continuous_cols)
exp_corrected = pd.DataFrame(combat_outp['data'], index=exp.index, columns=exp.columns)

# print(exp_corrected)
# exp_corrected.to_csv("test_output/batch_corrected_exp.csv", sep='\t', index='miRNA')
exp_c = exp_corrected.T
exp_c.index.name = "Sample"

adata = ad.AnnData(X=exp_c, obs=metadata)

adata.write_h5ad("test_output/batch_corrected_blood_healthy.h5ad")

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
pl = sc.pl.umap(adata, color="Age", save="corrected_age.svg", title="Colored by Age", cmap="plasma")
