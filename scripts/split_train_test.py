import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split


h5ad_to_process = snakemake.input.rpmm_filtered_annotated
train = snakemake.output.blood_healthy_train
test = snakemake.output.blood_healthy_test
expression_train = snakemake.output.expression_train
metadata_train = snakemake.output.metadata_train
expression_test = snakemake.output.expression_test
metadata_test = snakemake.output.metadata_test

to_process = sc.read_h5ad(h5ad_to_process)
print(to_process.obs)

#filtering out projects with 2 or less occurences for stratified sampling

to_process.obs["age_bin"] = pd.cut(to_process.obs["Age"], bins=2)
to_process.obs["strata"] = (to_process.obs["Project"].astype(str) + "_" + to_process.obs["age_bin"].astype(str)) 

train_id, test_id = train_test_split(to_process.obs_names.to_numpy(),
                                     test_size=0.1,
                                     random_state=42,
                                     stratify=to_process.obs["strata"])

adata_train = to_process[train_id].copy()
adata_test = to_process[test_id].copy()
print(adata_train.obs["Age"].min())
print(adata_test.obs)

adata_train.obs.drop(columns=["age_bin", "strata"], inplace=True)
adata_test.obs.drop(columns=["age_bin", "strata"], inplace=True)


adata_train.write_h5ad(train)
adata_test.write_h5ad(test)

adata_train.to_df().to_csv(expression_train, sep='\t', index_label= "Sample")
adata_train.obs.to_csv(metadata_train, sep='\t', index_label= "Sample")

adata_test.to_df().to_csv(expression_test, sep='\t', index_label= "Sample")
adata_test.obs.to_csv(metadata_test, sep='\t', index_label= "Sample")
