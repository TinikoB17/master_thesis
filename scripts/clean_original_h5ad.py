import pandas as pd
import scanpy as sc
from preprocess_disease_condition import alter_disease_condition

annotated_miRNA_path = snakemake.input.original_h5ad
metadata_path = snakemake.input.metadata
high_confidence_mirnas = snakemake.input.high_confidence_mirnas

output_cleaned = snakemake.output.cleaned_data
print(metadata_path)

annotated_miRNA = sc.read_h5ad(annotated_miRNA_path)


with open(high_confidence_mirnas) as f:
    content = f.readlines()

# print(content)

mirnas = [mir.strip() for mir in content]
print(len(mirnas))

valid_mirnas = annotated_miRNA.var_names.intersection(mirnas).tolist()
print(len(valid_mirnas))
annotated_miRNA = annotated_miRNA[:, valid_mirnas].copy()
print(f"Number of miRNAs retained after high-confidence filtering: {annotated_miRNA.n_vars}")
metadata = pd.read_csv(metadata_path, sep='\t').set_index("Sample").drop(columns=["Unnamed: 0"])

#Find the common columns betweeen the metadata and annotated_miRNA.obs
common_ind = annotated_miRNA.obs.index.intersection(metadata.index)

#Columns that need to be added from metadata, even though the annotated data object already includes "Project"
new_columns = ["Age", "Disease_Condition", "Project", "Sequencing_Method"]

annotated_miRNA = annotated_miRNA[common_ind]
annotated_miRNA.obs = annotated_miRNA.obs.drop(columns=["Project"])

annotated_miRNA.obs = annotated_miRNA.obs.join(metadata[new_columns])
annotated_miRNA = annotated_miRNA[~annotated_miRNA.obs.isna().any(axis=1)].copy()

annotated_miRNA = alter_disease_condition(annotated_miRNA)

annotated_miRNA = annotated_miRNA[annotated_miRNA.obs["Sex"] != "u"]
annotated_miRNA.write_h5ad(output_cleaned)

