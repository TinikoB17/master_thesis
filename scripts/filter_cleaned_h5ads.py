import scanpy as sc
import pandas as pd
import numpy as np

#snakemake params
annotated_miRNA_path = snakemake.input.cleaned_data
blood_h5ad_all_diseases = snakemake.output.blood_h5ad_all_diseases
blood_h5ad_healthy = snakemake.output.blood_h5ad_healthy
blood_metadata = snakemake.output.blood_metadata
blood_metadata_healthy = snakemake.output.blood_healthy_metadata

annotated_miRNA = sc.read_h5ad(annotated_miRNA_path)
annotated_miRNA = annotated_miRNA[annotated_miRNA.obs["Biotype"] == "tissue"]
annotated_miRNA_blood = annotated_miRNA[annotated_miRNA.obs["Tissue"] == "blood"]
annotated_miRNA_blood.write_h5ad(blood_h5ad_all_diseases)

mirna_sums = np.array(annotated_miRNA_blood.X.sum(axis=0)).ravel()
annotated_miRNA_blood = annotated_miRNA_blood[:, mirna_sums > 0]

annotated_miRNA_blood_healthy = annotated_miRNA_blood[annotated_miRNA_blood.obs["Disease_Condition"] == "Healthy"]
annotated_miRNA_blood_healthy.write_h5ad(blood_h5ad_healthy)

annotated_miRNA_blood.obs.to_csv(blood_metadata, sep='\t')
annotated_miRNA_blood_healthy.obs.to_csv(blood_metadata_healthy, sep='\t')
