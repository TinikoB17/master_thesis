import pandas as pd
import scanpy as sc
import visualize_input_data
from preprocess_disease_condition import alter_disease_condition

annotated_miRNA_path = "original_input/all_human_miRNA_rpmm_harmonized_meta_data.h5ad"
metadata_path = "original_input/human_combined.tsv"

output_cleaned = "test_output/all_human_miRNA_rpmm_harmonized_cleaned.h5ad"
print(metadata_path)

annotated_miRNA = sc.read_h5ad(annotated_miRNA_path)
metadata = pd.read_csv(metadata_path, sep='\t').set_index("Sample").drop(columns=["Unnamed: 0"])
# deconvoluted_samples = pd.read_csv("filtered_input/deconvoluted_tissueatlas.csv", sep='\t').set_index("Sample")
print(metadata.head())
#Find the common columns betweeen the metadata and annotated_miRNA.obs

common_ind = annotated_miRNA.obs.index.intersection(metadata.index)

#Columns that need to be added from metadata, even though the annotated data object already includes "Project"
new_columns = ["Age", "Disease_Condition", "Project", "Sequencing_Method"]

annotated_miRNA = annotated_miRNA[common_ind]
annotated_miRNA.obs = annotated_miRNA.obs.drop(columns=["Project"])

annotated_miRNA.obs = annotated_miRNA.obs.join(metadata[new_columns])
annotated_miRNA = annotated_miRNA[~annotated_miRNA.obs.isna().any(axis=1)].copy()

annotated_miRNA = alter_disease_condition(annotated_miRNA)

annotated_miRNA.write_h5ad(output_cleaned)


#Produce the plots for the original input - all tissues
visualize_input_data.plot_umap_from_anndata(annotated_miRNA, color_by="Disease_Condition", blood_or_all_tissues="All Tissues")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA, color_by="Tissue", blood_or_all_tissues="All Tissues")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA, color_by="Project", blood_or_all_tissues="All Tissues")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA, color_by="Sequencing_Method", blood_or_all_tissues="All Tissues")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA, color_by="Age", blood_or_all_tissues="All Tissues")
visualize_input_data.plot_highlight_blood(annotated_miRNA)

#Filter out the samples that are not blood

annotated_miRNA_blood = annotated_miRNA[annotated_miRNA.obs["Tissue"] == "blood"]
# metadata_deconvolution_data = metadata.join(deconvoluted_samples, how="right")

# deconvolution_columns = deconvoluted_samples.columns.to_list()
# annotated_miRNA_blood.obs = annotated_miRNA_blood.obs.join(metadata_deconvolution_data[deconvolution_columns])
# common_ind = annotated_miRNA_blood.obs.index.intersection(metadata_deconvolution_data.index)

annotated_miRNA_blood.write_h5ad("filtered_input/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad")

annotated_miRNA_blood_healthy = annotated_miRNA_blood[annotated_miRNA_blood.obs["Disease_Condition"] == "Healthy"]
annotated_miRNA_blood_healthy.write_h5ad("filtered_input/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad")
print(annotated_miRNA_blood.obs.head())
annotated_miRNA_blood.obs.to_csv("filtered_input/blood_metadata_tissueatlas.csv", sep='\t')

visualize_input_data.plot_umap_from_anndata(annotated_miRNA_blood, color_by="Disease_Condition", blood_or_all_tissues="Blood")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA_blood, color_by="Tissue", blood_or_all_tissues="Blood")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA_blood, color_by="Project", blood_or_all_tissues="Blood")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA_blood, color_by="Sequencing_Method", blood_or_all_tissues="Blood")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA_blood, color_by="Age", blood_or_all_tissues="Blood")
