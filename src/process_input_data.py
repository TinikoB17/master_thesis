import pandas as pd
import scanpy as sc
import visualize_input_data
from preprocess_disease_condition import alter_disease_condition

annotated_miRNA_path = "original_input/all_human_miRNA_rpmm_harmonized_meta_data.h5ad"
metadata_path = "original_input/human_combined.tsv"

annotated_miRNA = sc.read_h5ad(annotated_miRNA_path)
metadata = pd.read_csv(metadata_path, sep='\t', index_col=False).set_index("Sample")

#Find the common columns betweeen the metadata and annotated_miRNA.obs

common_ind = annotated_miRNA.obs.index.intersection(metadata.index)

#Columns that need to be added from metadata, even though the annotated data object already includes "Project"
new_columns = ["Age", "Disease_Condition", "Project", "Sequencing_Method"]

annotated_miRNA = annotated_miRNA[common_ind]
annotated_miRNA.obs = annotated_miRNA.obs.drop(columns=["Project"])

annotated_miRNA.obs = annotated_miRNA.obs.join(metadata[new_columns])
annotated_miRNA = annotated_miRNA[~annotated_miRNA.obs.isna().any(axis=1)].copy()

annotated_miRNA = alter_disease_condition(annotated_miRNA)

annotated_miRNA.write_h5ad("filtered_input/all_human_miRNA_rpmm_harmonized_cleaned.h5ad")


#Produce the plots again
visualize_input_data.plot_umap_from_anndata(annotated_miRNA, color_by="Disease_Condition", blood_or_all_tissues="All Tissues")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA, color_by="Tissue", blood_or_all_tissues="All Tissues")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA, color_by="Project", blood_or_all_tissues="All Tissues")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA, color_by="Sequencing_Method", blood_or_all_tissues="All Tissues")
visualize_input_data.plot_umap_from_anndata(annotated_miRNA, color_by="Age", blood_or_all_tissues="All Tissues")
visualize_input_data.plot_highlight_blood(annotated_miRNA)

