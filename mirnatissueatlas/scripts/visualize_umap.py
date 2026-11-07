import scanpy as sc
from visualize_input_data import plot_umap_from_anndata, plot_highlight_blood

cleaned_h5ad = snakemake.input.cleaned_data
blood_h5ad = snakemake.input.blood_h5ad_all_diseases
blood_h5ad_healthy = snakemake.input.blood_h5ad_healthy

annotated_miRNA = sc.read_h5ad(cleaned_h5ad)
annotated_miRNA_blood = sc.read_h5ad(blood_h5ad)
annotated_miRNA_blood_healthy = sc.read_h5ad(blood_h5ad_healthy)

sdai = annotated_miRNA[annotated_miRNA.obs["Disease_Condition"].str.contains("SDAI")].copy()

plot_umap_from_anndata(annotated_miRNA, color_by="Disease_Condition", blood_or_all_tissues="All Tissues")
plot_umap_from_anndata(annotated_miRNA, color_by="Tissue", blood_or_all_tissues="All Tissues")
plot_umap_from_anndata(annotated_miRNA, color_by="Project", blood_or_all_tissues="All Tissues")
plot_umap_from_anndata(annotated_miRNA, color_by="Sequencing_Method", blood_or_all_tissues="All Tissues")
plot_umap_from_anndata(annotated_miRNA, color_by="Age", blood_or_all_tissues="All Tissues")
plot_highlight_blood(annotated_miRNA)

plot_umap_from_anndata(annotated_miRNA_blood, color_by="Disease_Condition", blood_or_all_tissues="Blood")
plot_umap_from_anndata(annotated_miRNA_blood, color_by="Tissue", blood_or_all_tissues="Blood")
plot_umap_from_anndata(annotated_miRNA_blood, color_by="Project", blood_or_all_tissues="Blood")
plot_umap_from_anndata(annotated_miRNA_blood, color_by="Sequencing_Method", blood_or_all_tissues="Blood")
plot_umap_from_anndata(annotated_miRNA_blood, color_by="Age", blood_or_all_tissues="Blood")
plot_umap_from_anndata(sdai, color_by="Disease_Condition", blood_or_all_tissues="All Tissues", saveas="_sdai.svg")
