import scanpy as sc
import visualize_input_data

blood_healthy = snakemake.input.blood_h5ad_healthy
saveas = snakemake.output.prevalence
list_saveas = snakemake.output.threshold_miRNAs
save_rpmm_filtered = snakemake.output.rpmm_filtered_annotated

annotated_data_blood_healthy = sc.read_h5ad(blood_healthy) 

print("new thresholds")
rpmm_filtered_mirnas = visualize_input_data.plot_count_of_miRNAs_at_threshold(annotated_data_blood_healthy, saveas, list_saveas)

filtered_anndata = annotated_data_blood_healthy[:, rpmm_filtered_mirnas]
filtered_anndata.write_h5ad(save_rpmm_filtered)
