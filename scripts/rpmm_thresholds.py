import scanpy as sc
import visualize_input_data

blood_healthy_train = snakemake.input.blood_h5ad_healthy_train
blood_healthy_test = snakemake.input.blood_healthy_test
saveas = snakemake.output.prevalence
list_saveas = snakemake.output.threshold_miRNAs
# save_rpmm_filtered = snakemake.output.rpmm_filtered_annotated

rpmm_filtered_train = snakemake.output.rpmm_filtered_train
rpmm_filtered_test = snakemake.output.rpmm_filtered_test

annotated_data_blood_healthy_train = sc.read_h5ad(blood_healthy_train) 
annotated_data_blood_healthy_test = sc.read_h5ad(blood_healthy_test)
print("new thresholds")
rpmm_filtered_mirnas = visualize_input_data.plot_count_of_miRNAs_at_threshold(annotated_data_blood_healthy_train, saveas, list_saveas)
print("nmirnas")
filtered_anndata_train = annotated_data_blood_healthy_train[:, rpmm_filtered_mirnas].copy()
filtered_anndata_test = annotated_data_blood_healthy_test[:, rpmm_filtered_mirnas].copy()
sc.pp.log1p(filtered_anndata_train, base=2)
sc.pp.log1p(filtered_anndata_test, base=2)
filtered_anndata_train.write_h5ad(rpmm_filtered_train)
filtered_anndata_test.write_h5ad(rpmm_filtered_test)
