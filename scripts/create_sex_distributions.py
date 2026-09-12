import scanpy as sc
import visualize_input_data


annotated_data_blood_healthy = sc.read_h5ad(snakemake.input.blood_h5ad_healthy)
# outp = snakemake.output.sex_barplots
# print(outp)
# visualize_input_data.plot_sex_distribution(annotated_data_blood_healthy, 
#                                            title = "Sex Distribution in Healthy Blood Samples", 
#                                            saveas = f"{outp}")
projects = annotated_data_blood_healthy.obs["Project"].unique().tolist()

for project in projects:
    ann = annotated_data_blood_healthy[annotated_data_blood_healthy.obs["Project"] == project]
    visualize_input_data.plot_sex_distribution(ann, 
                                           title = f"Sex Distribution in Healthy Blood Samples - Project {project}",
                                           saveas = f"figures/sex_distribution/{project}.svg")
