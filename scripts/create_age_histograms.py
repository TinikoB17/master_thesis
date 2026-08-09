import scanpy as sc
import visualize_input_data

annotated_data_all_tissues = sc.read_h5ad(snakemake.input.cleaned_data)
annotated_data_blood = sc.read_h5ad(snakemake.input.blood_h5ad_all_diseases)
annotated_data_blood_healthy = sc.read_h5ad(snakemake.input.blood_h5ad_healthy)
train_age = sc.read_h5ad(snakemake.input.train_age)
test_age = sc.read_h5ad(snakemake.input.test_age)

annotated_data_blood_disease = annotated_data_blood[annotated_data_blood.obs["Disease_Condition"] != "Healthy"]
visualize_input_data.plot_age_histograms(annotated_data_all_tissues, condition="All", tissue="All")
visualize_input_data.plot_age_histograms(annotated_data_blood, condition="All", tissue="Blood")
visualize_input_data.plot_age_histograms(annotated_data_blood_healthy, condition="Healthy", tissue="Blood")
visualize_input_data.plot_age_histograms(annotated_data_blood_disease, condition="All Diseases", tissue="Blood")
visualize_input_data.plot_age_histograms(train_age, condition="All Diseases", tissue="Blood", train_test="train")
visualize_input_data.plot_age_histograms(test_age, condition="All Diseases", tissue="Blood", train_test="test")
