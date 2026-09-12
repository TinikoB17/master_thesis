from helpers import gather_projects


rule all:
    input:
        "figures/umap_all_Tissue.svg",
        "figures/umap_all_Project.svg",
        "figures/umap_all_Sequencing_Method.svg",
        "figures/umap_all_Age.svg",
        "figures/umap_blood_Tissue.svg",
        "figures/umap_blood_Project.svg",
        "figures/umap_blood_Sequencing_Method.svg",
        "figures/umap_blood_Age.svg",
        "figures/umap_highlight_blood.svg",
        "figures/umap_sdai.svg",
        # top_diseases = "test_output/top_diseases.csv",
        all_diseases_blood = "figures/All Diseases_Blood_age_histogram.svg",
        all_tissues_all_diseases = "figures/All_All_age_histogram.svg",
        all_conditions_blood = "figures/All_Blood_age_histogram.svg",
        healthy_blood = "figures/Healthy_Blood_age_histogram.svg",
        cleaned_data = "test_output/all_human_miRNA_rpmm_harmonized_cleaned.h5ad",
        blood_h5ad_all_diseases = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad",
        blood_h5ad_healthy = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad",
        rpmm_filtered_annotated = "test_output/rpmm_filtered_blood_healthy.h5ad",
        blood_healthy_train = "test_output/blood_healthy_train.h5ad",
        blood_healthy_test = "test_output/blood_healthy_test.h5ad",
        rpmm_filtered_train = "test_output/blood_healthy_rpmm_filt_train.h5ad",
        rpmm_filtered_test = "test_output/blood_healthy_rpmm_filt_test.h5ad",
        blood_metadata = "test_output/blood_metadata_tissueatlas.csv",
        blood_healthy_metadata = "test_output/blood_metadata_tissueatlas_healthy.csv",
        prevalence = "figures/gene_expression_prevalence.svg",
        threshold_miRNAs = "test_output/pass_threshold_mirnas.csv",
        model_input = "model_input_data/blood_healthy_age_scaled_train.csv",
        # test_inp = "models/test_disease.csv",
        changing_mirnas = "test_output/changing_mirnas.csv",
        linearly_changing_mirnas = "test_output/linearly_changing_mirnas.csv",
        age_filtered = "test_output/rpmm_age_filtered_blood_healthy.h5ad",
        model_test = "model_input_data/blood_healthy_age_scaled_test.csv",
        train_age = "figures/train_age_histogram.svg",
        test_age = "figures/test_age_histogram.svg",
        # sex_barplots = "figures/sex_distribution/healthy_blood_sex_dist.svg",
        projects = gather_projects("test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad"),
        linear_regression_eval = "models/evaluation/linear_regression_eval.txt",
        linear_regression_pol2_eval = "models/evaluation/linear_regression_pol2_eval.txt",
        lasso_eval = "models/evaluation/lasso_eval.txt",
        hgb_eval = "models/evaluation/hgb_eval.txt",
        model_inp_all = "model_input_data/blood_healthy_age_all_train.csv",
        model_test_all = "model_input_data/blood_healthy_age_all_test.csv"



rule clean_original_h5ad:
    input:
        original_h5ad = "original_input/all_human_miRNA_rpmm_harmonized_meta_data.h5ad",
        metadata = "original_input/human_combined.tsv"
    output: 
        cleaned_data = "test_output/all_human_miRNA_rpmm_harmonized_cleaned.h5ad"
    script:
        "scripts/clean_original_h5ad.py"

rule produce_blood_data:
    input:
        cleaned_data = "test_output/all_human_miRNA_rpmm_harmonized_cleaned.h5ad"
    output:
        blood_h5ad_all_diseases = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad",
        blood_h5ad_healthy = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad",
        blood_metadata = "test_output/blood_metadata_tissueatlas.csv",
        blood_healthy_metadata = "test_output/blood_metadata_tissueatlas_healthy.csv"
    script:
        "scripts/filter_cleaned_h5ads.py"

rule filter_age:
    input:
        # rpmm_filtered_annotated = "test_output/batch_corrected_blood_healthy.h5ad"
        rpmm_filtered_annotated = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad"
    output:
        age_filtered = "test_output/age_filtered_blood_healthy.h5ad"
    script:
        "scripts/filter_age.py"

rule split_train_test:
    input:
        rpmm_filtered_annotated = "test_output/age_filtered_blood_healthy.h5ad"
    output:
        blood_healthy_train = "test_output/blood_healthy_train.h5ad",
        blood_healthy_test = "test_output/blood_healthy_test.h5ad",
        expression_train = "test_output/expression_train.csv",
        metadata_train = "test_output/metadata_train.csv",
        expression_test = "test_output/expression_test.csv",
        metadata_test = "test_output/metadata_test.csv"
    script:
        "scripts/split_train_test.py"


rule visualize_rpmm_thresholds:
    input:
        blood_h5ad_healthy_train = "test_output/blood_healthy_train.h5ad",
        blood_healthy_test = "test_output/blood_healthy_test.h5ad"
    output:
        prevalence = "figures/gene_expression_prevalence.svg",
        threshold_miRNAs = "test_output/pass_threshold_mirnas.csv",
        rpmm_filtered_train = "test_output/blood_healthy_rpmm_filt_train.h5ad",
        rpmm_filtered_test = "test_output/blood_healthy_rpmm_filt_test.h5ad"
    script:
        "scripts/rpmm_thresholds.py"

rule calculate_corr:
    input:
        rpmm_filtered_train = "test_output/blood_healthy_rpmm_filt_train.h5ad",
        metadata_train = "test_output/metadata_train.csv"
    output:
        changing_mirnas = "test_output/changing_mirnas.csv",
        linearly_changing_mirnas = "test_output/linearly_changing_mirnas.csv"
    script:
        "scripts/calculate_correlations.py"


rule produce_model_input:
    input:
        blood_h5ad_healthy = "test_output/blood_healthy_rpmm_filt_train.h5ad",
        threshold_miRNAs = "test_output/changing_mirnas.csv",
        # linearly_changing_mirnas = "",
        test_h5ad = "test_output/blood_healthy_rpmm_filt_test.h5ad"
    output:
        model_input = "model_input_data/blood_healthy_age_scaled_train.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test.csv",
        model_inp_all = "model_input_data/blood_healthy_age_all_train.csv",
        model_test_all = "model_input_data/blood_healthy_age_all_test.csv"

    script:
        "scripts/model_preprocessing.py"

# rule find_top_diseases:
#     input:
#         blood_metadata = "test_output/blood_metadata_tissueatlas.csv"
#     output:
#         top_diseases = "test_output/top_diseases.csv"
#     script:
#         "scripts/top_diseases.py"
include: "visualization.smk"
include: "run_models.smk"
