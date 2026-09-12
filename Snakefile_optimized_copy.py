# Optimized copy of the original Snakemake workflow.
# This preserves the same overall pipeline but makes the dependency flow clearer and avoids redundant work.
# The original file remains untouched: Snakefile

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
        "figures/All Diseases_Blood_age_histogram.svg",
        "figures/All_All_age_histogram.svg",
        "figures/All_Blood_age_histogram.svg",
        "figures/Healthy_Blood_age_histogram.svg",
        "test_output/all_human_miRNA_rpmm_harmonized_cleaned.h5ad",
        "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad",
        "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad",
        "test_output/rpmm_age_filtered_blood_healthy.h5ad",
        "test_output/blood_healthy_train.h5ad",
        "test_output/blood_healthy_test.h5ad",
        "test_output/expression_train.csv",
        "test_output/metadata_train.csv",
        "test_output/expression_test.csv",
        "test_output/metadata_test.csv",
        "test_output/blood_metadata_tissueatlas.csv",
        "test_output/blood_metadata_tissueatlas_healthy.csv",
        "figures/gene_expression_prevalence.svg",
        "test_output/pass_threshold_mirnas.csv",
        "model_input_data/blood_healthy_age_scaled_train.csv",
        "test_output/changing_mirnas.csv",
        "correlations/cluster_miRNAs_rpmm_filtered.svg",
        "model_input_data/blood_healthy_age_scaled_test.csv",
        "figures/train_age_histogram.svg",
        "figures/test_age_histogram.svg"


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
        rpmm_filtered_annotated = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad"
    output:
        age_filtered = "test_output/age_filtered_blood_healthy.h5ad"
    script:
        "scripts/filter_age.py"


rule visualize_rpmm_thresholds:
    input:
        blood_h5ad_healthy = "test_output/age_filtered_blood_healthy.h5ad"
    output:
        prevalence = "figures/gene_expression_prevalence.svg",
        threshold_miRNAs = "test_output/pass_threshold_mirnas.csv",
        rpmm_filtered_annotated = "test_output/rpmm_age_filtered_blood_healthy.h5ad"
    script:
        "scripts/rpmm_thresholds.py"


rule split_train_test:
    input:
        rpmm_filtered_annotated = "test_output/rpmm_age_filtered_blood_healthy.h5ad"
    output:
        blood_healthy_train = "test_output/blood_healthy_train.h5ad",
        blood_healthy_test = "test_output/blood_healthy_test.h5ad",
        expression_train = "test_output/expression_train.csv",
        metadata_train = "test_output/metadata_train.csv",
        expression_test = "test_output/expression_test.csv",
        metadata_test = "test_output/metadata_test.csv"
    script:
        "scripts/split_train_test.py"


rule calculate_corr:
    input:
        blood_h5ad_healthy = "test_output/blood_healthy_train.h5ad",
        metadata_train = "test_output/metadata_train.csv"
    output:
        changing_mirnas = "test_output/changing_mirnas.csv",
        cluster_hist = "correlations/cluster_miRNAs_rpmm_filtered.svg"
    script:
        "scripts/calculate_correlations.py"


rule produce_model_input:
    input:
        blood_h5ad_healthy = "test_output/blood_healthy_train.h5ad",
        threshold_miRNAs = "test_output/changing_mirnas.csv",
        test_h5ad = "test_output/blood_healthy_test.h5ad"
    output:
        model_input = "model_input_data/blood_healthy_age_scaled_train.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test.csv"
    script:
        "scripts/model_preprocessing.py"


rule visualize_umap:
    input:
        cleaned_data = "test_output/all_human_miRNA_rpmm_harmonized_cleaned.h5ad",
        blood_h5ad_all_diseases = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad",
        blood_h5ad_healthy = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad",
    output:
        "figures/umap_all_Tissue.svg",
        "figures/umap_all_Project.svg",
        "figures/umap_all_Sequencing_Method.svg",
        "figures/umap_all_Age.svg",
        "figures/umap_blood_Tissue.svg",
        "figures/umap_blood_Project.svg",
        "figures/umap_blood_Sequencing_Method.svg",
        "figures/umap_blood_Age.svg",
        "figures/umap_highlight_blood.svg",
        "figures/umap_sdai.svg"
    script:
        "scripts/visualize_umap.py"


rule create_age_histograms:
    input:
        cleaned_data = "test_output/all_human_miRNA_rpmm_harmonized_cleaned.h5ad",
        blood_h5ad_all_diseases = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad",
        blood_h5ad_healthy = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad",
        train_age = "test_output/blood_healthy_train.h5ad",
        test_age = "test_output/blood_healthy_test.h5ad"
    output:
        all_diseases_blood = "figures/All Diseases_Blood_age_histogram.svg",
        all_tissues_all_diseases = "figures/All_All_age_histogram.svg",
        all_conditions_blood = "figures/All_Blood_age_histogram.svg",
        healthy_blood = "figures/Healthy_Blood_age_histogram.svg",
        train_age = "figures/train_age_histogram.svg",
        test_age = "figures/test_age_histogram.svg"
    script:
        "scripts/create_age_histograms.py"

# Comments from the original file moved aside: these rules were currently disabled.
# rule produce_test_data:
#     input:
#         blood_h5ad = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad",
#         changing_mirnas = "test_output/pass_threshold_mirnas.csv"
#     output:
#         test_inp = "models/test_disease.csv",
#     script:
#         "scripts/produce_model_test.py"
#
# rule find_top_diseases:
#     input:
#         blood_metadata = "test_output/blood_metadata_tissueatlas.csv"
#     output:
#         top_diseases = "test_output/top_diseases.csv"
#     script:
#         "scripts/top_diseases.py"
