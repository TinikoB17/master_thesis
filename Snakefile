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
        all_diseases_blood = "figures/All Diseases_Blood_age_histogram.svg",
        all_tissues_all_diseases = "figures/All_All_age_histogram.svg",
        all_conditions_blood = "figures/All_Blood_age_histogram.svg",
        healthy_blood = "figures/Healthy_Blood_age_histogram.svg",
        cleaned_data = "test_output/all_human_miRNA_rpmm_harmonized_cleaned.h5ad",
        blood_h5ad_all_diseases = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad",
        blood_h5ad_healthy = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad",
        blood_metadata = "test_output/blood_metadata_tissueatlas.csv",
        blood_healthy_metadata = "test_output/blood_metadata_tissueatlas_healthy.csv",
        prevalence = "new_figs/gene_expression_prevalence.svg",
        model_inp = "model_input_data/blood_healthy_age_scaled.csv",
        threshold_miRNAs = "test_output/pass_threshold_mirnas.csv"


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
        "figures/umap_highlight_blood.svg"
    script:
        "scripts/visualize_umap.py"

rule visualize_rpmm_thresholds:
    input:
        blood_h5ad_healthy = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad"
    output:
        prevalence = "new_figs/gene_expression_prevalence.svg",
        threshold_miRNAs = "test_output/pass_threshold_mirnas.csv"
    script:
        "scripts/rpmm_thresholds.py"

rule create_age_histograms:
    input:
        cleaned_data = "test_output/all_human_miRNA_rpmm_harmonized_cleaned.h5ad",
        blood_h5ad_all_diseases = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad",
        blood_h5ad_healthy = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad",
    output:
        all_diseases_blood = "figures/All Diseases_Blood_age_histogram.svg",
        all_tissues_all_diseases = "figures/All_All_age_histogram.svg",
        all_conditions_blood = "figures/All_Blood_age_histogram.svg",
        healthy_blood = "figures/Healthy_Blood_age_histogram.svg"
    script:
        "scripts/create_age_histograms.py"

rule produce_model_input:
    input:
        blood_h5ad_healthy = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad",
        threshold_miRNAs = "test_output/pass_threshold_mirnas.csv"
    output:
        model_input = "model_input_data/blood_healthy_age_scaled.csv",

    script:
        "scripts/model_preprocessing.py"
