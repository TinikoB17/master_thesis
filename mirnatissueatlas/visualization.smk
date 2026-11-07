import  scanpy as sc
from helpers import gather_projects

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

# Histogram of the Pearson/Spearman correlations of all miRNAs with age (training set)
rule plot_correlation_distribution:
    input:
        correlations = "test_output/correlations.csv"
    output:
        pearson = "figures/Pearson_correlation_histogram.svg",
        spearman = "figures/Spearman_correlation_histogram.svg"
    params:
        min_abs_corr = config["correlation"]["min_abs_corr"],
        max_pval = config["correlation"]["max_pval"]
    script:
        "scripts/plot_correlation_distribution.py"

# Healthy blood samples: sample count and oldest sample per project
rule explore_projects:
    input:
        blood_healthy_metadata = "test_output/blood_metadata_tissueatlas_healthy.csv"
    output:
        project_counts = "figures/healthy_blood_project_counts.svg",
        project_max_age = "figures/healthy_blood_project_max_age.svg"
    script:
        "scripts/explore_projects.py"

# All blood samples: sample count per sequencing method and per disease condition
rule prevalent_seq_methods:
    input:
        blood_metadata = "test_output/blood_metadata_tissueatlas.csv"
    output:
        sequencing_method = "figures/count_sequencing_method_in_blood.svg",
        disease_condition = "figures/count_disease_condition_in_blood.svg"
    script:
        "scripts/prevalent_seq_methods.py"

# Expression vs. age, one SVG per miRNA; h5ad, miRNA list and log2 option from config.yaml
rule plot_mirna_expression_vs_age:
    input:
        h5ad = config["expression_plots"]["h5ad"]
    output:
        plot_dir = directory("figures/mirna_expression_vs_age")
    params:
        projects = config["expression_plots"]["projects"],
        mirnas = config["expression_plots"]["mirnas"],
        top_n = config["expression_plots"]["top_n"],
        log2 = config["expression_plots"]["log2"]
    script:
        "scripts/plot_mirna_expression_vs_age.py"
