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

rule create_sex_barplots:
    input: 
        blood_h5ad_healthy = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad"
    output:
        # sex_barplots = "figures/sex_distribution/healthy_blood_sex_dist.svg",
        projects = gather_projects("test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad")

    script:
        "scripts/create_sex_distributions.py"
