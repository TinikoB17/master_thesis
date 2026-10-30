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
        sex_barplots = "figures/sex_distribution/healthy_blood_sex_dist.svg",
        # projects = gather_projects("test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad")

    script:
        "scripts/create_sex_distributions.py"

rule workflow_diagram:
    input:
        raw = "original_input/all_human_miRNA_rpmm_harmonized_meta_data.h5ad",
        cleaned = "test_output/all_human_miRNA_rpmm_harmonized_cleaned.h5ad",
        blood = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad",
        blood_healthy = "test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad",
        age_filtered = "test_output/age_filtered_blood_healthy.h5ad",
        train = "test_output/blood_healthy_train.h5ad",
        test = "test_output/blood_healthy_test.h5ad",
        prevalence_filtered = "test_output/blood_healthy_rpmm_filt_train.h5ad",
        high_confidence_mirnas = "test_output/high_confidence_mirnas.txt",
        changing_mirnas = "test_output/changing_mirnas.csv",
        cv_summaries = expand("models/repeated_cv/{model}_summary.txt", model=["linear_regression", "linear_regression_all", "linear_regression_pol2", "lasso", "lasso_all", "hgb"]),
        holdout_evals = ["models/evaluation/linear_regression_eval_train_stats.txt", "models/evaluation/linear_regression_eval_all_train_stats.txt", "models/evaluation/linear_regression_pol2_eval_train_stats.txt", "models/evaluation/lasso_eval_train_stats.txt", "models/evaluation/lasso_all_eval_train_stats.txt", "models/evaluation/hgb_eval_train_stats.txt"]
    params:
        cv_dir = "models/repeated_cv",
        eval_dir = "models/evaluation"
    output:
        workflow = "figures/workflow.svg"
    script:
        "scripts/workflow_diagram.py"
