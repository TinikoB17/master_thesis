from helpers import gather_projects

OUT = "test_output"
FIG = "figures"
MOD = "models"

MODELS = ["linear_regression", "linear_regression_pol2", "lasso", "hgb"]
UMAP_TARGETS = ["Tissue", "Project", "Sequening_Method", "Age"]

# 1. Define an input function that queries the checkpoint output
def get_dynamic_projects(wildcards):
    # .get() forces Snakemake to wait for this checkpoint to finish
    checkpoint_output = checkpoints.produce_blood_data.get().output.blood_h5ad_healthy
    return gather_projects(checkpoint_output)

rule all:
    input:
        expand(f"{FIG}/umap_all_{{target}}.svg", target=UMAP_TARGETS),
        expand(f"{FIG}/umap_blood_{{target}}.svg", target=UMAP_TARGETS),
        expand(f"{MOD}/evaluation/{{model}}_eval.txt", model=MODELS),
        # 2. Pass the dynamic function to your target inputs
        get_dynamic_projects


rule clean_original_h5ad:
    input:
        original_h5ad = "original_input/all_human_miRNA_rpmm_harmonized_meta_data.h5ad",
        metadata = "original_input/human_combined.tsv"
    output: 
        cleaned_data = f"{OUT}/all_human_miRNA_rpmm_harmonized_cleaned.h5ad"
    script:
        "scripts/clean_original_h5ad.py"

# 3. Convert the rule into a checkpoint
checkpoint produce_blood_data:
    input:
        cleaned_data = f"{OUT}/all_human_miRNA_rpmm_harmonized_cleaned.h5ad"
    output:
        blood_h5ad_all_diseases = f"{OUT}/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad",
        blood_h5ad_healthy = f"{OUT}/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad",
        blood_metadata = f"{OUT}/blood_metadata_tissueatlas.csv",
        blood_healthy_metadata = f"{OUT}/blood_metadata_tissueatlas_healthy.csv"
    script:
        "scripts/filter_cleaned_h5ads.py"

rule filter_age:
    input:
        rpmm_filtered_annotated = f"{OUT}/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad"
    output:
        age_filtered = f"{OUT}/age_filtered_blood_healthy.h5ad"
    script:
        "scripts/filter_age.py"

rule visualize_rpmm_thresholds:
    input:
        blood_h5ad_healthy = f"{OUT}/age_filtered_blood_healthy.h5ad"
    output:
        prevalence = f"{FIG}/gene_expression_prevalence.svg",
        threshold_miRNAs = f"{OUT}/pass_threshold_mirnas.csv",
        rpmm_filtered_annotated = f"{OUT}/rpmm_age_filtered_blood_healthy.h5ad"
    script:
        "scripts/rpmm_thresholds.py"

rule split_train_test:
    input:
        rpmm_filtered_annotated = f"{OUT}/rpmm_age_filtered_blood_healthy.h5ad"
    output:
        blood_healthy_train = f"{OUT}/blood_healthy_train.h5ad",
        blood_healthy_test = f"{OUT}/blood_healthy_test.h5ad",
        expression_train = f"{OUT}/expression_train.csv",
        metadata_train = f"{OUT}/metadata_train.csv",
        expression_test = f"{OUT}/expression_test.csv",
        metadata_test = f"{OUT}/metadata_test.csv"
    script:
        "scripts/split_train_test.py"

rule calculate_corr:
    input:
        blood_h5ad_healthy = f"{OUT}/blood_healthy_train.h5ad",
        metadata_train = f"{OUT}/metadata_train.csv"
    output:
        changing_mirnas = f"{OUT}/changing_mirnas.csv",
        linearly_changing_mirnas = f"{OUT}/linearly_changing_mirnas.csv"
    script:
        "scripts/calculate_correlations.py"

rule produce_model_input:
    input:
        blood_h5ad_healthy = f"{OUT}/blood_healthy_train.h5ad",
        threshold_miRNAs = f"{OUT}/changing_mirnas.csv",
        test_h5ad = f"{OUT}/blood_healthy_test.h5ad"
    output:
        model_input = "model_input_data/blood_healthy_age_scaled_train.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test.csv"
    script:
        "scripts/model_preprocessing.py"

include: "visualization.smk"
include: "run_models.smk"
