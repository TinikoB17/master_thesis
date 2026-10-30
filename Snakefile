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
        variance_partitioning = "test_output/variance_partitioning.csv",
        variance_partitioning_50_70 = "test_output/variance_partitioning_age50-70.csv",
        lasso_lmm_eval = "models/evaluation/lasso_lmm_eval_train_stats.txt",
        workflow = "figures/workflow.svg",
        age_filtered = "test_output/rpmm_age_filtered_blood_healthy.h5ad",
        model_test = "model_input_data/blood_healthy_age_scaled_test.csv",
        train_age = "figures/train_age_histogram.svg",
        test_age = "figures/test_age_histogram.svg",
        sex_barplots = "figures/sex_distribution/healthy_blood_sex_dist.svg",
        projects = gather_projects("test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood_healthy.h5ad"),
        linear_regression_eval = "models/evaluation/linear_regression_eval.txt",
        linear_regression_pol2_eval = "models/evaluation/linear_regression_pol2_eval.txt",
        lasso_eval = "models/evaluation/lasso_eval.txt",
        hgb_eval = "models/evaluation/hgb_eval.txt",
        model_inp_all = "model_input_data/blood_healthy_age_all_train.csv",
        model_test_all = "model_input_data/blood_healthy_age_all_test.csv",
        linear_regression_eval_all = "models/evaluation/linear_regression_eval_all.txt",
        linear_reg_model_all = "models/linear_regression_all.pkl",
        lasso_all_eval = "models/evaluation/lasso_all_eval.txt",
        lasso_all_model = "models/lasso_all.pkl",
        # models evaluated on test data z-scored with training-set stats
        linear_regression_eval_ts = "models/evaluation/linear_regression_eval_train_stats.txt",
        linear_regression_eval_all_ts = "models/evaluation/linear_regression_eval_all_train_stats.txt",
        linear_regression_pol2_eval_ts = "models/evaluation/linear_regression_pol2_eval_train_stats.txt",
        lasso_eval_ts = "models/evaluation/lasso_eval_train_stats.txt",
        lasso_all_eval_ts = "models/evaluation/lasso_all_eval_train_stats.txt",
        hgb_eval_ts = "models/evaluation/hgb_eval_train_stats.txt",
        # repeated nested CV on all samples
        repeated_cv = expand("models/repeated_cv/{model}_summary.txt", model=["linear_regression", "linear_regression_all", "linear_regression_pol2", "lasso", "lasso_all", "hgb"])


rule clean_original_h5ad:
    input:
        original_h5ad = "original_input/all_human_miRNA_rpmm_harmonized_meta_data.h5ad",
        metadata = "original_input/human_combined.tsv",
        high_confidence_mirnas = "test_output/high_confidence_mirnas.txt"
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


rule variance_partitioning:
    input:
        rpmm_filtered_train = "test_output/blood_healthy_rpmm_filt_train.h5ad"
    output:
        variance = "test_output/variance_partitioning.csv",
        lmm_plot = "figures/lmm.svg"
    threads: 8
    script:
        "scripts/variance_partitioning.py"

# Same LMM restricted to an age window, e.g. test_output/variance_partitioning_age50-70.csv
rule variance_partitioning_age:
    input:
        rpmm_filtered_train = "test_output/blood_healthy_rpmm_filt_train.h5ad"
    output:
        variance = "test_output/variance_partitioning_age{min_age}-{max_age}.csv",
        lmm_plot = "figures/lmm_age{min_age}-{max_age}.svg"
    wildcard_constraints:
        min_age = r"\d+",
        max_age = r"\d+"
    threads: 8
    script:
        "scripts/variance_partitioning.py"


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

# Same as produce_model_input, but test set is z-scored with per-project stats from the training set
rule produce_model_input_train_stats:
    input:
        blood_h5ad_healthy = "test_output/blood_healthy_rpmm_filt_train.h5ad",
        threshold_miRNAs = "test_output/changing_mirnas.csv",
        test_h5ad = "test_output/blood_healthy_rpmm_filt_test.h5ad"
    output:
        model_input = "model_input_data/blood_healthy_age_scaled_train_train_stats.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test_train_stats.csv",
        model_inp_all = "model_input_data/blood_healthy_age_all_train_train_stats.csv",
        model_test_all = "model_input_data/blood_healthy_age_all_test_train_stats.csv"
    script:
        "scripts/model_preprocessing_train_stats.py"

# Feature set from the LMM: miRNAs with a significant age effect (BH-adjusted p < 0.05)
rule select_lmm_age_mirnas:
    input:
        variance = "test_output/variance_partitioning.csv"
    output:
        lmm_age_mirnas = "test_output/lmm_age_mirnas.csv"
    params:
        max_padj = 0.05
    run:
        import pandas as pd
        variance = pd.read_csv(input.variance, index_col="miRNA")
        significant = variance.index[variance["Age_padj"] < params.max_padj]
        pd.Series(significant, name="miRNA").to_csv(output.lmm_age_mirnas, index=False)

# Same as produce_model_input_train_stats, with the LMM age miRNAs instead of the correlation filter
rule produce_model_input_lmm:
    input:
        blood_h5ad_healthy = "test_output/blood_healthy_rpmm_filt_train.h5ad",
        threshold_miRNAs = "test_output/lmm_age_mirnas.csv",
        test_h5ad = "test_output/blood_healthy_rpmm_filt_test.h5ad"
    output:
        model_input = "model_input_data/blood_healthy_age_lmm_train_train_stats.csv",
        model_test = "model_input_data/blood_healthy_age_lmm_test_train_stats.csv"
    script:
        "scripts/model_preprocessing_train_stats.py"

# rule find_top_diseases:
#     input:
#         blood_metadata = "test_output/blood_metadata_tissueatlas.csv"
#     output:
#         top_diseases = "test_output/top_diseases.csv"
#     script:
#         "scripts/top_diseases.py"
include: "visualization.smk"
include: "run_models.smk"
include: "repeated_cv.smk"
