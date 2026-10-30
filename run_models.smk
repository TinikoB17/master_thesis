rule run_linear_regression_lin:
    input:
        model_input = "model_input_data/blood_healthy_age_scaled_train.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test.csv"
    output:
        linear_regression_eval = "models/evaluation/linear_regression_eval.txt",
        linear_reg_model = "models/linear_regression.pkl"
    script:
        "scripts/multiple_linear_regression_tuned.py"

rule run_linear_regression_all:
    input:
        model_input = "model_input_data/blood_healthy_age_all_train.csv",
        model_test = "model_input_data/blood_healthy_age_all_test.csv"
    output:
        linear_regression_eval_all = "models/evaluation/linear_regression_eval_all.txt",
        linear_reg_model_all = "models/linear_regression_all.pkl"
    script:
        "scripts/multiple_linear_regression_tuned_all.py"

rule run_linear_regression_pol2:
    input:
        model_input = "model_input_data/blood_healthy_age_scaled_train.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test.csv"
    output:
        linear_regression_pol2_eval = "models/evaluation/linear_regression_pol2_eval.txt",
        pol2_model = "models/linear_reg_pol2.pkl"
    script:
        "scripts/multiple_linear_regression_poly2_tuned.py"

rule run_lasso:
    input:
        model_input = "model_input_data/blood_healthy_age_scaled_train.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test.csv"
    output:
        lasso_eval = "models/evaluation/lasso_eval.txt",
        lasso_model = "models/lasso_model.pkl"
    script:
        "scripts/lasso_tuned.py"

rule run_lasso_all:
    input:
        model_input = "model_input_data/blood_healthy_age_all_train.csv",
        model_test = "model_input_data/blood_healthy_age_all_test.csv"
    output:
        lasso_all_eval = "models/evaluation/lasso_all_eval.txt",
        lasso_all_model = "models/lasso_all.pkl"
    script:
        "scripts/lasso_tuned_all.py"

rule run_hgb:
    input:
        model_input = "model_input_data/blood_healthy_age_all_train.csv",
        model_test = "model_input_data/blood_healthy_age_all_test.csv"
    output:
        hgb_eval = "models/evaluation/hgb_eval.txt",
        hgb_model = "models/hgb_model.pkl"
    script:
        "scripts/boosting_tuned.py"


# ---- Same models, test data z-scored with training-set per-project stats ----

rule run_linear_regression_lin_train_stats:
    input:
        model_input = "model_input_data/blood_healthy_age_scaled_train_train_stats.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test_train_stats.csv"
    output:
        linear_regression_eval = "models/evaluation/linear_regression_eval_train_stats.txt",
        linear_reg_model = "models/linear_regression_train_stats.pkl"
    script:
        "scripts/multiple_linear_regression_tuned.py"

rule run_linear_regression_all_train_stats:
    input:
        model_input = "model_input_data/blood_healthy_age_all_train_train_stats.csv",
        model_test = "model_input_data/blood_healthy_age_all_test_train_stats.csv"
    output:
        linear_regression_eval_all = "models/evaluation/linear_regression_eval_all_train_stats.txt",
        linear_reg_model_all = "models/linear_regression_all_train_stats.pkl"
    script:
        "scripts/multiple_linear_regression_tuned_all.py"

rule run_linear_regression_pol2_train_stats:
    input:
        model_input = "model_input_data/blood_healthy_age_scaled_train_train_stats.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test_train_stats.csv"
    output:
        linear_regression_pol2_eval = "models/evaluation/linear_regression_pol2_eval_train_stats.txt",
        pol2_model = "models/linear_reg_pol2_train_stats.pkl"
    script:
        "scripts/multiple_linear_regression_poly2_tuned.py"

rule run_lasso_train_stats:
    input:
        model_input = "model_input_data/blood_healthy_age_scaled_train_train_stats.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test_train_stats.csv"
    output:
        lasso_eval = "models/evaluation/lasso_eval_train_stats.txt",
        lasso_model = "models/lasso_model_train_stats.pkl"
    script:
        "scripts/lasso_tuned.py"

rule run_lasso_all_train_stats:
    input:
        model_input = "model_input_data/blood_healthy_age_all_train_train_stats.csv",
        model_test = "model_input_data/blood_healthy_age_all_test_train_stats.csv"
    output:
        lasso_all_eval = "models/evaluation/lasso_all_eval_train_stats.txt",
        lasso_all_model = "models/lasso_all_train_stats.pkl"
    script:
        "scripts/lasso_tuned_all.py"

rule run_hgb_train_stats:
    input:
        model_input = "model_input_data/blood_healthy_age_all_train_train_stats.csv",
        model_test = "model_input_data/blood_healthy_age_all_test_train_stats.csv"
    output:
        hgb_eval = "models/evaluation/hgb_eval_train_stats.txt",
        hgb_model = "models/hgb_model_train_stats.pkl"
    script:
        "scripts/boosting_tuned.py"

# ---- Lasso on miRNAs with a significant age effect in the LMM (variance partitioning) ----
rule run_lasso_lmm_train_stats:
    input:
        model_input = "model_input_data/blood_healthy_age_lmm_train_train_stats.csv",
        model_test = "model_input_data/blood_healthy_age_lmm_test_train_stats.csv"
    output:
        lasso_all_eval = "models/evaluation/lasso_lmm_eval_train_stats.txt",
        lasso_all_model = "models/lasso_lmm_train_stats.pkl"
    script:
        "scripts/lasso_tuned_all.py"
