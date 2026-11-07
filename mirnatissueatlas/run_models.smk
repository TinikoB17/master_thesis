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

rule run_svr_train_stats:
    input:
        model_input = "model_input_data/blood_healthy_age_all_train_train_stats.csv",
        model_test = "model_input_data/blood_healthy_age_all_test_train_stats.csv"
    output:
        svr_eval = "models/evaluation/svr_eval_train_stats.txt",
        svr_model = "models/svr_model_train_stats.pkl",
        cv_plot = "figures/models/svr_train_stats_cv_prediction.svg",
        test_plot = "figures/models/svr_train_stats_test_prediction.svg"
    script:
        "scripts/svr_tuned.py"
