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
