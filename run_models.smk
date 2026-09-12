rule run_linear_regression:
    input:
        model_input = "model_input_data/blood_healthy_age_scaled_train.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test.csv"
    output:
        linear_regression_eval = "models/evaluation/linear_regression_eval.txt"
    script:
        "scripts/multiple_linear_regression_tuned.py"

rule run_linear_regression_pol2:
    input:
        model_input = "model_input_data/blood_healthy_age_scaled_train.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test.csv"
    output:
        linear_regression_pol2_eval = "models/evaluation/linear_regression_pol2_eval.txt"
    script:
        "scripts/multiple_linear_regression_poly2_tuned.py"

rule run_lasso:
    input:
        model_input = "model_input_data/blood_healthy_age_scaled_train.csv",
        model_test = "model_input_data/blood_healthy_age_scaled_test.csv"
    output:
        lasso_eval = "models/evaluation/lasso_eval.txt"
    script:
        "scripts/lasso_tuned.py"

rule run_hgb:
    input:
        model_input = "model_input_data/blood_healthy_age_all_train.csv",
        model_test = "model_input_data/blood_healthy_age_all_test.csv"
    output:
        hgb_eval = "models/evaluation/hgb_eval.txt"
    script:
        "scripts/boosting_tuned.py"
