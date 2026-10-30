REPEATED_CV_MODELS = ["linear_regression", "linear_regression_all", "linear_regression_pol2", "lasso", "lasso_all", "hgb"]

# Repeated nested CV (5 folds x 10 repeats) on all healthy blood samples aged 30-80.
# All preprocessing (prevalence filter, log2, per-project z-score, correlation filter,
# power transform, feature selection) is refit inside each training fold.
rule repeated_cv:
    input:
        age_filtered = "test_output/age_filtered_blood_healthy.h5ad"
    output:
        folds = "models/repeated_cv/{model}_folds.tsv",
        oof_predictions = "models/repeated_cv/{model}_oof_predictions.tsv",
        summary = "models/repeated_cv/{model}_summary.txt",
        plot = "figures/repeated_cv/{model}_oof_predictions.svg"
    wildcard_constraints:
        model = "|".join(REPEATED_CV_MODELS)
    threads: 12
    script:
        "scripts/repeated_cv.py"

rule repeated_cv_all:
    input:
        expand("models/repeated_cv/{model}_summary.txt", model=REPEATED_CV_MODELS)
