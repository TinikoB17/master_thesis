import pandas as pd
import scanpy as sc
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler


eligible_mirnas = snakemake.input.threshold_miRNAs
h5ad_to_process = snakemake.input.blood_h5ad_healthy
test_h5ad = snakemake.input.test_h5ad
model_inp = snakemake.output.model_input
model_test = snakemake.output.model_test

model_inp_all_mirnas = snakemake.output.model_inp_all
model_test_all_mirnas = snakemake.output.model_test_all
cols_to_add = ["Age", "Project", "Sex"]


def preprocess_data(annotated_object: sc.AnnData, columns_to_add: list, train_stats=None) -> pd.DataFrame:
    mirna_expression = annotated_object.to_df()

    eligible_miRNAs = pd.read_csv(eligible_mirnas)["miRNA"].to_list()
    
    obs_dataframe = sc.get.obs_df(annotated_object, keys=columns_to_add)
    expression_with_added_cols = mirna_expression.join(obs_dataframe, how="left")
    # cols_to_apply_zscore = [col for col in expression_with_added_cols if col not in columns_to_add and col != "Project"]

    # scaler = StandardScaler()

    # X = expression_with_added_cols[cols_to_apply_zscore]
    # y = expression_with_added_cols["Age"]

    # X_filtered = X[eligible_miRNAs]
    
    # exp_all_scaled = scaler.fit_transform(X, y)
    # exp_scaled = scaler.fit_transform(X_filtered, y)
    print("EXP ADDED COLS")
    eligible_miRNAs.extend(["Age", "Sex"])
    print(expression_with_added_cols[eligible_miRNAs])
    return expression_with_added_cols, expression_with_added_cols[eligible_miRNAs]

    # return expression_with_added_cols




ann_obj = sc.read_h5ad(h5ad_to_process)
ann_test = sc.read_h5ad(test_h5ad)

exp_all_train, exp_train = preprocess_data(ann_obj, cols_to_add)
print(exp_all_train)
exp_all_test, exp_test = preprocess_data(ann_test, cols_to_add)


exp_train.to_csv(model_inp, sep='\t', index_label="Sample")
exp_test.to_csv(model_test, sep='\t', index_label="Sample")

print(exp_train)
print(exp_test)
exp_all_train.to_csv(model_inp_all_mirnas, sep='\t', index_label="Sample")
exp_all_test.to_csv(model_test_all_mirnas, sep='\t', index_label="Sample")
