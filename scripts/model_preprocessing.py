import pandas as pd
import scanpy as sc
from scipy.stats import zscore


eligible_mirnas = snakemake.input.threshold_miRNAs
h5ad_to_process = snakemake.input.blood_h5ad_healthy
model_inp = snakemake.output.model_input
cols_to_add = ["Age", "Sequencing_Method", "Project"]


def preprocess_data(annotated_object: sc.AnnData, columns_to_add: list) -> pd.DataFrame:
    mirna_expression = annotated_object.to_df()
    column_sums = mirna_expression.sum(axis=0, numeric_only=True)
    drop_cols = column_sums[column_sums == 0].index

    mirna_expression = mirna_expression.drop(columns=drop_cols)

    eligible_miRNAs = pd.read_csv(eligible_mirnas)["0"].to_list()
    mirna_expression = mirna_expression[eligible_miRNAs]

    obs_dataframe = sc.get.obs_df(annotated_object, keys=columns_to_add)
    expression_with_added_cols = mirna_expression.join(obs_dataframe, how="left")
    cols_to_apply_zscore = [col for col in expression_with_added_cols if col not in columns_to_add]
    grouped = expression_with_added_cols.groupby("Project")
    expression_with_added_cols[cols_to_apply_zscore] = (
    grouped[cols_to_apply_zscore]
    .transform(lambda x: zscore(x, ddof=1))
    .fillna(0))
    
    expression_with_added_cols.to_csv("model_input_data/blood_healthy_age_scaled.csv", sep='\t')

    # return expression_with_added_cols




ann_obj = sc.read_h5ad(h5ad_to_process)
preprocess_data(ann_obj, cols_to_add)
