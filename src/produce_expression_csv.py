import pandas as pd
import scanpy as sc



annotated_obj_blood = sc.read_h5ad("filtered_input/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad")
mirna_expression = annotated_obj_blood.to_df()
mirna_expression = mirna_expression.transpose()
breakpoint()
mirna_expression.to_csv("filtered_input/tissueatlas_blood_expression.csv", sep='\t')
breakpoint()