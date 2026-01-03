import pandas as pd
import numpy as np
import os

def make_pseudo(expr_df, meta_df, props):
    genes = expr_df.index
    mixture = pd.Series(0, index=genes, dtype=float)

    for ct, prop in props.items():
        samp_ids = meta_df.loc[meta_df.cell_type == ct, "Sample"]
        profile = expr_df[samp_ids].mean(axis=1)
        mixture += prop * profile

    return mixture
print(os.getcwd())
expr_df = pd.read_csv("deconvolution/std_quantification_rpmm_norm_mirna.csv", sep='\t', index_col=0)
meta_df = pd.read_csv("deconvolution/mirblood_metadata.csv", sep='\t')

df = pd.DataFrame()
# Example:
props = [{'Monocytes':0.5, 'Eosinophils':0.3, 'CD4+ T cells':0.2}, {'Monocytes':0.1, 'Eosinophils':0.4, 'CD4+ T cells':0.5}]
for i, prop in enumerate(props):
    pseudo_expr = make_pseudo(expr_df, meta_df, prop)
    df[f"column{i}"] = pseudo_expr
    breakpoint()
df.to_csv("pseudo_bulk.csv", sep='\t')