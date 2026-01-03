import pandas as pd

avg_expression = snakemake.input.avg_rpmm_expression
output = snakemake.output.tsi

avg_expressions_mirblood = pd.read_csv(avg_expression, sep='\t').set_index("miRNA")



for ind, row in avg_expressions_mirblood.iterrows():
    max_val_row = row.max()
    avg_expressions_mirblood.loc[ind, "Tissue_of_max_Expresison"] = row.idxmax()
    avg_expressions_mirblood.loc[ind, "TSI"] = sum(
        [((1 - val / max_val_row)) / (len(row) - 1) for val in row.values]
    )


print(avg_expressions_mirblood)

avg_expressions_mirblood.to_csv(output, sep='\t', index_label="miRNA")
