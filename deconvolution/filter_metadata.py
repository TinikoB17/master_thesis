import pandas as pd 

metadata_path = snakemake.input.metadata
metadata = pd.read_csv(metadata_path, sep='\t')

expression_path = snakemake.input.filtered_rpmm
mirblood_expression_rpmmm = pd.read_csv(expression_path, sep='\t')

output_path = snakemake.output.avg_rpmm_expression

rna_col = mirblood_expression_rpmmm.columns[0]
samples = metadata["Sample"].to_list()


# selected_samples = metadata.groupby("cell_type", group_keys=False).apply(lambda x: x.sample(frac=1.0, random_state=42))
# selected_sample_names = selected_samples["Sample"].to_list()
#Subset the expression matrix to include miRNA column + only the selected samples
expression_subset = mirblood_expression_rpmmm[[rna_col] + samples]

sample_to_type = dict(zip(metadata["Sample"], metadata["cell_type"]))

new_columns = [rna_col] + [sample_to_type.get(c, c) for c in expression_subset.columns if c != rna_col]
expression_subset.columns = new_columns

rna_values = expression_subset[rna_col]
expr_only = expression_subset.drop(columns=[rna_col])

expr_avg = expr_only.groupby(expr_only.columns, axis=1).mean()

# Add RNA column back to the front
expr_avg.insert(0, rna_col, rna_values)

# --- Save result ---
# expr_avg.to_csv("expression_avg_by_celltype_rpmm.tsv", sep="\t", index=False)
expr_avg.to_csv(output_path, sep="\t", index=False)

