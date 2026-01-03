import pandas as pd

rpmm_dataframe = snakemake.input.rpmm_quantified
metadata = snakemake.input.metadata

output = snakemake.output.filtered_rpmm

rpmm_df = pd.read_csv(rpmm_dataframe, sep='\t')
rpmm_df = rpmm_df.rename(columns={"miRNA": "Sample"})
rpmm_df.set_index("Sample", inplace=True)

rpmm_t = rpmm_df.T

mirblood_metadata = pd.read_csv(metadata, sep='\t').set_index("Sample")

rpmm_t["cell_type"] = mirblood_metadata["cell_type"]

mask = rpmm_t.drop(columns=["cell_type"]) > 1 
proportion = mask.groupby(rpmm_t["cell_type"]).mean()
cols_to_keep = proportion.columns[(proportion >= 0.5).any()]
filtered_df = rpmm_t[cols_to_keep]


filtered_df_t = filtered_df.T
filtered_df_t = filtered_df_t.reset_index().rename(columns={"Sample": "miRNA"}).set_index("miRNA")

filtered_df_t.to_csv(output, sep='\t')
