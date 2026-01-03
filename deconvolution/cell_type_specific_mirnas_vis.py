import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filt_exp = snakemake.input.filtered_rpmm
metadata_path = snakemake.input.metadata
miR = snakemake.params.mirna

print(miR)
# mirna_list
filtered_exp = pd.read_csv(filt_exp, sep='\t').set_index("miRNA")
# high_tsi_miRNAs = pd.read_csv("dataframes/mirblood_rpmm_with_tsi.csv", sep='\t').set_index("miRNA")
mirblood_metadata = pd.read_csv(metadata_path, sep='\t').set_index("Sample")
# # expression = snakemake.input.filtered_rpmm
# # tsi = snakemake.input.tsi
# high_tsi_miRNAs = high_tsi_miRNAs[high_tsi_miRNAs["TSI"] >= 0.8]
# expression_high_tsi_mirnas = filtered_exp.loc[high_tsi_miRNAs.index]

transposed = filtered_exp.T
transposed.index.name = "Sample"
transposed.columns.name = None
# transposed.reset_index(name="Sample", inplace=True).drop(columns=["miRNA"])

transposed["cell_type"] = mirblood_metadata["cell_type"]



plt.figure(figsize=(10, 6))
boxplt = sns.boxplot(x="cell_type", y=miR, data=transposed, hue="cell_type")
plt.title(f'Expression of {miR} in Cell Type Groups')
plt.ylabel(f'Expression of {miR}')
plt.xticks(rotation=70, ha='right')
plt.tight_layout()
tr_fig = boxplt.get_figure()
tr_fig.savefig(f"plots/{miR}_boxplot.svg")
