import pandas as pd


inp = snakemake.input.blood_metadata
out = snakemake.output.top_diseases

md = pd.read_csv(inp, sep='\t')
print(md["Disease_Condition"].value_counts())
md = md[md["Disease_Condition"] != "Healthy"]
md = md.groupby(by="Disease_Condition")["Age"].agg(lambda x: x.max() - x.min())

md.to_csv(out, sep='\t')
