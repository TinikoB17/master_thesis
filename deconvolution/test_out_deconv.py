import pandas as pd

list_of_ref_samples = pd.read_csv("deconvolution/samples_for_reference.csv", sep='\t')
expr = pd.read_csv("deconvolution/mirblood_original_rpmmm.csv", sep='\t').drop(columns=list_of_ref_samples["SampleNames"].to_list())

cols = expr.sample(n=10, axis="columns", random_state=42).columns.to_list()
expr = expr[["miRNA"] + cols]

# expr = expr.drop(columns=["miRNA"]).sample(n=10, axis="columns", random_state=42)

breakpoint()
expr.to_csv("test_deconv.csv", sep='\t', index=False)
