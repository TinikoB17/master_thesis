import scanpy as sc

rpmm_filtered_healthy = snakemake.input.rpmm_filtered_annotated
age_filtered = snakemake.output.age_filtered

inp = sc.read_h5ad(rpmm_filtered_healthy)

outp = inp[(inp.obs["Age"] >= 25) & (inp.obs["Age"] <= 85)].copy()
print(outp.obs["Age"].min(), outp.obs["Age"].max())
outp.write_h5ad(age_filtered)
