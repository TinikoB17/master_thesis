import scanpy as sc

rpmm_filtered_healthy = snakemake.input.rpmm_filtered_annotated
age_filtered = snakemake.output.age_filtered

inp = sc.read_h5ad(rpmm_filtered_healthy)

outp = inp[(inp.obs["Age"] >= 30) & (inp.obs["Age"] <= 80)].copy()

print(outp.obs["Age"].min(), outp.obs["Age"].max())
# single_sex_projects = []
# for proj in outp.obs["Project"].unique().tolist():
#     project_obs = outp[outp.obs["Project"] == proj]
#     if len(project_obs.obs["Sex"].unique().tolist()) == 1:
#         single_sex_projects.append(proj)

# print(single_sex_projects)
# outp = outp[~outp.obs["Project"].isin(single_sex_projects)].copy()

# print(outp.obs)

outp.write_h5ad(age_filtered)
