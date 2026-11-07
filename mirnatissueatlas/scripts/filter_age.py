import scanpy as sc

rpmm_filtered_healthy = snakemake.input.rpmm_filtered_annotated
age_filtered = snakemake.output.age_filtered

# Projects and age range come from config.yaml (see rule filter_projects_age)
projects = snakemake.params.projects
min_age = snakemake.params.min_age
max_age = snakemake.params.max_age

inp = sc.read_h5ad(rpmm_filtered_healthy)

# An empty project list keeps every project
if projects:
    available = set(inp.obs["Project"].astype(str))
    missing = [proj for proj in projects if proj not in available]
    if missing:
        print(f"Warning: projects not found in the data: {missing}")
    inp = inp[inp.obs["Project"].astype(str).isin(projects)]
    if inp.n_obs == 0:
        raise ValueError(f"None of the configured projects are in the data: {projects}")

outp = inp[(inp.obs["Age"] >= min_age) & (inp.obs["Age"] <= max_age)].copy()

print(outp.obs["Age"].min(), outp.obs["Age"].max())
print(outp.obs["Project"].value_counts())
# single_sex_projects = []
# for proj in outp.obs["Project"].unique().tolist():
#     project_obs = outp[outp.obs["Project"] == proj]
#     if len(project_obs.obs["Sex"].unique().tolist()) == 1:
#         single_sex_projects.append(proj)

# print(single_sex_projects)
# outp = outp[~outp.obs["Project"].isin(single_sex_projects)].copy()

# print(outp.obs)

outp.write_h5ad(age_filtered)
