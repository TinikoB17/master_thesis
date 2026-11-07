import scanpy as sc
from snakemake.io import expand

def gather_projects(h5ad):
    ann_obj = sc.read_h5ad(h5ad, backed='r')
    project_values = ann_obj.obs["Project"].unique().astype(str).tolist()

    return expand("figures/sex_distribution/{val}.svg", val=project_values)
