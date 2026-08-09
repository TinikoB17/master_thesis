import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

exp = pd.read_csv("test_output/rpmm_tissueatlas_blood_healthy.csv", sep='\t').set_index("miRNA").T
exp.index.name = "Sample"
exp.columns.name = None
md = pd.read_csv("test_output/blood_metadata_tissueatlas.csv", sep='\t')
md = md[(md["Disease_Condition"].isin(["Healthy"])) & (md["Project"] == "PPMI")]
merged = exp.merge(md, how="right", on="Sample")
merged = merged.drop(columns = ["Disease_Condition", "Sex", "Sequencing_Method", "Project", "Biotype", "Tissue",
        "Organ_system", "Species", "Sample"])

merged = merged[(merged["Age"] >= 40) & (merged["Age"] <= 80)]

increasing = ["hsa-miR-22-3p", "hsa-miR-134-5p", "hsa-miR-328-5p"]
merged["Age"] = (merged["Age"] / 5).round() * 5
merged = merged.groupby(["Age"]).mean()
for mi in increasing:
    pl = sns.lineplot(
        x=merged.index,
        y=np.log2(merged[mi])
    )
    plt.xlabel("Index")
    plt.ylabel(f"log2({mi})")
    fig = pl.get_figure()
    fig.savefig(f"increasing_mirnas/{mi}.svg")
    plt.close(fig)
breakpoint()
