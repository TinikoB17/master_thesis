import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# mirnas = pd.read_csv("test_output/changing_mirnas.csv")["miRNA"].to_list()

exp = pd.read_csv("test_output/expression_train.csv", sep='\t').set_index("Sample")
mirnas = exp.columns.tolist()
md = pd.read_csv("test_output/metadata_train.csv", sep='\t').set_index("Sample")
exp_age = exp.merge(md["Age"], how="inner", right_index=True, left_index=True)
changing_mirnas = pd.read_csv("test_output/changing_mirnas.csv", sep='\t')["miRNA"].tolist()


breakpoint()
# exp_age = exp_age[(exp_age["Age"] > 65)]

# breakpoint()


n_mirs = 0
print(len(changing_mirnas))
for mirna in changing_mirnas:
        x = exp_age["Age"]
        y = np.log2(exp_age[mirna] + 1)

        r, p = pearsonr(x, y)
        if p < 0.05:
            n_mirs += 1
            slope, intercept = np.polyfit(x, y, 1)
            line = slope * x + intercept

            plt.scatter(x, y, color="blue")
            plt.plot(x, line, color="red", label=f"Fit (r={r:.2f})")

            plt.xlabel("Age")
            plt.ylabel(f"{mirna} expression")

            plt.text(
                0.05, 0.95,
                f"r = {r:.2f}\np = {p:.2e}",
                transform=plt.gca().transAxes,
                verticalalignment="top"
            )

            plt.legend()
            plt.savefig(f"mirna_plots/{mirna}.svg")
            plt.close()

print(n_mirs)
