import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
corrs = pd.read_csv("test_output/correlations.csv", sep='\t')


plt.figure()
sns.set_theme(style="whitegrid")
xmin = -1
xmax = 1
plt.xticks(np.arange(-1, 1.1, 0.5))
plt.axvline(-0.1, color="purple", ls="--")
plt.axvline(0.1, color="purple", ls="--")


plt.xlim(xmin, xmax)
histogram = sns.histplot(data=corrs, x="Pearson_corr", kde=True)
plt.ylabel("Number of miRNAs")
plt.xlabel("Pearson Correlation Coefficient with Age")

plt.title(f"Pearson Correlation Distribution")
hist = histogram.get_figure()
hist.savefig(f"figures/Pearson_correlation_histogram.svg")
