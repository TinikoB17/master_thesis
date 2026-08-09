import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

md = pd.read_csv("test_output/blood_metadata_tissueatlas_healthy.csv", sep='\t')
fig, ax = plt.subplots(figsize=(12, 8))
bp = sns.countplot(x="Project", data=md)
plt.tight_layout(rect=[0, 0.9, 1, 0.95])
plt.title("Project Sample Counts - Healthy")
plt.xticks(rotation=45)
p = bp.get_figure()
p.savefig("projects.svg")
plt.close()

fig, ax = plt.subplots(figsize=(12, 8))
b = sns.barplot(data=md, x="Project", y="Age", estimator=pd.Series.max, errorbar=None, ax=ax)
plt.tight_layout()
fig = b.get_figure()
fig.savefig("ages_max.svg")
