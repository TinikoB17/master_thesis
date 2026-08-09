import plotly.graph_objects as go
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

data_blood = pd.read_csv("test_output/blood_metadata_tissueatlas.csv", sep='\t')


seq_method_counts = data_blood["Sequencing_Method"].value_counts()

fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
colors = [ "olivedrab", "salmon", "lightgreen", "sandybrown", "crimson", "mediumslateblue", "darkseagreen", "tan",
          "maroon", "rosybrown", "steelblue", "lightcoral", "mediumaquamarine", "peru", "indianred", "lightsteelblue"]
bars = ax.barh(seq_method_counts.index, seq_method_counts.values, color=colors)
# ax.tick_params(axis='y', labelrotation=90)
ax.set_xlabel("N of Samples", fontsize=16)

plt.tight_layout()
plt.title("Sample Count by Sequencing Method", fontsize=20)
plt.subplots_adjust(top=0.9, left=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig("count_Sequencing_Method.svg")
