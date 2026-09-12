import pandas as pd
import umap
import seaborn as sns
import matplotlib.pyplot as plot

corrected = pd.read_csv("test_output/batch_corrected_exp.csv", sep='\t').set_index('miRNA')
metadata = pd.read_csv("test_output/blood_metadata_tissueatlas_healthy.csv", sep='\t').set_index("Sample")

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

emb = reducer.fit_transform(corrected)

pl = pd.DataFrame(emb, columns=["UMAP1", "UMAP2"], index=corrected.index)

df_plt = pl.join(metadata)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_plot, 
    x='UMAP1', 
    y='UMAP2', 
    hue='project_id',   # Change this to 'age' or 'sex' to color differently
    alpha=0.7,          # Slight transparency to see overlapping points
    s=20                # Dot size
)
plt.title('UMAP of miRNA Expression')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
