import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import scanpy as sc

exp = sc.read_h5ad("test_output/all_human_miRNA_rpmm_harmonized_cleaned_blood.h5ad")
expr = exp.to_df()
expr.index.name = "Sample"

expr = expr.reset_index()
md = exp.obs
breakpoint()
expr_long = (
    pd.melt(frame=expr, id_vars="Sample", var_name="miRNA", value_name="expression")
    .rename(columns={"index": "miRNA"})
)
expr_merged = expr.merge(md, on='Sample', how='right')
expr_merged = expr_merged.dropna()
breakpoint()

mirnas = expr_merged['miRNA'].unique()
anova_results = {}

for mirna in mirnas:
    data = expr_merged[expr_merged['miRNA'] == mirna]
    print(
    mirna,
    data['Sex'].nunique(),
    data['Disease_Condition'].nunique()
)
    # Convert categorical features
    for col in ['Sex', 'Disease_Condition']:
        data[col] = data[col].astype('category')
    
    model = smf.ols('expression ~ Age + Sex + Disease_Condition + Sequencing_Method', data=data).fit()
    anova_table = anova_lm(model, typ=2)
    anova_table['prop_variance'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
    anova_results[mirna] = anova_table

all_variance = pd.concat({mirna: df['prop_variance'] for mirna, df in anova_results.items()}, axis=1).T
all_variance.index.name = "miRNA"

all_variance = all_variance.dropna()
all_variance.T.to_csv("mixed_models_result.csv", sep='\t')
breakpoint()
