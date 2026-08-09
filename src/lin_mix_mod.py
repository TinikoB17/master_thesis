import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

expr = pd.read_csv("filtered_input/tissueatlas_blood_expression.csv", sep='\t')
md = pd.read_csv("test_output/blood_metadata_tissueatlas.csv", sep='\t')

expr_long = (
    pd.melt(frame=expr, id_vars="miRNA", var_name="Sample", value_name="expression")
    .rename(columns={"index": "miRNA"})
)
expr_merged = expr_long.merge(md, on='Sample', how='right')

data = expr_merged[expr_merged['Sex'].isin(['m', 'f'])]

# - Ensure types
data['Sex'] = data['Sex'].astype('category')
data['Project'] = data['Project'].astype('category')
data['Age'] = pd.to_numeric(data['Age'])

# - Center Age for numerical stability
data['Age_c'] = data['Age'] - data['Age'].mean()

# - Optional: log-transform expression (stabilizes variance)
data['expression_log'] = np.log2(data['expression'] + 1)

# 4. Run mixed models per miRNA
results = []

for mirna in data['miRNA'].unique():
    d = data[data['miRNA'] == mirna].copy()
    
    # Skip miRNAs with <2 projects or no variance in Age
    if d['Project'].nunique() < 2 or d['Age_c'].std() == 0:
        continue
    
    try:
        model = smf.mixedlm(
            "expression_log ~ Age_c + Sex",
            data=d,
            groups=d["Project"]
        ).fit(reml=True, disp=False)
        
        results.append({
            "miRNA": mirna,
            "beta_age": model.params.get("Age_c", np.nan),
            "se_age": model.bse.get("Age_c", np.nan),
            "p_age": model.pvalues.get("Age_c", np.nan),
            "var_project": model.cov_re.iloc[0, 0],
            "var_resid": model.scale
        })
        
    except Exception as e:
        # skip miRNAs with convergence issues
        print(f"Skipped {mirna} due to error: {e}")
        continue

# 5. Convert to DataFrame
results_df = pd.DataFrame(results)

# 6. Optional: sort by age effect size
results_df = results_df.sort_values(by="beta_age", key=abs, ascending=False)

# Quick look
print(results_df.head())
