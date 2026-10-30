import pandas as pd
import scanpy as sc

# Same as model_preprocessing.py, except that the per-project z-score statistics
# (mean, std) are computed on the training set only and then applied to the test set.
# model_preprocessing.py computes them separately on the test set, which breaks for
# projects with only 1 test sample (std = NaN -> all features become 0).

eligible_mirnas = snakemake.input.threshold_miRNAs
h5ad_to_process = snakemake.input.blood_h5ad_healthy
test_h5ad = snakemake.input.test_h5ad
model_inp = snakemake.output.model_input
model_test = snakemake.output.model_test

# Optional: the files with all miRNAs are skipped when the rule doesn't ask for them
model_inp_all_mirnas = snakemake.output.get("model_inp_all")
model_test_all_mirnas = snakemake.output.get("model_test_all")
cols_to_add = ["Age", "Project", "Sex"]


def add_obs_columns(annotated_object: sc.AnnData, columns_to_add: list) -> pd.DataFrame:
    mirna_expression = annotated_object.to_df()
    obs_dataframe = sc.get.obs_df(annotated_object, keys=columns_to_add)
    return mirna_expression.join(obs_dataframe, how="left")


def compute_train_stats(train_df: pd.DataFrame, mirna_cols: list):
    grouped = train_df.groupby("Project", observed=True)[mirna_cols]
    project_means = grouped.mean()
    project_stds = grouped.std(ddof=1)

    global_mean = train_df[mirna_cols].mean()
    global_std = train_df[mirna_cols].std(ddof=1)

    return project_means, project_stds, global_mean, global_std


def apply_zscore(df: pd.DataFrame, mirna_cols: list, train_stats) -> pd.DataFrame:
    project_means, project_stds, global_mean, global_std = train_stats
    df = df.copy()
    projects = df["Project"].astype(str)

    known = projects.isin(project_means.index.astype(str))
    if (~known).any():
        print(f"Projects not in training set, using global train stats: {projects[~known].unique().tolist()}")

    means = project_means.reindex(projects).set_index(df.index)
    stds = project_stds.reindex(projects).set_index(df.index)
    means = means.fillna(global_mean)
    stds = stds.fillna(global_std)

    # Constant miRNAs within a project: centre only, so they become 0
    stds = stds.mask(stds == 0, 1.0).fillna(1.0)

    df[mirna_cols] = (df[mirna_cols] - means[mirna_cols]) / stds[mirna_cols]
    df[mirna_cols] = df[mirna_cols].fillna(0)
    return df


def select_outputs(df: pd.DataFrame, eligible: list, columns_to_add: list):
    expression_all = df.drop(columns=["Project"])
    expression_eligible = df[eligible + columns_to_add].drop(columns=["Project"])
    return expression_all, expression_eligible


ann_obj = sc.read_h5ad(h5ad_to_process)
ann_test = sc.read_h5ad(test_h5ad)

train_df = add_obs_columns(ann_obj, cols_to_add)
test_df = add_obs_columns(ann_test, cols_to_add)

mirna_cols = [col for col in train_df.columns if col not in cols_to_add]
eligible_miRNAs = pd.read_csv(eligible_mirnas)["miRNA"].to_list()

train_stats = compute_train_stats(train_df, mirna_cols)

exp_all_train, exp_train = select_outputs(apply_zscore(train_df, mirna_cols, train_stats), eligible_miRNAs, cols_to_add)
exp_all_test, exp_test = select_outputs(apply_zscore(test_df, mirna_cols, train_stats), eligible_miRNAs, cols_to_add)

print(exp_train)

exp_train.to_csv(model_inp, sep='\t', index_label="Sample")
exp_test.to_csv(model_test, sep='\t', index_label="Sample")

if model_inp_all_mirnas:
    exp_all_train.to_csv(model_inp_all_mirnas, sep='\t', index_label="Sample")
    exp_all_test.to_csv(model_test_all_mirnas, sep='\t', index_label="Sample")
