# Thesis Outline (Draft)

**Working title:** Predicting Chronological Age from Circulating miRNA Expression: A Multi-Cohort Machine-Learning Study in Healthy Human Blood

*Status: skeleton generated from the current state of the analysis pipeline (2026-10-30). Numbers marked (prelim.) come from runs that may still change. Items marked TODO are not done yet. All references must be verified before citing.*

## Abstract (write last)

- One sentence each: motivation (aging clocks), data (944 healthy blood small-RNA-seq samples from 7 cohorts), approach (pipeline + linear/Lasso/boosting models), key result (MAE ~ X years vs. 8.7 baseline), main insight (cohort/batch effects and evaluation design dominate), implication.

## 1. Introduction

- Aging and the need for molecular aging biomarkers ("biological age", age acceleration as a health indicator).
- Existing molecular clocks: DNA methylation clocks (e.g. Horvath 2013, Hannum 2013), transcriptomic and proteomic clocks.
- Why miRNAs: stable in blood, minimally invasive, regulators of aging-related pathways; age-associated miRNAs in peripheral blood (e.g. Huan et al. 2018).
- Problem: miRNA datasets are small and heterogeneous; pooling public cohorts introduces strong batch effects.
- Research questions:
  - RQ1: How well can chronological age be predicted from blood miRNA expression across pooled public cohorts?
  - RQ2: Which preprocessing, feature selection and model choices matter most?
  - RQ3: How strongly do cohort/batch effects and the evaluation design influence the reported accuracy?
- Contributions: reproducible Snakemake pipeline; leakage-free repeated nested CV evaluation; comparison of model families and feature sets; analysis of cohort effects.
- Thesis structure (one short paragraph).

## 2. Background

### 2.1 microRNAs

- Biogenesis (pri-/pre-miRNA, hairpin, mature 5p/3p arms), function, circulating miRNAs.
- miRBase annotation and the high-confidence miRNA subset.

### 2.2 Small RNA sequencing

- Library prep, sequencing platforms (the data includes Illumina HiSeq 2000/2500, MiSeq, NextSeq 500, NovaSeq 6000).
- Normalization: reads per million mapped miRNA reads (RPMM), log transformation.

### 2.3 Batch effects in multi-cohort data

- Sources: lab, protocol, platform, cohort design. Confounding with age (cohorts differ in age distribution).
- Correction approaches: per-batch standardization (used here), ComBat and related methods, mixed models.

### 2.4 Machine learning for age regression

- Linear regression, regularization (Lasso), polynomial features, gradient-boosted trees (HistGradientBoosting).
- Feature selection: univariate filters (correlation, F-test), embedded (Lasso).
- Evaluation: MAE, RMSE, R^2; hold-out vs. k-fold vs. repeated nested CV; data leakage; grouped (leave-one-cohort-out) validation; bootstrap confidence intervals.
- Regression toward the mean in age predictors (young over-predicted, old under-predicted).

### 2.5 Related work

- miRNA-based age predictors and their reported accuracy, cohort sizes and validation designs.
- Comparison table to fill: study, tissue, n, cohorts, method, validation, MAE / R^2.

## 3. Data

### 3.1 Source

- Harmonized human miRNA expression atlas (AnnData, 46,997 samples x 2,656 miRNAs, RPMM) plus sample metadata (age, sex, disease condition, project, sequencing method).
- TODO: describe origin, accession and licence of the atlas; cite the PPMI study for the largest cohort.

### 3.2 Sample and feature selection (funnel figure)

- High-confidence miRNAs: 1,315 mature miRNAs from miRBase high-confidence hairpins; 1,060 present in the data.
- Complete metadata and known sex: 9,420 samples, 68 tissues, 50 projects.
- Blood (tissue biotype): 7,150 samples, 11 projects (mostly PD, SDAI, genetic and prodromal cohorts).
- Healthy blood: 984 samples, 8 projects.
- Age 30-80: 944 samples, 7 projects; 584 male / 360 female; mean age 59.7 (SD 11.0).
- Strong imbalance: PPMI contributes ~85% of samples (799); the other 6 cohorts have 3-51 samples each.
- Figure: funnel chart (src/funnel_chart.py).

### 3.3 Cohort characteristics

- Age histograms per subset and train/test (figures/*_age_histogram.svg).
- Sex distribution per cohort (figures/sex_distribution/).
- Sequencing platforms per cohort.
- Table: cohort, n, age mean/range, sex ratio, platform.

## 4. Exploratory Analysis

- UMAPs coloured by tissue, project, sequencing method and age (all samples and blood): samples cluster by project/platform rather than age -> motivates batch handling.
- Detection/prevalence: number of miRNAs expressed above 1 RPMM vs. fraction of samples; threshold of 20% -> ~352 miRNAs.
- Univariate association with age: Pearson/Spearman per miRNA; ~71 miRNAs with |r| >= 0.15 and p < 0.05; distribution of correlations (src/histogram_correlations.py); example miRNA trajectories (src/plot_changing_miRNAs.py).
- Variance partitioning: share of variance explained by project vs. age vs. sex (scripts/variance_partitioning.py, src/lin_mix_mod.py).
- TODO: decide which exploratory figures go in the main text vs. appendix.

## 5. Methods

### 5.1 Pipeline overview

- Snakemake workflow: cleaning -> blood/healthy subsetting -> age filter -> split -> prevalence filter + log2 -> correlation analysis -> per-project z-scoring -> models -> evaluation.
- Figure: workflow DAG (snakemake --rulegraph).

### 5.2 Preprocessing

- Prevalence filter (RPMM > 1 in >= 20% of samples), log2(1 + x).
- Per-project z-score normalization; statistics estimated on training data only; fallback to global statistics for unseen or very small cohorts (< 5 samples).
- Yeo-Johnson power transform for skewed features; one-hot encoded sex.

### 5.3 Feature sets

- "Correlated": age-associated miRNAs (correlation filter).
- "All": all miRNAs passing the prevalence filter.
- Additional univariate selection (SelectKBest, F-test) tuned as a hyperparameter.

### 5.4 Models

- Multiple linear regression, polynomial (degree 2) regression, Lasso, HistGradientBoosting.
- Hyperparameter grids (table in appendix).

### 5.5 Evaluation design

- Baseline: predict the training mean age.
- Initial design: stratified 90/10 hold-out (project x age bin) + 5-fold CV for tuning.
- Final design: repeated nested CV (5 outer folds x 10 repeats, 5-fold inner CV); every data-dependent step refit inside each training fold.
- Uncertainty: fold spread and bootstrap confidence intervals.
- TODO: leave-one-project-out evaluation for cross-cohort generalization.
- Metrics: MAE (primary), RMSE, R^2, median AE, error by age group.

## 6. Results

### 6.1 Hold-out results

- Table: all models, CV MAE, test MAE, R^2, RMSE.
- Best: Lasso (all miRNAs) test MAE 7.64, R^2 0.24; HGB test MAE 7.59, R^2 0.23.
- Effect of correcting test-set normalization (train vs. test statistics): changes of up to ~0.3 years; two samples previously had all-zero features.

### 6.2 Uncertainty of the hold-out estimate

- Only 95 test samples (81 from PPMI); bootstrap 95% CI for MAE about +-1.2 years; Lasso vs. HGB difference CI includes 0; models are not distinguishable on the hold-out.

### 6.3 Repeated nested cross-validation (prelim.)

- Table: MAE mean +- SD, R^2 per model, baseline MAE 8.66.
- Linear regression (correlated) 8.49, Lasso (correlated) 8.46, linear regression (all) 8.05.
- TODO: Lasso (all), polynomial, HGB.
- Finding: the all-miRNA feature set outperforms the pre-selected correlated set; hold-out estimates were optimistic.
- Stability of selected hyperparameters across folds.

### 6.4 Error analysis

- Predicted vs. actual age plots; regression toward the mean.
- Error by age group (30-40, ..., 70-80) and by cohort.
- Influence of tiny cohorts on per-project normalization (failure case: 3-sample cohort -> exploding predictions).

### 6.5 Age-associated miRNAs

- TODO: Lasso coefficients / permutation importance; stability of selected miRNAs across folds; overlap with literature-reported age miRNAs.

### 6.6 Cross-cohort generalization

- TODO: leave-one-project-out results per cohort.

## 7. Discussion

- Answer to each research question.
- Accuracy in context: modest improvement over baseline vs. published miRNA and methylation clocks; why (cross-cohort heterogeneity, narrow healthy age range 30-80, sample size, single dominant cohort).
- Evaluation pitfalls: leakage through feature selection and normalization outside CV; optimistic small hold-out sets -> methodological lesson.
- Batch correction trade-off: per-project z-scoring removes cohort mean differences but also age differences between cohorts (age-cohort confounding).
- Limitations: cross-sectional data, "healthy" definition depends on metadata, PPMI dominance, RPMM normalization, missing covariates (blood cell composition, medication).
- Future work: disease cohorts and "miRNA age acceleration" (e.g. PD vs. healthy, data already available in blood subset), other tissues, sex-specific models, alternative batch correction (ComBat, mixed models), larger external validation cohort.

## 8. Conclusion

- Short summary of findings and the main take-away.

## Appendix

- A. Software and reproducibility: Snakemake version, Python packages, Docker/devcontainer, how to run the pipeline.
- B. Hyperparameter grids for all models.
- C. Full per-fold results and additional figures (all UMAPs, per-cohort sex distributions).
- D. List of age-associated miRNAs with correlation statistics.

## Suggested next steps before writing

1. Finish repeated nested CV (Lasso all, polynomial, HGB) and add leave-one-project-out evaluation.
2. Feature importance / stable miRNA list for Section 6.5.
3. Clarify data provenance and citations for the atlas and cohorts.
4. Fix pipeline reproducibility (targets without rules, append-mode evaluation files) so all thesis figures are regenerated by `snakemake`.
