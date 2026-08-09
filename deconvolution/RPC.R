library("DeconmiR")

reference <- read.table("deconvolution/dataframes/mirblood_avg_expression_per_cell_type_all_samples_rpmm_filtered.csv", header=TRUE, row.names = 1, sep='\t')
reference.m <- apply(reference, 2, as.numeric)
rownames(reference.m) <- rownames(reference)

samples <- read.table("to_deconvolute_test.csv", header=TRUE, row.names = 1, check.names = FALSE, sep='\t')
samples.m <- apply(samples, 2, as.numeric)
# samples.m <- as.matrix(sapply(samples, function(x) as.numeric(as.character(x))))
rownames(samples.m) <- rownames(samples)

est.o <- DeconmiR(samples.m, reference.m, method="RPC")
BloodFrac.m <- est.o$estF

write.table(BloodFrac.m, file='filtered_input/deconvoluted_tissueatlas_test.csv', row.names=TRUE, sep='\t', quote=FALSE)
