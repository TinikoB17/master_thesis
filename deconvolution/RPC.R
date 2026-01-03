library("DeconmiR")

reference <- read.table("deconvolution/expression_avg_by_celltype_rpmm.csv", header=TRUE, row.names = 1, sep='\t')
reference.m <- apply(reference, 2, as.numeric)
rownames(reference.m) <- rownames(reference)

samples <- read.table("filtered_input/tissueatlas_blood_expression.csv", header=TRUE, row.names = 1, check.names = FALSE, sep='\t')
samples.m <- apply(samples, 2, as.numeric)
# samples.m <- as.matrix(sapply(samples, function(x) as.numeric(as.character(x))))
rownames(samples.m) <- rownames(samples)

est.o <- DeconmiR(samples.m, reference.m, method="RPC")
BloodFrac.m <- est.o$estF

write.table(BloodFrac.m, file='filtered_input/deconvoluted_tissueatlas.csv', row.names=TRUE, sep='\t', quote=FALSE)