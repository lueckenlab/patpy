#!/usr/bin/env Rscript

# install.packages(c("optparse", "devtools"))
# devtools::install_github("epurdom/gloscope")

# Load required libraries
library(optparse)
library(GloScope)

# Define options and parse command-line arguments
option_list <- list(
  make_option(c("-d", "--df_path"), type="character", help="Path to the embedding DataFrame"),
  make_option(c("-s", "--sample_ids_path"), type="character", help="Path to the sample IDs"),
  make_option(c("-m", "--dist_mat"), default="KL", type="character", help="Distance matrix: 'KL' or 'JS'"),
  make_option(c("-n", "--dens"), default="GMM", type="character", help="Density estimation: 'GMM' or 'KNN'"),
  make_option(c("-k", "--k"), default=25, type="integer", help="Integer k for k-NN"),
  make_option(c("-r", "--seed"), default=0, type="integer", help="Random seed"),
  make_option(c("-o", "--output_path"), type="character", help="Path to the output file"),
  make_option(c("-w", "--n_workers"), default=1, type="integer", help="Number of workers"),
  make_option(c("-v", "--verbose"), default=FALSE, action="store_true", help="Print verbose messages")
)

parser <- OptionParser(option_list=option_list)
arguments <- parse_args(parser, args = commandArgs(trailingOnly = TRUE))
verbose = arguments$verbose

if(verbose) {
    print("Reading data frame")
}
embedding_df <- read.csv(arguments$df_path, row.names = 1)
if(verbose) {
    print(paste("Data dimensions:", dim(embedding_df)))
    print(head(embedding_df))
}

if(verbose) {
    print("Reading sample IDs")
}
sample_ids <- read.csv(arguments$sample_ids_path)[[1]]  # Only take the first column (it should be the only one)
if(verbose) {
    print(paste("Number of IDs:", length(sample_ids)))

    print("Calculating distance matrix")
}

# Call your gloscope function
dist_matrix <- gloscope(
    embedding_df,
    sample_ids,
    dens = arguments$dens,
    dist_mat = arguments$dist_mat,
    k = arguments$k,
    BPPARAM = BiocParallel::MulticoreParam(workers = arguments$n_workers, RNGseed = arguments$seed)
)
if(verbose) {
    print(paste("Distance matrix dimensions:", dim(dist_matrix)))
    print("Writing distance matrix to the file")
}

write.csv(dist_matrix, arguments$output_path)
