if (.Platform$OS.type == "windows") {
  conda_prefix <- Sys.getenv("CONDA_PREFIX")
  if (nchar(conda_prefix) > 0) {
    rtools_bin <- file.path(conda_prefix, "Rtools", "bin")
    Sys.setenv(PATH = paste(rtools_bin, Sys.getenv("PATH"), sep = ";"))
  }
}

install.packages(
  c("devtools", "remotes", "TTR", "BiocManager", "SeuratObject"),
  repos = "https://cloud.r-project.org",
  ask = FALSE
)

install.packages(
  "https://cran.r-project.org/src/contrib/Archive/smoother/smoother_1.1.tar.gz",
  repos = NULL,
  type = "source"
)

BiocManager::install(c("glmGamPoi", "destiny"), ask = FALSE, update = FALSE)

remotes::install_github("satijalab/seurat", ref = "v5.3.1_scSLIDE_compatible", upgrade = "never")

install.packages(
  "BPCells",
  repos = c("https://bnprks.r-universe.dev", "https://cloud.r-project.org"),
  ask = FALSE
)

devtools::install_github("satijalab/scSLIDE", upgrade = "never")
