# Run from the riemtorch project directory. Maintenance only, not a site-build step.
stopifnot(requireNamespace("palmerpenguins", quietly = TRUE))
if (as.character(utils::packageVersion("palmerpenguins")) != "0.1.1") {
  stop("Use palmerpenguins version 0.1.1 to reproduce this snapshot.")
}
snapshot_source <- system.file("extdata", "penguins.csv", package = "palmerpenguins")
snapshot_target <- file.path("vignettes", "articles", "data", "penguins.csv")
snapshot_md5 <- "a06a0210251465a86fb970018292304d"
stopifnot(unname(tools::md5sum(snapshot_source)) == snapshot_md5)
if (!dir.exists(dirname(snapshot_target))) {
  stop("Run this script from the riemtorch RStudio project directory.")
}
stopifnot(file.copy(snapshot_source, snapshot_target, overwrite = TRUE))
stopifnot(unname(tools::md5sum(snapshot_target)) == snapshot_md5)
message("Reproduced the unchanged palmerpenguins 0.1.1 CSV snapshot.")
