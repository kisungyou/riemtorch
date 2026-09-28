# Maintenance only: recreate website data from a local Riemann source checkout.
# No package installation, numerical backend, or download is needed.
# Source: https://github.com/kisungyou/Riemann/tree/e847692576f8b250c8140e1343b145f72caa379f
extract_riemann_snapshots <- function(source_dir,
    output_dir = "vignettes/articles/data") {
  expected <- c(cities = "e21b318bb16fcc0c3c0ee0675c26ce7b",
                hands = "8fd1f0d05668b37f9c32f0ec812c4154",
                ERP = "98447031b591125cec4d1be61ec5040e")
  paths <- file.path(source_dir, "data", paste0(names(expected), ".rda"))
  if (!all(file.exists(paths)) ||
      !identical(unname(tools::md5sum(paths)), unname(expected))) {
    stop("Use the unchanged Riemann data files from commit ",
         "e847692576f8b250c8140e1343b145f72caa379f (version 0.1.7).")
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  for (i in seq_along(paths)) {
    data_env <- new.env(parent = emptyenv())
    load(paths[[i]], envir = data_env)
    name <- names(expected)[[i]]
    snapshot <- file.path(output_dir, paste0("riemann-", name, ".rds"))
    saveRDS(data_env[[name]], snapshot, version = 2, compress = "xz")
    stopifnot(identical(readRDS(snapshot), data_env[[name]]))
  }
  invisible(file.path(output_dir, paste0("riemann-", names(expected), ".rds")))
}
