# Run from the riemtorch RStudio project directory. Requires documentation tools.
# Writes generated verification evidence and refreshes the gallery timing CSV.
stopifnot(file.exists("riemtorch.Rproj"))
for (package in c("rmarkdown", "callr", "torch", "riemtorch")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing package: ", package)
}
if (!rmarkdown::pandoc_available()) stop("Run in RStudio or configure RSTUDIO_PANDOC.")
if (!torch::torch_is_installed()) stop("Install the torch runtime before verification.")

root <- normalizePath(".")
out <- file.path(root, "development", "documentation", "public-data")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
articles <- c("example-sphere-pca", "example-robust-regression", "example-spd-mean",
              "example-matrix-completion", "example-penguin-subspace",
              "example-stackloss-regression", "example-airquality-completion",
              "example-covariance-means", "example-tree-constraints",
              "example-mtcars-proximal")
applications <- articles[5:10]
stopifnot(unname(tools::md5sum(file.path(root, "vignettes/articles/data/penguins.csv"))) ==
            "a06a0210251465a86fb970018292304d")

worker <- function(input, output, check_result) {
  # Any attempted dataset download via the usual base-R downloader is an error.
  trace("download.file", where = asNamespace("utils"), print = FALSE,
        tracer = quote(stop("Downloads are disabled during article verification.")))
  on.exit(untrace("download.file", where = asNamespace("utils")), add = TRUE)
  e <- new.env(parent = globalenv())
  rmarkdown::render(input, output_dir = output, envir = e, quiet = TRUE,
                    output_options = list(mathjax = NULL))
  if (!check_result) return(NULL)
  if (!exists("example_result", envir = e, inherits = FALSE)) {
    stop("Article must supply example_result for independent repeat verification.")
  }
  result <- e$example_result
  stopifnot(is.numeric(result), length(result) > 0, !is.null(names(result)),
            !anyDuplicated(names(result)), all(is.finite(result)))
  result
}

records <- vector("list", length(articles))
results <- vector("list", length(articles))
names(results) <- articles
for (i in seq_along(articles)) {
  article <- articles[[i]]
  elapsed <- numeric(2)
  snapshots <- vector("list", 2)
  for (run in 1:2) {
    destination <- file.path(out, paste0("run-", run), article)
    dir.create(destination, recursive = TRUE, showWarnings = FALSE)
    message("Rendering ", article, " (run ", run, ")")
    elapsed[[run]] <- unname(system.time({
      snapshots[run] <- list(callr::r(worker, args = list(
        input = file.path(root, "vignettes/articles", paste0(article, ".Rmd")),
        output = destination, check_result = article %in% applications
      ), stdout = file.path(destination, "render.log"),
      stderr = file.path(destination, "render-errors.log")))
    })[["elapsed"]])
  }
  difference <- NA_real_
  if (article %in% applications) {
    stopifnot(identical(names(snapshots[[1]]), names(snapshots[[2]])))
    difference <- max(abs(snapshots[[2]] - snapshots[[1]]) /
                        pmax(1, abs(snapshots[[1]])))
    if (difference > 1e-8) stop(article, ": repeated numerical results differ.")
  }
  records[[i]] <- data.frame(article = article, run1_seconds = elapsed[[1]],
    run2_seconds = elapsed[[2]], render_seconds = mean(elapsed),
    max_relative_difference = difference)
  results[[i]] <- snapshots
  # Preserve evidence progressively if a later article fails.
  write.csv(do.call(rbind, records[seq_len(i)]), file.path(out, "timings.csv"),
            row.names = FALSE)
  saveRDS(results, file.path(out, "numeric-results.rds"))
}

timings <- do.call(rbind, records)
write.csv(timings, file.path(root, "vignettes/articles/data/example-runtimes.csv"),
          row.names = FALSE)
writeLines(c(capture.output(sessionInfo()),
             paste("Platform:", paste(Sys.info(), collapse = "; ")),
             paste("Pandoc:", rmarkdown::pandoc_version()),
             "Timing includes fresh process startup and complete standalone rendering.",
             "New-application numerical repeat tolerance: 1e-8 relative to pmax(1, abs(result))."),
           file.path(out, "environment.txt"))
new <- timings[timings$article %in% applications, ]
if (any(new$run1_seconds > 30 | new$run2_seconds > 30)) {
  stop("An application exceeded the 30-second target; see timings.csv.")
}
if (max(sum(new$run1_seconds), sum(new$run2_seconds)) > 180) {
  stop("New application renders exceeded the three-minute aggregate target.")
}
print(timings, row.names = FALSE)
message("All application checks and repeated-render comparisons passed.")
