# Reproducible single-device smoke benchmark. Run from the package directory:
# Rscript development/benchmarks/device-run.R --device=auto --dtype=float64
library(torch)
library(riemtorch)

arguments <- commandArgs(trailingOnly = TRUE)
option_value <- function(name, default) {
  hit <- arguments[startsWith(arguments, paste0("--", name, "="))]
  if (!length(hit)) return(default)
  sub(paste0("^--", name, "="), "", hit[[1L]])
}
requested_device <- option_value("device", "auto")
requested_dtype <- option_value("dtype", "float64")
output <- option_value("output", "development/benchmarks/device-results.csv")
torch_manual_seed(2026)
torch_set_num_threads(1)

synchronize <- function(device) {
  if (identical(device$type, "cuda")) torch::cuda_synchronize()
}
resolved <- riem.device(requested_device, requested_dtype)

cases <- list(
  sphere = manifold.sphere(1000),
  stiefel = manifold.stiefel(500, 10, "euclidean"),
  spd = manifold.spd(32, "airm"))

rows <- lapply(names(cases), function(label) {
  M <- cases[[label]]
  host <- riem.random(M, device = "cpu", dtype = requested_dtype)
  transfer <- system.time(x <- riem.to(host, device = resolved,
                                       dtype = requested_dtype))["elapsed"]
  target <- riem.random(M, device = resolved, dtype = requested_dtype)
  P <- riem.problem(M, function(x, data) ((x-data)^2)$sum()/2, data = target)
  invisible(riem.optimize(P, x, control = list(max_iterations = 1),
                          device = resolved, dtype = requested_dtype))
  synchronize(resolved)
  elapsed <- system.time({
    fit <- riem.optimize(P, x, "steepest_descent",
      control = list(max_iterations = 10), device = resolved,
      dtype = requested_dtype)
    synchronize(resolved)
  })["elapsed"]
  data.frame(case = label, requested_device = requested_device,
    selected_device = fit$execution$device, dtype = fit$execution$dtype,
    transfer_seconds = unname(transfer), compute_seconds = unname(elapsed),
    objective = fit$objective, gradient_norm = fit$gradient_norm,
    constraint = max(unlist(fit$constraint_residuals)),
    iterations = fit$iterations, termination = fit$termination,
    stringsAsFactors = FALSE)
})
results <- do.call(rbind, rows)
write.csv(results, output, row.names = FALSE)
print(results)
