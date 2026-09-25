test_that("device policy honors overrides and reports execution metadata", {
  old_option <- getOption("riemtorch.device", NULL)
  old_environment <- Sys.getenv("RIEMTORCH_DEVICE", unset = NA_character_)
  on.exit({
    options(riemtorch.device = old_option)
    if (is.na(old_environment)) Sys.unsetenv("RIEMTORCH_DEVICE") else
      Sys.setenv(RIEMTORCH_DEVICE = old_environment)
  }, add = TRUE)

  Sys.setenv(RIEMTORCH_DEVICE = "cpu")
  options(riemtorch.device = NULL)
  from_environment <- riemtorch:::.resolve_execution()
  expect_equal(from_environment$source, "environment")
  expect_equal(from_environment$device_string, "cpu")

  options(riemtorch.device = "cpu")
  from_option <- riemtorch:::.resolve_execution()
  expect_equal(from_option$source, "option")
  expect_equal(from_option$device_string, "cpu")

  explicit <- riemtorch:::.resolve_execution(device = "cpu")
  expect_equal(explicit$source, "argument")
  expect_equal(explicit$requested, "cpu")
  expect_equal(explicit$dtype_string, "float64")
  expect_s3_class(riem.device("cpu"), "torch_device")

  # Explicit auto performs discovery instead of using the session default.
  automatic <- riemtorch:::.resolve_execution(device = "auto")
  expect_equal(automatic$source, "argument")
  expect_equal(automatic$requested, "auto")
  expect_true(nzchar(automatic$device_string))
})

test_that("device discovery and explicit availability failures are clear", {
  devices <- riem.devices(dtype = "float64")
  expect_s3_class(devices, "data.frame")
  expect_named(devices, c("device", "type", "index", "available",
                          "compatible", "reason", "model", "operations"))
  expect_true("cpu" %in% devices$device)
  expect_true(devices$compatible[devices$device == "cpu"])
  expect_equal(attr(devices, "dtype"), "float64")
  ledger<-riem.support("devices")
  expect_true(all(c("backend","dtype","status")%in%names(ledger)))

  count <- if (torch::cuda_is_available()) torch::cuda_device_count() else 0L
  expect_error(riem.device(paste0("cuda:", count + 10L)), "not available")
  expect_error(riem.device("not-a-device"), "Unsupported device")
  expect_error(riem.device("cpu", dtype = torch::torch_int64()),
               "floating point")
})

test_that("riem.to moves floating tensor trees and preserves index dtypes", {
  tree <- list(
    value = torch::torch_ones(3, dtype = torch::torch_float64()),
    nested = list(index = torch::torch_tensor(1:3,
      dtype = torch::torch_int64()),
      mask = torch::torch_tensor(c(TRUE, FALSE, TRUE),
        dtype = torch::torch_bool())))
  moved <- riem.to(tree, device = "cpu", dtype = "float32")
  expect_true(moved$value$dtype == torch::torch_float32())
  expect_true(moved$nested$index$dtype == torch::torch_int64())
  expect_true(moved$nested$mask$dtype == torch::torch_bool())
  expect_equal(moved$value$device$type, "cpu")
})

test_that("riem.to rebuilds device-bound geometry and registered problem state", {
  B <- torch::torch_diag(torch::torch_tensor(c(1, 2, 3),
    dtype = torch::torch_float64()))
  M <- manifold.stiefel.generalized(3, 2, B)
  converted <- riem.to(M, device = "cpu", dtype = "float32")
  expect_true(converted$specification$B$dtype == torch::torch_float32())
  x <- riem.random(converted, device = "cpu", dtype = "float32")
  expect_true(x$dtype == torch::torch_float32())
  expect_true(riem.belongs(converted, x, tol = 1e-5))

  data <- list(target = torch::torch_ones(3, 2,
    dtype = torch::torch_float64()),
    rows = torch::torch_tensor(1:3, dtype = torch::torch_int64()))
  P <- riem.problem(M, function(x, data) ((x - data$target)^2)$sum(),
                    data = data)
  moved_problem <- riem.to(P, device = "cpu", dtype = "float32")
  expect_true(moved_problem$data$target$dtype == torch::torch_float32())
  expect_true(moved_problem$data$rows$dtype == torch::torch_int64())
  expect_true(moved_problem$manifold$specification$B$dtype ==
              torch::torch_float32())
})

test_that("riem.random follows automatic policy and accepts CPU override", {
  x <- riem.random(manifold.sphere(4), device = "cpu", dtype = "float32")
  expect_equal(x$device$type, "cpu")
  expect_true(x$dtype == torch::torch_float32())
  expect_true(riem.belongs(manifold.sphere(4), x))

  product <- manifold.product(list(a = manifold.sphere(3),
                                   b = manifold.euclidean(2)))
  point <- riem.random(product, device = "cpu")
  expect_named(point, c("a", "b"))
  expect_true(all(vapply(point, function(z) z$device$type == "cpu",
                         logical(1))))
})

test_that("managed solves move registered state and record CPU overrides", {
  M<-manifold.sphere(3)
  target<-torch::torch_tensor(c(1,2,3),dtype=torch::torch_float64())
  P<-riem.problem(M,function(x,data)-torch::torch_sum(x*data),data=target)
  fit<-riem.optimize(P,riem.random(M,device="cpu"),
    control=list(max_iterations=2),device="cpu",dtype="float32")
  expect_equal(fit$execution$requested,"cpu")
  expect_equal(fit$execution$device,"cpu")
  expect_equal(fit$execution$dtype,"float32")
  expect_true(fit$point$dtype==torch::torch_float32())
  expect_true(riem.belongs(M,fit$point,tol=1e-5))
})

test_that("default feasibility tolerances follow floating precision", {
  B<-torch::torch_diag(torch::torch_tensor(c(1,2,3),dtype=torch::torch_float32()))
  M<-manifold.stiefel.generalized(3,2,B)
  x<-riem.random(M,device="cpu",dtype="float32")
  expect_true(riem.belongs(M,x))
  expect_true(riem.istangent(M,x,riem.tangent(M,x,torch::torch_randn_like(x))))
  expect_equal(riemtorch:::.default_tolerance(x),1e-5)
  expect_equal(riemtorch:::.default_tolerance(x$to(dtype=torch::torch_float64())),1e-7)
})
