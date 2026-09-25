test_that("new exact exponential and local logarithm maps invert locally", {
  skip_if_not(torch::torch_is_installed())
  torch::torch_manual_seed(171)

  M_oblique <- manifold.oblique(4, 2)
  x_oblique <- riem.random(M_oblique)
  u_oblique <- riem.tangent(M_oblique, x_oblique, torch::torch_randn_like(x_oblique))
  u_oblique <- u_oblique * (0.15 / scalar(riem.norm(M_oblique, x_oblique, u_oblique)))

  M_simplex <- manifold.multinomial(4)
  x_simplex <- tensor(c(0.1, 0.2, 0.3, 0.4))
  u_simplex <- riem.tangent(M_simplex, x_simplex, tensor(c(0.3, -0.2, 0.1, -0.4)))
  u_simplex <- u_simplex * (0.08 / scalar(riem.norm(M_simplex, x_simplex, u_simplex)))

  M_ball <- manifold.hyperbolic(2, "poincare", curvature = -0.7)
  x_ball <- tensor(c(0.1, -0.15))
  u_ball <- tensor(c(0.03, 0.02))

  M_hyperboloid <- manifold.hyperbolic(2, "hyperboloid", curvature = -0.7)
  x_hyperboloid <- riem.project(M_hyperboloid, tensor(c(1, 0.15, -0.1)))
  u_hyperboloid <- riem.tangent(M_hyperboloid, x_hyperboloid, tensor(c(0.01, 0.03, -0.02)))

  cases <- list(
    list(M_oblique, x_oblique, u_oblique),
    list(M_simplex, x_simplex, u_simplex),
    list(M_ball, x_ball, u_ball),
    list(M_hyperboloid, x_hyperboloid, u_hyperboloid)
  )
  for (case in cases) {
    M <- case[[1]]; x <- case[[2]]; u <- case[[3]]
    y <- riem.exp(M, x, u)
    recovered <- riem.log(M, x, y)
    expect_true(all(riem.belongs(M, y)), info = M$metric)
    expect_true(all(riem.istangent(M, x, recovered)), info = M$metric)
    expect_lt(normf(recovered - u), 2e-8)
    expect_equal(scalar(riem.sqdist(M, x, y)),
                 scalar(riem.inner(M, x, u, u)), tolerance = 1e-7,
                 info = M$metric)
  }

  expect_error(riem.exp(manifold.multinomial(2), tensor(c(0.5, 0.5)),
                        tensor(c(-1, 1))), "positive orthant")

  # A full turn has a positive endpoint but crosses the boundary twice.  The
  # exponential on the open simplex must reject the path at its first exit.
  expect_error(riem.exp(manifold.multinomial(2), tensor(c(0.5, 0.5)),
                        tensor(c(-2 * pi, 2 * pi))), "positive orthant")
})

test_that("new maps are stable at zero displacement", {
  skip_if_not(torch::torch_is_installed())
  cases <- list(
    list(manifold.oblique(3, 2), NULL),
    list(manifold.multinomial(3), tensor(c(0.2, 0.3, 0.5))),
    list(manifold.hyperbolic(2), tensor(c(0.1, -0.2))),
    list(manifold.hyperbolic(2, "hyperboloid"), NULL)
  )
  for (case in cases) {
    M <- case[[1]]
    x <- if (is.null(case[[2]])) riem.random(M) else case[[2]]
    zero <- torch::torch_zeros_like(x)
    expect_true(normf(riem.exp(M, x, zero) - x) < 2e-12, info = M$metric)
    expect_true(normf(riem.log(M, x, x)) < 2e-12, info = M$metric)
    expect_true(scalar(riem.sqdist(M, x, x)) >= 0, info = M$metric)
    expect_true(scalar(riem.sqdist(M, x, x)) < 2e-12, info = M$metric)
  }
})

test_that("expanded ambient Hessian conversions are exact and self adjoint", {
  skip_if_not(torch::torch_is_installed())
  torch::torch_manual_seed(172)
  B <- torch::torch_diag(tensor(c(1, 2, 3, 4)))

  builders <- list(
    function() {
      M <- manifold.oblique(4, 2); x <- riem.random(M); target <- torch::torch_randn_like(x)
      list(M = M, x = x, fn = function(z) ((z - target)^2)$sum() / 2)
    },
    function() {
      M <- manifold.stiefel(4, 2, "euclidean"); x <- riem.random(M); target <- torch::torch_randn_like(x)
      list(M = M, x = x, fn = function(z) ((z - target)^2)$sum() / 2)
    },
    function() {
      M <- manifold.grassmann(4, 2); x <- riem.random(M)
      A <- torch::torch_randn(c(4, 4), dtype = x$dtype); A <- (A + A$t()) / 2
      list(M = M, x = x, fn = function(z) (z * A$matmul(z))$sum() / 2)
    },
    function() {
      M <- manifold.grassmann(4, 2, "projection"); x <- riem.random(M)
      target <- torch::torch_randn_like(x); target <- (target + target$t()) / 2
      list(M = M, x = x, fn = function(z) ((z - target)^2)$sum() / 2)
    },
    function() {
      M <- manifold.stiefel.generalized(4, 2, B); x <- riem.random(M); target <- torch::torch_randn_like(x)
      list(M = M, x = x, fn = function(z) ((z - target)^2)$sum() / 2)
    },
    function() {
      M <- manifold.grassmann.generalized(4, 2, B); x <- riem.random(M)
      A <- torch::torch_randn(c(4, 4), dtype = x$dtype); A <- (A + A$t()) / 2
      list(M = M, x = x, fn = function(z) (z * A$matmul(z))$sum() / 2)
    },
    function() {
      M <- manifold.rotation(3); x <- riem.random(M); target <- torch::torch_randn_like(x)
      list(M = M, x = x, fn = function(z) ((z - target)^2)$sum() / 2)
    }
  )

  for (build in builders) {
    case <- build(); M <- case$M; x <- case$x
    u <- riem.tangent(M, x, torch::torch_randn_like(x))
    v <- riem.tangent(M, x, torch::torch_randn_like(x))
    u <- u / (1 + scalar(riem.norm(M, x, u)))
    v <- v / (1 + scalar(riem.norm(M, x, v)))
    P <- riem.problem(M, case$fn)
    Hu <- riem.hessian(P, x, u); Hv <- riem.hessian(P, x, v)
    expect_true(M$capabilities$hessian, info = M$name)
    expect_true(all(riem.istangent(M, x, Hu)), info = M$name)
    expect_equal(scalar(riem.inner(M, x, Hu, v)),
                 scalar(riem.inner(M, x, u, Hv)), tolerance = 3e-7,
                 info = M$name)

    h <- 1e-4
    fp <- scalar(riem.evaluate(P, riem.retr(M, x, u, h), FALSE)$value)
    fm <- scalar(riem.evaluate(P, riem.retr(M, x, u, -h), FALSE)$value)
    f0 <- scalar(riem.evaluate(P, x, FALSE)$value)
    directional <- (fp + fm - 2 * f0) / h^2
    expect_equal(directional, scalar(riem.inner(M, x, u, Hu)),
                 tolerance = 3e-5, info = M$name)
  }

  expect_true(manifold.stiefel(4, 2, "canonical")$capabilities$hessian)
})

test_that("connection corrections annihilate ambient constraint energies", {
  skip_if_not(torch::torch_is_installed())
  torch::torch_manual_seed(175)
  B <- torch::torch_diag(tensor(c(1, 2, 3, 4)))
  cases <- list(
    list(M = manifold.oblique(4, 2),
         fn = function(z) (z^2)$sum() / 2),
    list(M = manifold.stiefel(4, 2, "euclidean"),
         fn = function(z) (z^2)$sum() / 2),
    list(M = manifold.grassmann(4, 2),
         fn = function(z) (z^2)$sum() / 2),
    list(M = manifold.grassmann(4, 2, "projection"),
         fn = function(z) torch::torch_trace(z)),
    list(M = manifold.stiefel.generalized(4, 2, B),
         fn = function(z) (z * B$matmul(z))$sum() / 2),
    list(M = manifold.grassmann.generalized(4, 2, B),
         fn = function(z) (z * B$matmul(z))$sum() / 2),
    list(M = manifold.rotation(3),
         fn = function(z) (z^2)$sum() / 2)
  )
  for (case in cases) {
    M <- case$M; x <- riem.random(M)
    u <- riem.tangent(M, x, torch::torch_randn_like(x))
    Hu <- riem.hessian(riem.problem(M, case$fn), x, u)
    expect_true(normf(Hu) < 2e-8, info = paste(M$name, M$metric))
  }
})

test_that("weighted products preserve exact Hessian scaling", {
  skip_if_not(torch::torch_is_installed())
  torch::torch_manual_seed(173)
  M <- manifold.product(list(a = manifold.oblique(3, 2),
                             b = manifold.stiefel(4, 2, "euclidean")),
                        weights = c(2, 5))
  x <- riem.random(M)
  target <- list(a = torch::torch_randn_like(x$a), b = torch::torch_randn_like(x$b))
  P <- riem.problem(M, function(z) ((z$a - target$a)^2)$sum() / 2 +
                                      ((z$b - target$b)^2)$sum() / 2)
  direction <- function() list(
    a = riem.tangent(M$factors$a, x$a, torch::torch_randn_like(x$a)),
    b = riem.tangent(M$factors$b, x$b, torch::torch_randn_like(x$b)))
  u <- direction(); v <- direction()
  u <- riemtorch:::.scale(u, 1 / (1 + scalar(riem.norm(M, x, u))))
  v <- riemtorch:::.scale(v, 1 / (1 + scalar(riem.norm(M, x, v))))
  Hu <- riem.hessian(P, x, u); Hv <- riem.hessian(P, x, v)
  expect_true(M$capabilities$hessian)
  expect_equal(scalar(riem.inner(M, x, Hu, v)),
               scalar(riem.inner(M, x, u, Hv)), tolerance = 3e-7)

  h <- 1e-4
  fp <- scalar(riem.evaluate(P, riem.retr(M, x, u, h), FALSE)$value)
  fm <- scalar(riem.evaluate(P, riem.retr(M, x, u, -h), FALSE)$value)
  f0 <- scalar(riem.evaluate(P, x, FALSE)$value)
  expect_equal(scalar(riem.inner(M, x, u, Hu)),
               (fp + fm - 2 * f0) / h^2, tolerance = 3e-5)
})

test_that("capability records expose the added exact primitives", {
  skip_if_not(torch::torch_is_installed())
  for (M in list(manifold.oblique(3, 2), manifold.multinomial(3),
                 manifold.hyperbolic(2), manifold.hyperbolic(2, "hyperboloid"))) {
    cap <- riem.capabilities(M)
    expect_true(all(cap$value[cap$operation %in% c("exp", "log")]), info = M$metric)
  }
  for (M in list(manifold.oblique(3, 2), manifold.stiefel(4, 2, "euclidean"),
                 manifold.grassmann(4, 2), manifold.grassmann(4, 2, "projection"),
                 manifold.stiefel.generalized(4, 2, torch::torch_eye(4, dtype = torch::torch_float64())),
                 manifold.grassmann.generalized(4, 2, torch::torch_eye(4, dtype = torch::torch_float64())),
                 manifold.rotation(3),
                 manifold.product(list(a = manifold.oblique(3, 2),
                                       b = manifold.euclidean(2))))) {
    cap <- riem.capabilities(M)
    expect_true(cap$value[cap$operation == "ehess2rhess"], info = M$name)
  }
})

test_that("second-order solvers consume an expanded Hessian conversion", {
  skip_if_not(torch::torch_is_installed())
  torch::torch_manual_seed(174)
  M <- manifold.stiefel(4, 2, "euclidean")
  target <- riem.random(M); initial <- riem.random(M)
  P <- riem.problem(M, function(x) ((x - target)^2)$sum() / 2)
  fit <- riem.optimize(P, initial, "trust_regions",
                       list(max_iterations = 60, inner_tolerance = 0.01))
  expect_lt(fit$objective, 1e-12)
  expect_gt(fit$evaluations$hessian, 0)
})
