# Find a two-dimensional penguin subspace

How can four body measurements be summarized in two coordinates? PCA
answers this by finding a plane with small reconstruction error. The
plane is a point on a Grassmann manifold: different orthonormal bases
describe the same point. Here we optimize that plane directly and
compare the result with
[`prcomp()`](https://rdrr.io/r/stats/prcomp.html).

This builds on [the sphere PCA
introduction](https://www.kisungyou.com/riemtorch/articles/example-sphere-pca.md).
For the geometry, see [geometry and
metrics](https://www.kisungyou.com/riemtorch/articles/geometry-and-metrics.md)
and
[`manifold.grassmann()`](https://www.kisungyou.com/riemtorch/reference/manifold.stiefel.md).

## Data: four measurements, one common scale

The [Palmer Penguins
data](https://allisonhorst.github.io/palmerpenguins/) were collected by
Kristen Gorman and Palmer Station LTER. This site bundles the CC0
`penguins` data from `palmerpenguins` 0.1.1, so building this article
requires no download. See [data sources and
reproducibility](https://www.kisungyou.com/riemtorch/articles/example-data-sources.md)
for attribution and the snapshot checksum.

``` r

penguins <- read.csv("data/penguins.csv", stringsAsFactors = FALSE)
measurements <- c("bill_length_mm", "bill_depth_mm",
                  "flipper_length_mm", "body_mass_g")
complete <- complete.cases(penguins[, measurements])
X <- scale(as.matrix(penguins[complete, measurements]))
species <- factor(penguins$species[complete])
data.frame(total = nrow(penguins), retained = nrow(X),
           missing_measurements = sum(!complete))
#>   total retained missing_measurements
#> 1   344      342                    2
```

Bill and flipper measurements are in millimeters; mass is in grams.
Centering and dividing each column by its sample standard deviation
gives the four variables equal variance before optimization. Only
missing body measurements cause row removal; missing sex values do not.
Species labels color the final plot but are never inputs to the
objective.

## Objective and geometry

Let $`X`$ contain the standardized observations, and let $`Q`$ be a
$`4\times2`$ matrix with orthonormal columns. We minimize

``` math
  f(Q) = \frac{1}{2n}\|X-XQQ^\mathsf{T}\|_F^2.
```

Multiplying $`Q`$ by any orthogonal $`2\times2`$ matrix leaves its
projector $`QQ^\mathsf{T}`$, and therefore the objective, unchanged.
This is why we use
[`manifold.grassmann(4, 2)`](https://www.kisungyou.com/riemtorch/reference/manifold.stiefel.md).

``` r

geometry <- manifold.grassmann(4, 2)
observations <- torch_tensor(X, dtype = torch_float64(), device = "cpu")
problem <- riem.problem(
  geometry,
  function(Q, data) {
    residual <- data - data$matmul(Q)$matmul(Q$t())
    (residual^2)$sum() / (2 * data$size(1))
  },
  data = observations
)
initial <- riem.random(geometry, device = "cpu", dtype = "float64")
```

Registering tensors through `data` lets the solver move them together
with the point when the execution device changes. Automatic
differentiation computes the ambient derivative, and the geometry
supplies the tangent gradient.

## Solve and inspect termination

``` r

fit <- riem.optimize(
  problem, initial, method = "conjugate_gradient",
  control = list(max_iterations = 200, gradient_tolerance = 1e-7),
  device = "cpu", dtype = "float64"
)
fit
#> <riem_fit> conjugate_gradient on grassmann 
#>  Objective: 0.23617148  | gradient norm: 4.075e-08 
#>  Termination: converged_gradient  | iterations: 38
Q <- as.matrix(fit$point)
scores <- X %*% Q
```

`converged_gradient` means the metric gradient met the requested
tolerance. `max_iterations` means the iteration allowance was exhausted,
which alone does not establish convergence. Keep both the termination
reason and gradient norm when comparing solvers.

![Palmer Penguins projected onto the optimized two-dimensional plane,
with points colored by Adelie, Chinstrap, and Gentoo
species.](example-penguin-subspace_files/figure-html/penguin-projection-1.png)

Species are used only to color the projection. The optimized basis can
rotate without changing the plane.

The two displayed axes are coordinates in the fitted plane; they need
not equal the ordered first and second principal-component axes. This is
an exploratory representation, not a trained species classifier.

## A short finite-sum comparison

The same objective is the mean of one reconstruction loss per penguin.
[`riem.problem.finitesum()`](https://www.kisungyou.com/riemtorch/reference/riem.problem.finitesum.md)
can evaluate a selected minibatch in one tensor operation through
`batch_fn`.

``` r

finite_problem <- riem.problem.finitesum(
  geometry,
  function(Q, i, data) {
    observation <- data[i, ]
    residual <- observation - observation$matmul(Q)$matmul(Q$t())
    (residual^2)$sum() / 2
  },
  n = nrow(X),
  batch_fn = function(Q, indices, data) {
    batch <- data[indices, , drop = FALSE]
    residual <- batch - batch$matmul(Q)$matmul(Q$t())
    (residual^2)$sum(dim = 2) / 2
  },
  data = observations
)
torch_manual_seed(6102)
svrg_fit <- riem.optimize(
  finite_problem, initial, method = "svrg",
  control = list(max_iterations = 120, batch_size = 32, step_size = 0.15,
                 epoch_length = 20, full_evaluation_every = 20,
                 gradient_tolerance = 1e-7),
  device = "cpu", dtype = "float64"
)
fits <- list(conjugate_gradient = fit, svrg = svrg_fit)
data.frame(
  method = names(fits),
  objective = vapply(fits, function(x) x$objective, numeric(1)),
  full_gradient_norm = vapply(fits, function(x) x$gradient_norm, numeric(1)),
  termination = vapply(fits, function(x) x$termination, character(1))
)
#>                                method objective full_gradient_norm
#> conjugate_gradient conjugate_gradient 0.2361715       4.075100e-08
#> svrg                             svrg 0.2361729       1.173544e-03
#>                           termination
#> conjugate_gradient converged_gradient
#> svrg                   max_iterations
```

SVRG uses a full gradient at a snapshot plus transported minibatch
corrections. The Grassmann frame representation supports the local
logarithm needed to connect each snapshot to the current point. The
small steps here stay within that local domain; the path is not defined
at a Grassmann cut locus. Unsupported geometries or paths fail
explicitly. See [variance reduction and solver
controls](https://www.kisungyou.com/riemtorch/articles/variance-reduction.md)
for the contract.

This deliberately short run is a demonstration, not a speed benchmark.
Its final gradient is evaluated on all observations; the minibatch
gradient does not certify stationarity. A small data matrix usually
favors a deterministic method. More iterations or different steps may be
needed for SVRG to satisfy the same stopping tolerance.

## Independent verification

Compare projectors, because comparing signed basis entries would
penalize equivalent planes.
[`prcomp()`](https://rdrr.io/r/stats/prcomp.html) uses a separate
singular-value decomposition. The fraction of standardized variance
retained is $`\|XQ\|_F^2/\|X\|_F^2`$.

``` r

reference <- prcomp(X, center = FALSE, scale. = FALSE)
reference_Q <- reference$rotation[, 1:2, drop = FALSE]
projector_error <- max(abs(tcrossprod(Q) - tcrossprod(reference_Q)))
captured <- sum(scores^2) / sum(X^2)
reference_captured <- sum(reference$sdev[1:2]^2) / sum(reference$sdev^2)

angle <- pi / 6
rotation <- matrix(c(cos(angle), sin(angle), -sin(angle), cos(angle)), 2, 2)
rotated_Q <- Q %*% rotation
rotation_error <- max(abs(tcrossprod(Q) - tcrossprod(rotated_Q)))
reconstruction_value <- function(basis) {
  sum((X - X %*% tcrossprod(basis))^2) / (2 * nrow(X))
}
finite_value <- as.numeric(riem.evaluate(finite_problem, fit$point,
                                        gradient = FALSE)$value)
initial_value <- as.numeric(riem.evaluate(problem, initial,
                                         gradient = FALSE)$value)
data.frame(
  projector_error, captured, reference_captured, rotation_error,
  svrg_objective_gap = svrg_fit$objective - fit$objective
)
#>   projector_error captured reference_captured rotation_error svrg_objective_gap
#> 1    4.492956e-08 0.881568           0.881568   1.110223e-16       1.382973e-06
stopifnot(
  all(riem.belongs(geometry, fit$point)),
  projector_error < 1e-6,
  abs(captured - reference_captured) < 1e-10,
  rotation_error < 1e-12,
  abs(reconstruction_value(rotated_Q) - fit$objective) < 1e-10,
  abs(finite_value - fit$objective) < 1e-10,
  fit$gradient_norm < 1e-6,
  is.finite(svrg_fit$gradient_norm),
  svrg_fit$objective < initial_value
)
```

These tolerances check the plane, the objective normalization, and basis
invariance separately. A second seeded execution should reproduce the
reported numeric checks to absolute tolerance `1e-10` on the same
CPU/runtime.

## Device selection and interpretation

This article explicitly uses CPU float64. In your own session, the same
registered-data problem can select a suitable device automatically or
honor an explicit override:

``` r

riem.optimize(problem, initial, device = "auto", dtype = "float64")
riem.optimize(problem, initial, device = "cpu", dtype = "float64")
riem.optimize(problem, initial, device = "cuda:0", dtype = "float64")
```

The CUDA call requires an available compatible GPU. Results describe the
complete observed measurements after standardization; different scaling
or missing-data assumptions define a different analysis. The package
supplies the optimization tools rather than a separate statistical PCA
estimator.
