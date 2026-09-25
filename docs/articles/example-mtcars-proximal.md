# Sparse regression with a proximal operator

An L1 penalty can set regression coefficients exactly to zero. Its
corner at zero calls for a nonsmooth optimization method. On Euclidean
space, the exact proximal map is coordinatewise soft thresholding, so
this problem provides a transparent introduction to
[`riem.problem.composite()`](https://www.kisungyou.com/riemtorch/reference/riem.problem.composite.md).
The same interface supports other manifolds when the supplied proximal
map solves the corresponding manifold-distance subproblem.

See the [regression
introduction](https://www.kisungyou.com/riemtorch/articles/example-robust-regression.md)
for the basic problem interface and the [constraints and nonsmooth
guide](https://www.kisungyou.com/riemtorch/articles/constrained-nonsmooth.md)
for the distinction between an intrinsic proximal map and ambient
shrinkage.

## Data and scaling

The [`mtcars`
data](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/mtcars.html)
contain 32 cars from 1973–74. We use fuel consumption (`mpg`, miles per
US gallon) and five continuous predictors: displacement (`disp`, cubic
inches), horsepower (`hp`), rear axle ratio (`drat`), weight (`wt`,
thousands of pounds), and quarter-mile time (`qsec`). These selected
columns have no missing values. The data come with R, so this example
needs no download. [Data sources and
reproducibility](https://www.kisungyou.com/riemtorch/articles/example-data-sources.md)
records attribution.

``` r

predictors <- c("disp", "hp", "drat", "wt", "qsec")
cars <- datasets::mtcars
stopifnot(!anyNA(cars[, c("mpg", predictors)]))
X <- scale(as.matrix(cars[, predictors]))
y <- cars$mpg - mean(cars$mpg)
n <- nrow(X)
p <- ncol(X)
registered <- list(
  X = torch_tensor(X, dtype = torch_float64(), device = "cpu"),
  y = torch_tensor(y, dtype = torch_float64(), device = "cpu"),
  n = n
)
```

[`scale()`](https://rdrr.io/r/base/scale.html) subtracts each column
mean and divides by its sample standard deviation. Standardization makes
a common penalty meaningful across different predictor units. Centering
the response removes the unpenalized intercept; on the original response
scale it is restored by adding `mean(cars$mpg)` to predictions. No
train/test split is used: this is an optimization illustration, not
evidence of predictive superiority or a procedure for selecting a
penalty.

## Objective and exact proximal map

For each penalty $`\lambda`$, solve

``` math
\min_{\beta\in\mathbb R^5}
  \frac{1}{2n}\|X\beta-y\|_2^2 + \lambda\|\beta\|_1.
```

The smooth gradient is $`X^T(X\beta-y)/n`$. For a step $`t>0`$, the
Euclidean proximal map is
$`\operatorname{sign}(q)\max(|q|-t\lambda,0)`$, applied coordinatewise.
It exactly minimizes the nonsmooth term plus $`\|\beta-q\|^2/(2t)`$. On
a curved manifold, applying this operation and then projecting would
generally solve a different problem.

``` r

make_problem <- function(lambda) {
  riem.problem.composite(
    manifold.euclidean(p),
    fn = function(beta, data) {
      (data$X$matmul(beta) - data$y)$square()$mean() / 2
    },
    egrad = function(beta, data) {
      data$X$t()$matmul(data$X$matmul(beta) - data$y) / data$n
    },
    nonsmooth = function(beta, data) beta$abs()$sum() * data$lambda,
    prox = function(q, step, data) {
      q$sign() * (q$abs() - step * data$lambda)$clamp(min = 0)
    },
    data = c(registered, list(lambda = lambda))
  )
}
```

The explicit gradient avoids constructing an automatic differentiation
graph at every iteration. All objective tensors and the penalty are
registered as problem data.

## Follow a short penalty path

The smallest penalty for which zero satisfies the optimality conditions
is $`\lambda_{\max}=\|X^Ty/n\|_\infty`$. We start there and decrease the
penalty through five values. Each penalty defines a separate fixed
objective. Warm starting reuses the preceding solution as the next
initial point.

``` r

lambda_max <- max(abs(drop(crossprod(X, y)) / n))
penalty_ratio <- c(1, 0.5, 0.2, 0.1, 0.05)
lambdas <- lambda_max * penalty_ratio
L <- max(eigen(crossprod(X) / n, symmetric = TRUE, only.values = TRUE)$values)
initial <- torch_zeros(p, dtype = torch_float64(), device = "cpu")
fits <- vector("list", length(lambdas))
for (j in seq_along(lambdas)) {
  fits[[j]] <- riem.optimize(
    make_problem(lambdas[j]), initial, method = "proximal_gradient",
    control = list(max_iterations = 1500, step_size = 1.8 / L,
                   proximal_tolerance = 1e-5),
    device = "cpu"
  )
  initial <- fits[[j]]$point
}
coefficients <- vapply(fits, function(fit) as.numeric(fit$point), numeric(p))
rownames(coefficients) <- predictors
colnames(coefficients) <- paste0("ratio_", penalty_ratio)
round(coefficients, 4)
#>      ratio_1 ratio_0.5 ratio_0.2 ratio_0.1 ratio_0.05
#> disp       0   -0.3492   -0.0773    0.0000     0.0000
#> hp         0   -0.4751   -1.5275   -1.6438    -1.4334
#> drat       0    0.0000    0.2121    0.5500     0.7180
#> wt         0   -1.9916   -2.9575   -3.1918    -3.4098
#> qsec       0    0.0000    0.0000    0.2284     0.5855
status <- data.frame(
  penalty_ratio = penalty_ratio,
  lambda = lambdas,
  objective = vapply(fits, function(fit) fit$objective, numeric(1)),
  iterations = vapply(fits, function(fit) fit$iterations, numeric(1)),
  proximal_residual = vapply(fits, function(fit) fit$proximal_residual, numeric(1)),
  termination = vapply(fits, function(fit) fit$termination, character(1))
)
status
#>   penalty_ratio    lambda objective iterations proximal_residual
#> 1          1.00 5.0659212 17.594487          0      0.000000e+00
#> 2          0.50 2.5329606 14.175585        191      9.539852e-06
#> 3          0.20 1.0131842  8.447814        198      9.843224e-06
#> 4          0.10 0.5065921  5.834250        118      9.685077e-06
#> 5          0.05 0.2532961  4.344771        135      9.988864e-06
#>                   termination
#> 1 converged_proximal_residual
#> 2 converged_proximal_residual
#> 3 converged_proximal_residual
#> 4 converged_proximal_residual
#> 5 converged_proximal_residual
```

Here $`L`$ is the largest eigenvalue of the smooth Hessian. The initial
step `1.8 / L` is below `2 / L`; the solver also checks sufficient
decrease and backtracks when necessary. The stopping rule uses the
proximal gradient mapping, not the smooth gradient alone. An iteration
limit would mean more work is needed; we verify successful termination
below.

## Visualize coefficients as the penalty decreases

![Five standardized regression coefficients along decreasing L1
penalties, starting at zero and tracing separate colored
paths.](example-mtcars-proximal_files/figure-html/coefficient-path-1.png)

The paths are coefficients per sample-standard-deviation change in each
predictor. Zero coefficients are exact soft-thresholding results.

## Independent verification with coordinate descent

For comparison, implement cyclic coordinate descent directly in base R.
It updates one coefficient at a time using the residual with that
coordinate removed. The denominator is the actual squared column norm
divided by `n`; with sample-standard-deviation scaling it is
`(n - 1) / n`, not one.

``` r

soft_threshold <- function(z, threshold) sign(z) * max(abs(z) - threshold, 0)
coordinate_descent <- function(lambda) {
  beta <- numeric(p)
  squared_norm <- colSums(X^2) / n
  for (iteration in seq_len(20000)) {
    previous <- beta
    for (j in seq_len(p)) {
      partial_residual <- y - drop(X %*% beta) + X[, j] * beta[j]
      score <- sum(X[, j] * partial_residual) / n
      beta[j] <- soft_threshold(score, lambda) / squared_norm[j]
    }
    if (max(abs(beta - previous)) < 1e-11) return(beta)
  }
  stop("Independent coordinate descent did not converge.")
}
reference <- vapply(lambdas, coordinate_descent, numeric(p))
coefficient_error <- apply(abs(coefficients - reference), 2, max)
```

At a nonzero coefficient, the gradient plus $`\lambda`$ times the
coefficient sign must vanish. At zero, the gradient must lie in
$`[-\lambda,\lambda]`$. These are the Lasso optimality conditions.
Checking them independently avoids relying only on a solver’s success
label.

``` r

kkt_residual <- function(beta, lambda) {
  gradient <- drop(crossprod(X, drop(X %*% beta) - y)) / n
  active <- abs(beta) > 1e-8
  residual <- pmax(abs(gradient) - lambda, 0)
  residual[active] <- abs(gradient[active] + lambda * sign(beta[active]))
  max(residual)
}
kkt <- vapply(seq_along(lambdas), function(j) {
  kkt_residual(coefficients[, j], lambdas[j])
}, numeric(1))
comparison <- data.frame(penalty_ratio = penalty_ratio,
                        maximum_coefficient_error = coefficient_error,
                        independent_kkt_residual = kkt)
comparison
#>            penalty_ratio maximum_coefficient_error independent_kkt_residual
#> ratio_1             1.00              0.000000e+00             0.000000e+00
#> ratio_0.5           0.50              8.974114e-05             7.445769e-06
#> ratio_0.2           0.20              9.967446e-05             7.973211e-06
#> ratio_0.1           0.10              6.302427e-05             6.844356e-06
#> ratio_0.05          0.05              6.500111e-05             7.059039e-06
stopifnot(all(status$termination == "converged_proximal_residual"),
          max(coefficient_error) < 5e-4,
          max(kkt) < 2e-5,
          max(abs(coefficients[, 1])) < 1e-8)
```

The tolerances allow small numerical differences between algorithms
while requiring close coefficient agreement and small optimality
residuals. The correlated predictors in this small historical dataset
make coefficient paths sensitive to modeling choices. A predictive
analysis would need an appropriate validation design, with preprocessing
estimated within each training split.

## Choose a device for your own run

This article explicitly uses CPU float64. In
[`riem.optimize()`](https://www.kisungyou.com/riemtorch/reference/riem.optimize.md),
change `device` to `"auto"` for automatic discovery, retain `"cpu"` to
force CPU even when CUDA is available, or choose `"cuda:0"` on a
compatible workstation. Registered data move to the resolved device
together with the point.

``` r

# Replace the device argument inside the loop above:
# device = "auto"    # automatic selection preserving the requested precision
# device = "cpu"     # explicit CPU override
# device = "cuda:0"  # explicit first CUDA GPU; errors if unavailable
```
