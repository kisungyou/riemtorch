# Fit a line with a robust loss

A least-squares problem separates the residual function from the choice
of loss. This makes it easy to compare a quadratic loss with a robust
alternative. We fit a line to a small synthetic data set containing
three positive outliers. Euclidean space is a manifold too, so the same
problem and solver interface applies without a geometric constraint.

## Generate observations and register the data

``` r

set.seed(51)
x <- seq(-2, 2, length.out = 41)
y <- 1 + 1.5 * x + rnorm(length(x), sd = 0.15)
outliers <- c(31, 35, 39)
y[outliers] <- y[outliers] + 3

design <- cbind(intercept = 1, slope = x)
registered <- list(
  X = torch_tensor(design, dtype = torch_float64(), device = "cpu"),
  y = torch_tensor(y, dtype = torch_float64(), device = "cpu")
)
```

The point has two coordinates: the intercept and the slope. The residual
callback returns one value for each observation, leaving the constructor
to apply the chosen loss.

``` r

coefficients <- manifold.euclidean(2)
make_problem <- function(loss) {
  riem.problem.leastsquares(
    coefficients,
    residual = function(beta, data) data$X$matmul(beta) - data$y,
    data = registered,
    loss = loss,
    loss_scale = 0.25
  )
}
```

With `loss = "linear"`, the objective is one half of the sum of squared
residuals. With `loss = "huber"`, each residual has a quadratic loss up
to an absolute value of `loss_scale`, and a linear loss beyond it. The
threshold is in the units of the response; the value `0.25` is a choice
for this simulated example, not an automatically estimated noise scale.

## Solve both problems

``` r

losses <- c("linear", "huber")
fits <- setNames(lapply(losses, function(loss) {
  riem.optimize(
    make_problem(loss),
    torch_zeros(2, dtype = torch_float64(), device = "cpu"),
    method = "levenberg_marquardt",
    control = list(max_iterations = 100, gradient_tolerance = 1e-7),
    device = "cpu"
  )
}), losses)

estimates <- t(vapply(fits, function(fit) as.numeric(fit$point), numeric(2)))
colnames(estimates) <- c("Intercept", "Slope")
round(rbind(Generating_line = c(1, 1.5), estimates), 4)
#>                 Intercept  Slope
#> Generating_line    1.0000 1.5000
#> linear             1.2185 1.7354
#> huber              1.0083 1.5276
```

Levenberg–Marquardt uses a positive-semidefinite Gauss–Newton
approximation. For robust losses it reweights individual residuals; that
approximation is not the full Hessian of the robust objective. Automatic
differentiation supplies the residual derivatives here.

![Observed responses with three marked outliers, and generating,
least-squares, and Huber regression
lines.](example-robust-regression_files/figure-html/fits-plot-1.png)

Huber loss reduces the effect of the three added outliers in this
example.

## Verify against independent R calculations

For ordinary least squares,
[`lm.fit()`](https://rdrr.io/r/stats/lmfit.html) gives a direct
reference. For Huber loss, write the scalar objective in base R and
minimize it with [`optim()`](https://rdrr.io/r/stats/optim.html). The
latter is a numerical cross-check rather than a closed-form solution.

``` r

least_squares_reference <- lm.fit(design, y)$coefficients
huber_objective <- function(beta) {
  residual <- drop(design %*% beta) - y
  threshold <- 0.25
  sum(ifelse(abs(residual) <= threshold,
             residual^2 / 2,
             threshold * (abs(residual) - threshold / 2)))
}
huber_reference <- optim(c(0, 0), huber_objective, method = "BFGS",
                         control = list(reltol = 1e-12))

comparison <- data.frame(
  loss = losses,
  maximum_coefficient_error = c(
    max(abs(estimates["linear", ] - least_squares_reference)),
    max(abs(estimates["huber", ] - huber_reference$par))
  ),
  gradient_norm = vapply(fits, function(fit) fit$gradient_norm, numeric(1)),
  termination = vapply(fits, function(fit) fit$termination, character(1))
)
comparison
#>          loss maximum_coefficient_error gradient_norm        termination
#> linear linear              2.209122e-12  1.119662e-10 converged_gradient
#> huber   huber              1.308997e-09  6.211976e-08 converged_gradient
stopifnot(huber_reference$convergence == 0,
          all(comparison$maximum_coefficient_error < 1e-5))
```

The two loss functions have different scales and meanings, so their raw
objective values should not be used to rank the fits. Compare the
coefficients, the residuals, and performance on data appropriate to your
application.

[`riem.problem.leastsquares()`](https://www.kisungyou.com/riemtorch/reference/riem.problem.leastsquares.md)
also supports `"soft_l1"` and `"cauchy"` losses. Nonquadratic losses
support diagonal residual weights. For automatic device selection,
replace the explicit CPU override in the solve with `device = "auto"`;
registered data travel with the problem.

Continue with [industrial stack-loss
measurements](https://www.kisungyou.com/riemtorch/articles/example-stackloss-regression.md)
for a public-data application, or add explicit modeling assumptions in
the [constrained tree-volume
example](https://www.kisungyou.com/riemtorch/articles/example-tree-constraints.md).
