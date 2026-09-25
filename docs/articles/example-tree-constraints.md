# Fit a constrained tree-volume model

Suppose a volume model should increase with both diameter and height,
and multiplying both measurements by a common factor should multiply
predicted volume by its cube. These are **modeling assumptions**, not
consequences of fitting the data. They lead to a small regression
problem with both an equality and inequalities. Euclidean coefficients
let us inspect the constraint solver without introducing a geometric
representation at the same time.

Start with [the regression
introduction](https://www.kisungyou.com/riemtorch/articles/example-robust-regression.md)
if the problem-and-solver interface is new to you. The [constraints and
nonsmooth
guide](https://www.kisungyou.com/riemtorch/articles/constrained-nonsmooth.md)
explains the solver contract in more detail.

## Data: diameter, height, and timber volume

R’s [`trees`
data](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/trees.html)
contain 31 black cherry trees. The misleadingly named `Girth` column
records **diameter in inches**, not circumference. Height is in feet and
timber volume in cubic feet. All three variables are positive and
complete. No download is needed; see [data sources and
reproducibility](https://www.kisungyou.com/riemtorch/articles/example-data-sources.md).

``` r

trees <- datasets::trees
stopifnot(nrow(trees) == 31, !anyNA(trees), all(as.matrix(trees) > 0))
log_predictors <- log(as.matrix(trees[, c("Girth", "Height")]))
log_volume <- log(trees$Volume)
predictor_center <- colMeans(log_predictors)
response_center <- mean(log_volume)
X <- cbind(intercept = 1, sweep(log_predictors, 2, predictor_center))
y <- log_volume - response_center

registered <- list(
  X = torch_tensor(X, dtype = torch_float64(), device = "cpu"),
  y = torch_tensor(y, dtype = torch_float64(), device = "cpu")
)
```

Centering the log predictors and response improves numerical
conditioning and changes only the intercept. We do not standardize the
predictors, so the slope constraints keep their original meaning.

## Objective and constraints

Write the model in the original coordinates as

``` math
\log V_i \approx a + b_D\log D_i + b_H\log H_i,
\qquad b_D+b_H=3,\quad b_D\geq0,\quad b_H\geq0.
```

We minimize the mean squared log residual divided by two. In the
constructor, equalities have the form $`h(\beta)=0`$ and inequalities
have the form $`g(\beta)\leq0`$. Our optimization point contains the
centered intercept followed by the two slopes. Registered data are
passed to every callback, even when a particular callback needs only the
coefficients.

``` r

problem <- riem.problem.constrained(
  manifold.euclidean(3),
  fn = function(beta, data) {
    (data$X$matmul(beta) - data$y)$square()$mean() / 2
  },
  equality = function(beta, data) (beta[2:3]$sum() - 3)$reshape(1),
  inequality = function(beta, data) -beta[2:3],
  data = registered
)
```

[`riem.problem.constrained()`](https://www.kisungyou.com/riemtorch/reference/riem.problem.constrained.md)
uses automatic differentiation for the objective gradient and constraint
adjoints here. The augmented-Lagrangian solver repeatedly solves smooth
inner problems and updates its multipliers.

## Solve and inspect termination

``` r

initial <- torch_tensor(c(0, 1.5, 1.5), dtype = torch_float64(), device = "cpu")
fit <- riem.optimize(
  problem, initial, method = "augmented_lagrangian",
  control = list(max_iterations = 30, gradient_tolerance = 1e-7,
                 feasibility_tolerance = 1e-7,
                 inner_control = list(max_iterations = 100)),
  device = "cpu"
)
centered_coefficients <- as.numeric(fit$point)
coefficients <- c(
  intercept = centered_coefficients[1] + response_center -
    sum(centered_coefficients[2:3] * predictor_center),
  log_diameter = centered_coefficients[2],
  log_height = centered_coefficients[3]
)
round(coefficients, 6)
#>    intercept log_diameter   log_height 
#>    -6.185685     1.990668     1.009332
data.frame(termination = fit$termination, as.data.frame(fit$kkt))
#>     termination  feasibility stationarity complementarity dual_feasibility
#> 1 converged_kkt 2.241893e-08 5.157096e-08               0                0
```

`converged_kkt` means the feasibility, stationarity, complementarity,
and dual feasibility tests passed. An iteration budget or an
inner-solver failure is a different outcome and must not be reported as
convergence. These residuals usually give local optimality conditions;
the independent calculation below also takes advantage of this
particular problem’s convexity.

## Visualize the fit

![Observed timber volume against exponentiated fitted log volume for 31
trees, with a diagonal perfect-fit
line.](example-tree-constraints_files/figure-html/fit-plot-1.png)

The fit is optimized in log-volume units. Exponentiating a fitted log
value does not correct for retransformation bias.

## Independent verification

Eliminate the height slope by setting $`b_H=3-b_D`$. Let
$`z_i=\log D_i-\log H_i`$ and $`w_i=\log V_i-3\log H_i`$. After
profiling out the intercept, this is simple least squares in $`b_D`$,
restricted to $`[0,3]`$. The unrestricted slope is a ratio of centered
cross products; clipping it to this interval gives the constrained
minimum. This reference uses base R rather than another call to the
riemtorch solver.

``` r

z <- log_predictors[, 1] - log_predictors[, 2]
w <- log_volume - 3 * log_predictors[, 2]
z_centered <- z - mean(z)
w_centered <- w - mean(w)
reference_diameter <- min(3, max(0, sum(z_centered * w_centered) /
                                  sum(z_centered^2)))
reference <- c(intercept = mean(w) - reference_diameter * mean(z),
               log_diameter = reference_diameter,
               log_height = 3 - reference_diameter)
coefficient_error <- max(abs(coefficients - reference))

# Independently evaluate the Lagrangian gradient and all constraint residuals.
beta <- centered_coefficients
lambda <- as.numeric(fit$multipliers$equality)
mu <- as.numeric(fit$multipliers$inequality)
objective_gradient <- drop(crossprod(X, drop(X %*% beta) - y)) / nrow(X)
lagrangian_gradient <- objective_gradient + c(0, lambda, lambda) - c(0, mu)
independent_kkt <- c(
  feasibility = max(abs(sum(beta[2:3]) - 3), pmax(0, -beta[2:3])),
  stationarity = sqrt(sum(lagrangian_gradient^2)),
  complementarity = max(abs(mu * beta[2:3])),
  dual_feasibility = max(0, -mu)
)
rbind(riemtorch = coefficients, independent = reference)
#>             intercept log_diameter log_height
#> riemtorch   -6.185685     1.990668   1.009332
#> independent -6.185685     1.990667   1.009333
data.frame(maximum_coefficient_error = coefficient_error,
           as.list(independent_kkt), check.names = FALSE)
#>   maximum_coefficient_error  feasibility stationarity complementarity
#> 1              4.764956e-07 2.241893e-08 5.157096e-08               0
#>   dual_feasibility
#> 1                0
stopifnot(fit$termination == "converged_kkt",
          coefficient_error < 2e-5,
          max(independent_kkt) < 1e-6,
          max(abs(unlist(fit$kkt) - independent_kkt)) < 1e-8)
```

The fitted slopes are close to the familiar
diameter-squared-times-height relationship. That agreement does not
establish that the constraints are appropriate for another species or
population. Predictions shown here are in-sample and are not a forecast
accuracy assessment.

## Choose a device for your own run

All code above uses CPU float64 explicitly so the published result is
easy to reproduce.
[`riem.optimize()`](https://www.kisungyou.com/riemtorch/reference/riem.optimize.md)
accepts `device = "auto"` to discover a suitable device,
`device = "cpu"` to override an available GPU, or an indexed CUDA device
on a compatible workstation. Registered tensors travel with the problem.

``` r

# Change only the device argument in the solve above:
# device = "auto"    # automatic selection while preserving float64
# device = "cpu"     # explicit CPU override
# device = "cuda:0"  # explicit first CUDA GPU; errors if unavailable
```
