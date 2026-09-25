# Check a Manifold or User-Supplied Derivatives

Numerical checks return evidence, not a proof of correctness. Directions
should be repeated at several interior points. Compact fixed-rank
manifold checks materialize dense matrices for the retraction derivative
comparison. Unsupported Taylor tests are reported as inconclusive, never
silently replaced by another metric.

## Usage

``` r
riem.check.manifold(manifold, x = NULL, u = NULL, tolerance = NULL)

riem.check.gradient(problem, x, u = NULL, steps = NULL, tolerance = NULL)

riem.check.hessian(problem, x, u = NULL, steps = NULL, tolerance = NULL)

riem.check.adjoint(problem, x, u = NULL, tolerance = NULL)
```

## Arguments

- manifold:

  A manifold specification.

- x:

  A feasible point. For the manifold check, NULL generates one.

- u:

  Optional nonzero tangent direction; otherwise generated randomly.

- tolerance:

  Relative error tolerance. NULL uses the point precision.

- problem:

  Scalar or least-squares problem.

- steps:

  Positive finite-difference step sizes; NULL uses a precision-aware
  grid.

## Value

A \`riem_check\` list with status, checks, curves and details.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.sphere(3)
  x <- riem.random(M, device = "cpu")
  riem.check.manifold(M, x)$checks
  P <- riem.problem(M, function(x) -x[1])
  riem.check.gradient(P, x)$status
  riem.check.hessian(P, x)$status
  L <- riem.problem.leastsquares(manifold.euclidean(2), function(x) x * 2)
  riem.check.adjoint(L, torch::torch_ones(2, dtype = torch::torch_float64()))$status
}
#> [1] "pass"
```
