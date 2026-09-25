# Estimate Extreme Riemannian Hessian Eigenvalues

Estimate Extreme Riemannian Hessian Eigenvalues

## Usage

``` r
riem.hessian.spectrum(problem, x, iterations = 20L, tolerance = 1e-10)
```

## Arguments

- problem:

  A problem supporting exact Hessian-vector products.

- x:

  A feasible point.

- iterations:

  Maximum Lanczos steps, capped by intrinsic dimension.

- tolerance:

  Breakdown tolerance.

## Value

Ritz eigenvalues, tangent eigenvectors, residual norms, iteration count
and an explicit approximation qualification. A small residual does not
certify that an unseen more extreme eigenvalue does not exist.

## Examples

``` r
if (torch::torch_is_installed()) {
  P <- riem.problem(manifold.euclidean(2), function(x) (x*x)$sum())
  x <- torch::torch_ones(2, dtype = torch::torch_float64())
  riem.hessian.spectrum(P, x)$values
}
#> [1] 2
```
