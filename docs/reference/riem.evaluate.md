# Evaluate Values and Metric Derivatives

Evaluate Values and Metric Derivatives

## Usage

``` r
riem.evaluate(problem, x, gradient = TRUE)

riem.hessian(problem, x, u)
```

## Arguments

- problem:

  A scalar, finite-sum or least-squares problem.

- x:

  A feasible point without batch axes.

- gradient:

  Whether to compute the gradient.

- u:

  Tangent direction for a Hessian-vector product.

## Value

\`riem.evaluate()\` returns value, optional metric gradient and counts.
\`riem.hessian()\` returns an exact Riemannian Hessian-vector product.

## Details

Automatic Hessian conversion is available for Euclidean, sphere, torus,
oblique, Euclidean-metric Stiefel, both Grassmann representations,
generalized Stiefel/Grassmann, rotations, AIRM SPD, and weighted
products of supported factors, when the objective supports double
backward. Objectives using custom first-order matrix logarithm/root
kernels require an analytic \`rhess\` for exact second-order methods.

## Examples

``` r
if (torch::torch_is_installed()) {
  P <- riem.problem(manifold.euclidean(2), function(x) torch::torch_sum(x^2)/2)
  x <- torch::torch_tensor(c(1, 2), dtype = torch::torch_float64())
  riem.evaluate(P, x)
  riem.hessian(P, x, x)
}
#> torch_tensor
#>  1
#>  2
#> [ CPUDoubleType{2} ]
```
