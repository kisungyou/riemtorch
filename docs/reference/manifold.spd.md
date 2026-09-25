# Symmetric Positive-Definite Matrix Geometry

Symmetric Positive-Definite Matrix Geometry

## Usage

``` r
manifold.spd(p, metric)
```

## Arguments

- p:

  Matrix size.

- metric:

  Required metric: \`"airm"\` (affine invariant), \`"lerm"\` (log
  Euclidean), or \`"wasserstein"\` (Bures–Wasserstein). Descriptive
  aliases \`"affine_invariant"\`, \`"log_euclidean"\`, and
  \`"bures_wasserstein"\` are accepted.

## Value

A manifold of p by p positive-definite matrices with symmetric tangents.

## Details

AIRM uses tr(X^-1 U X^-1 V), a second-order polynomial retraction, and
congruence transport along the endpoint AIRM geodesic. LERM pulls back
the Frobenius metric through log, with exact chart retraction and
transport. Wasserstein uses 0.5 tr(L_X(U) V), where X L_X(U)+L_X(U) X=U;
its local exponential requires I+L_X(U) positive definite. Custom matrix
logarithm/root derivatives are first-order only.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.spd(2, metric = "airm")
  x <- torch::torch_eye(2, dtype = torch::torch_float64())
  riem.egrad2rgrad(M, x, x)
  riem.retr(M, x, 0.1 * x)
}
#> torch_tensor
#>  1.1050  0.0000
#>  0.0000  1.1050
#> [ CPUDoubleType{2,2} ]
```
