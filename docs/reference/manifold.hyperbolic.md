# Hyperbolic Ball and Hyperboloid

Hyperbolic Ball and Hyperboloid

## Usage

``` r
manifold.hyperbolic(d, model = c("poincare", "hyperboloid"), curvature = -1)
```

## Arguments

- d:

  Intrinsic dimension.

- model:

  \`"poincare"\` or \`"hyperboloid"\`.

- curvature:

  Strictly negative sectional curvature.

## Value

A manifold with d ball coordinates or d+1 hyperboloid coordinates.

## Details

Hyperboloid uses signature (-,+,...,+) and the positive-time sheet. Ball
metric is 4/(1-c\*\|\|x\|\|^2)^2 times Euclidean, where c=-curvature.
Retractions are local additive (ball) or timelike normalization
(hyperboloid). Exact exponential and logarithm maps are also available
for both models.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.hyperbolic(2, "hyperboloid")
  x <- riem.random(M)
  u <- riem.tangent(M, x, torch::torch_randn_like(x)) * 0.05
  y <- riem.exp(M, x, u)
  riem.log(M, x, y)
}
#> torch_tensor
#>  0.1688
#> -0.2223
#>  0.0272
#> [ CPUDoubleType{3} ]
```
