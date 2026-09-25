# Positive Doubly Stochastic Matrix Geometry

Positive Doubly Stochastic Matrix Geometry

## Usage

``` r
manifold.doublystochastic(p)
```

## Arguments

- p:

  Matrix dimension, at least two.

## Value

Strictly positive matrices with unit row and column sums, with the
Fisher metric sum(u\*v/x). No exact distance or Hessian conversion is
claimed.

## Details

Retraction exponentiates a tangent ratio then balances its rows and
columns. Tangent projection solves a gauge-fixed weighted normal system.
Boundary points with zero entries are outside the manifold.

## Examples

``` r
if(torch::torch_is_installed()) {
  M <- manifold.doublystochastic(3)
  x <- riem.random(M,device="cpu")
  x$sum(dim=1)
}
#> torch_tensor
#>  1.0000
#>  1.0000
#>  1.0000
#> [ CPUDoubleType{3} ]
```
