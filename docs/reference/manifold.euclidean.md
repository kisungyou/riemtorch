# Euclidean Tensor Geometry

Euclidean Tensor Geometry

## Usage

``` r
manifold.euclidean(shape, field = c("real", "complex"))
```

## Arguments

- shape:

  Positive integer point dimensions.

- field:

  Real (default) or complex scalar field.

## Value

A \`riem_manifold\` with the Frobenius metric.

## Examples

``` r
M <- manifold.euclidean(c(2, 3))
M
#> <riem_manifold> euclidean  | metric: euclidean  | dimension: 6 
```
