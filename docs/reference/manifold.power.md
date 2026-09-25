# Repeated and Scaled Manifold Geometries

Repeated and Scaled Manifold Geometries

## Usage

``` r
manifold.power(manifold, copies)

manifold.scaled(manifold, scale)
```

## Arguments

- manifold:

  Base manifold with tensor points.

- copies:

  Positive number of repeated factors.

- scale:

  Positive distance multiplier; the metric is multiplied by scale
  squared.

## Value

A manifold specification. Power points have shape \`c(copies,
base_shape)\`. Those factors belong to one optimization point;
additional leading axes still denote independent batches. Scaling
preserves the base representation.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.power(manifold.sphere(3), 4)
  x <- riem.random(M,device="cpu")
  riem.belongs(M,x)
  manifold.scaled(manifold.sphere(3),2)
}
#> <riem_manifold> scaled  | metric: scaled_round  | dimension: 2 
```
