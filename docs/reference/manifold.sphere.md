# Round Sphere Geometry

Round Sphere Geometry

## Usage

``` r
manifold.sphere(p, field = c("real", "complex"))
```

## Arguments

- p:

  Ambient dimension; \`sphere(3)\` represents the two-dimensional
  sphere.

- field:

  Real (default) or complex scalar field.

## Value

A manifold with unit vector points, Euclidean tangents, and round
metric.

## Details

Retraction is normalization, transport is tangent projection. Exact
exponential and local logarithm are available. The intrinsic Hessian
includes the sphere connection correction.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.sphere(3)
  x <- riem.random(M)
  riem.belongs(M, x)
}
#> [1] TRUE
```
