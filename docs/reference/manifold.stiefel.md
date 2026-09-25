# Stiefel and Grassmann Frame Geometries

Stiefel and Grassmann Frame Geometries

## Usage

``` r
manifold.stiefel(p, k, metric, field = c("real", "complex"))

manifold.grassmann(
  p,
  k,
  embedding = c("frame", "projection"),
  field = c("real", "complex")
)
```

## Arguments

- p:

  Ambient row dimension.

- k:

  Number of orthonormal columns.

- metric:

  Required Stiefel metric, \`"euclidean"\` or \`"canonical"\`.

- field:

  Real (default) or complex scalar field.

- embedding:

  Grassmann point representation: \`"frame"\` (p by k) or
  \`"projection"\` (p by p orthogonal projectors, with half-Frobenius
  metric).

## Value

A geometry specification using polar retraction and projection
transport. Exact ambient Hessian conversion is available for
Euclidean-metric Stiefel and both Grassmann representations, and
canonical-metric Stiefel.

## Details

Grassmann objectives must be invariant to orthogonal changes of frame.
Projection representation uses the metric equivalent to the frame
metric, including its factor of one half.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.stiefel(4, 2, metric = "euclidean")
  x <- riem.random(M)
  u <- riem.tangent(M, x, torch::torch_randn_like(x))
  riem.ehess2rhess(M, x, u, x, u)

  manifold.grassmann(4, 2, embedding = "projection")
}
#> <riem_manifold> grassmann  | metric: projection_half_frobenius  | dimension: 4 
```
