# Kendall Landmark Shape Geometry

Kendall Landmark Shape Geometry

## Usage

``` r
manifold.landmark(k, p, reflections = FALSE)
```

## Arguments

- k:

  Number of labeled landmarks.

- p:

  Spatial dimension, with k \> p.

- reflections:

  Whether to identify reflections as well as rotations. FALSE uses SO(p)
  (Kendall); TRUE uses O(p), matching Riemann's shape quotient.

## Value

Centered k by p preshapes of unit Frobenius norm, modulo SO(p).

## Details

The supported regular stratum has full column rank. Horizontal tangents
are centered, orthogonal to the preshape, and satisfy X' U symmetric.
Retraction normalizes a centered tangent step; transport projects
horizontally. Objectives must be invariant to rotations of the
representative.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.landmark(5, 2)
  riem.belongs(M, riem.random(M))
}
#> [1] TRUE
```
