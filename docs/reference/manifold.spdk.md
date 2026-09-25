# Fixed-Rank Positive-Semidefinite Geometry

Fixed-Rank Positive-Semidefinite Geometry

## Usage

``` r
manifold.spdk(p, k, metric, representation = c("matrix", "factor"))
```

## Arguments

- p:

  Matrix dimension.

- k:

  Rank.

- metric:

  Required metric, \`"embedded"\` or \`"wasserstein"\`.

- representation:

  \`"matrix"\` (p by p) or \`"factor"\` (p by k, available for
  Wasserstein).

## Value

A fixed-rank PSD manifold. Factor points Y represent Y Y'.

## Details

Embedded tangents eliminate the null-null block. Wasserstein uses the
quotient of full-column-rank factors by right orthogonal
transformations; matrix metric is 0.5 tr(L_X(U) V) with the
rank-restricted Sylvester inverse. Factor tangents are horizontal (Y' U
symmetric) and have Frobenius metric. The factor metric matches
Riemann's spdk factor geometry. Matrix and factor forms are isometric
under U -\> U Y' + Y U'. They are not the embedded metric.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.spdk(4, 2, "wasserstein", "factor")
  riem.belongs(M, riem.random(M))
}
#> [1] TRUE
```
