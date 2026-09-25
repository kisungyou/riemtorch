# Fixed-Rank Rectangular Matrix Geometry

Fixed-Rank Rectangular Matrix Geometry

## Usage

``` r
manifold.fixedrank(m, p, k, representation = c("matrix", "svd"))
```

## Arguments

- m, p:

  Matrix dimensions.

- k:

  Fixed rank, at most min(m,p).

- representation:

  Dense \`matrix\` (default) or compact \`svd\`; see
  \[riem.materialize()\] for compact point and tangent contracts.

## Value

An embedded Frobenius geometry with dense or compact matrix points.

## Details

Retraction truncates an SVD to rank k. Steps are valid only while the
selected singular values are positive and separated from discarded ones.
SVD/retraction derivatives at repeated singular values are not
advertised.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.fixedrank(4, 3, 2)
  riem.belongs(M, riem.random(M))
}
#> [1] TRUE
```
