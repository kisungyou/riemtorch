# Materialize or Extract Entries from a Compact Matrix Point

Materialize or Extract Entries from a Compact Matrix Point

## Usage

``` r
riem.materialize(manifold, x, rows = NULL, columns = NULL)
```

## Arguments

- manifold:

  A fixed-rank geometry with representation \`"svd"\`.

- x:

  A named list U, S, V representing U S V'. S is a full rank square
  core; allowing a full core makes autodiff valid at repeated singular
  values.

- rows, columns:

  Equal-length one-based integer entry indices. NULL for both
  materializes the complete matrix.

## Value

A matrix tensor, or a vector of requested entries without allocating the
full matrix. Tangents use U, S, V leaves for U_perp, core, V_perp, and
represent U_perp V' + U core V' + U V_perp'. They are not point
increments.

## Details

Objectives must depend only on the represented matrix, independently of
the choice of orthonormal factors. Compact points currently support one
solve at a time and first-order derivatives; no automatic Hessian is
claimed.

## Examples

``` r
if(torch::torch_is_installed()) {
  M <- manifold.fixedrank(10,8,2,representation="svd")
  x <- riem.random(M,device="cpu")
  riem.materialize(M,x,rows=c(1,4),columns=c(2,5))
}
#> torch_tensor
#> 0.01 *
#>  3.1750
#>  1.2177
#> [ CPUDoubleType{2} ]
```
