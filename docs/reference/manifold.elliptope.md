# Fixed-Rank Elliptope and Spectrahedron

Fixed-Rank Elliptope and Spectrahedron

## Usage

``` r
manifold.elliptope(p, k)

manifold.spectrahedron(p, k)
```

## Arguments

- p:

  Matrix dimension.

- k:

  Rank; elliptope requires k \>= 2, spectrahedron k \>= 1.

## Value

Embedded Frobenius geometry of PSD matrices with unit diagonal
(elliptope) or unit trace (spectrahedron).

## Details

Tangents are orthogonal projections onto the intersection of the
rank-stratum tangent and the diagonal/trace constraint. Retraction is
spectral rank truncation followed by diagonal/trace normalization.
Singular constraint strata and loss of rank are outside the supported
domain.

## Examples

``` r
if (torch::torch_is_installed()) {
  riem.random(manifold.elliptope(4, 2))
  riem.random(manifold.spectrahedron(4, 2))
}
#> torch_tensor
#>  0.2464  0.0399  0.1529 -0.0380
#>  0.0399  0.1337  0.2117 -0.1825
#>  0.1529  0.2117  0.3697 -0.2827
#> -0.0380 -0.1825 -0.2827  0.2503
#> [ CPUDoubleType{4,4} ]
```
