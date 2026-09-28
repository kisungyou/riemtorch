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
#>  0.2188  0.0172  0.0182 -0.2716
#>  0.0172  0.1588  0.1539 -0.1676
#>  0.0182  0.1539  0.1492 -0.1643
#> -0.2716 -0.1676 -0.1643  0.4732
#> [ CPUDoubleType{4,4} ]
```
