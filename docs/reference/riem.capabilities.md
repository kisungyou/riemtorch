# Inspect Primitive-Level Geometry Capabilities

Inspect Primitive-Level Geometry Capabilities

## Usage

``` r
riem.capabilities(manifold)
```

## Arguments

- manifold:

  A geometry specification.

## Value

A data frame with operation, value, first_derivative, second_derivative,
and native_batch columns. FALSE derivative entries mean unsupported or
unverified, not necessarily mathematical nonsmoothness.
Membership/random operations are not differentiated. Domain restrictions
still apply.

## Details

A first-order optimizer usually only needs values of retraction and
gradient conversion. It does not imply differentiating those primitives.
A true value for \`ehess2rhess\` records a verified exact conversion
from an ambient Hessian-vector product, rather than a projected
approximation. Product records take the intersection of all factors'
capabilities.

## Examples

``` r
riem.capabilities(manifold.stiefel(4, 2, "euclidean"))
#>      operation value first_derivative second_derivative native_batch
#> 1      belongs  TRUE            FALSE             FALSE        FALSE
#> 2      tangent  TRUE             TRUE              TRUE         TRUE
#> 3        inner  TRUE             TRUE              TRUE         TRUE
#> 4  egrad2rgrad  TRUE             TRUE              TRUE         TRUE
#> 5         retr  TRUE             TRUE             FALSE         TRUE
#> 6    transport  TRUE             TRUE              TRUE         TRUE
#> 7  ehess2rhess  TRUE            FALSE             FALSE         TRUE
#> 8          exp  TRUE            FALSE             FALSE        FALSE
#> 9          log FALSE            FALSE             FALSE        FALSE
#> 10      sqdist FALSE            FALSE             FALSE        FALSE
#> 11     project  TRUE             TRUE             FALSE         TRUE
#> 12      random  TRUE            FALSE             FALSE         TRUE
```
