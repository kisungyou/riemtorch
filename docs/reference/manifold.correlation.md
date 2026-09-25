# Full-Rank Correlation Geometries

Full-Rank Correlation Geometries

## Usage

``` r
manifold.correlation(p, metric)
```

## Arguments

- p:

  Matrix size, at least two.

- metric:

  Required metric: \`"ecm"\`, \`"lec"\`, or \`"affine_quotient"\`.

## Value

A manifold of positive-definite p by p matrices with unit diagonal.

## Details

Let L=chol(C) be lower triangular and Theta(C)=diag(L)^-1 L. ECM pulls
back the Frobenius metric on the strictly lower part of Theta. LEC uses
the strictly lower nilpotent log(Theta). Their inverse chart forms L L'
from the unit-lower matrix and normalizes its diagonal. Retractions and
transports are exact chart operations. The affine quotient uses the AIRM
quotient by positive diagonal congruences, with a horizontal lift and a
normalized SPD retraction. No approximate chart is substituted for this
metric.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.correlation(3, "ecm")
  x <- riem.random(M)
  riem.belongs(M, x)
  riem.chart(M, x)
}
#> torch_tensor
#>  0.0000  0.0000  0.0000
#>  0.9830  0.0000  0.0000
#> -0.5224 -0.2493  0.0000
#> [ CPUDoubleType{3,3} ]
```
