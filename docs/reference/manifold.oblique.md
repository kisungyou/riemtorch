# Oblique Matrices and the Fisher–Rao Simplex

Oblique Matrices and the Fisher–Rao Simplex

## Usage

``` r
manifold.oblique(p, k)

manifold.multinomial(p, metric = "fisher_rao")
```

## Arguments

- p:

  Ambient dimension (number of rows or probabilities).

- k:

  Number of oblique columns.

- metric:

  Simplex metric, currently only \`"fisher_rao"\`.

## Value

A geometry specification.

## Details

Oblique columns have unit norm and the product round metric. The simplex
has strictly positive entries summing to one and metric sum(u\*v/x),
equivalent to a radius-two sphere under \`2\*sqrt(x)\`. The simplex
retraction is an exponential-coordinate normalization. Both geometries
provide exact exponential and local logarithm maps. The simplex
exponential rejects geodesics that leave its positive orthant.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.oblique(3, 2)
  x <- riem.random(M)
  u <- riem.tangent(M, x, torch::torch_randn_like(x)) * 0.1
  riem.log(M, x, riem.exp(M, x, u))

  S <- manifold.multinomial(4)
  riem.random(S)
}
#> torch_tensor
#>  0.2100
#>  0.0410
#>  0.3220
#>  0.4270
#> [ CPUDoubleType{4} ]
```
