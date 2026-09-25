# Correlation Chart Coordinates

Correlation Chart Coordinates

## Usage

``` r
riem.chart(manifold, x)

riem.chart.inverse(manifold, z)
```

## Arguments

- manifold:

  An ECM or LEC correlation manifold.

- x:

  A correlation point or batch of points.

- z:

  Strictly lower triangular chart coordinates, with square point axes.

## Value

A tensor of chart coordinates or correlation matrices.

## Details

Both chart and inverse use native differentiable torch operations.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.correlation(2, "lec")
  x <- riem.random(M)
  riem.chart.inverse(M, riem.chart(M, x))
}
#> torch_tensor
#>  1.0000  0.5696
#>  0.5696  1.0000
#> [ CPUDoubleType{2,2} ]
```
