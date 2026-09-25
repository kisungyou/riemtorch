# Flat Torus in Angle Coordinates

Flat Torus in Angle Coordinates

## Usage

``` r
manifold.torus(d)
```

## Arguments

- d:

  Number of circular factors.

## Value

A manifold with d real angles, each interpreted modulo 2\*pi.

## Examples

``` r
if (torch::torch_is_installed()) riem.random(manifold.torus(2))
#> torch_tensor
#> -0.4008
#>  1.8734
#> [ CPUDoubleType{2} ]
```
