# Positive Tensors with a Log-Euclidean Metric

Positive Tensors with a Log-Euclidean Metric

## Usage

``` r
manifold.positive(shape)
```

## Arguments

- shape:

  Positive point dimensions.

## Value

A positive-tensor manifold with metric sum(u\*v/x^2), exact maps,
isometric transport and exact ambient Hessian conversion.

## Examples

``` r
if(torch::torch_is_installed()) {
  M <- manifold.positive(3)
  riem.random(M,device="cpu")
}
#> torch_tensor
#>  0.2743
#>  5.8222
#>  1.9203
#> [ CPUDoubleType{3} ]
```
