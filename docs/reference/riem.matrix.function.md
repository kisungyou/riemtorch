# Stable Symmetric Matrix Functions

Matrix logarithm, exponential, square root and inverse square root, and
their Frechet differentials. Point axes are the last two dimensions.

## Usage

``` r
riem.matrix.function(x, type = c("log", "exp", "sqrt", "invsqrt"))

riem.matrix.frechet(x, u, type = c("log", "exp", "sqrt", "invsqrt"))
```

## Arguments

- x:

  A symmetric tensor, positive definite except for \`type = "exp"\`.

- type:

  One of \`"log"\`, \`"exp"\`, \`"sqrt"\`, or \`"invsqrt"\`.

- u:

  A symmetric perturbation with the same shape as \`x\`.

## Value

A tensor with the same shape, dtype and device as \`x\`.

## Details

Logarithm and roots use a spectral forward evaluation and a
divided-difference backward rule with continuous repeated-eigenvalue
limits. Their custom backward supports first derivatives only; double
backward is not a supported operation. The Frechet helper is a
value-level differential, not a differentiable replacement for a second
derivative. Exponential uses torch's native matrix exponential,
including its higher derivatives.

## Examples

``` r
if (torch::torch_is_installed()) {
  x <- torch::torch_eye(2, dtype = torch::torch_float64())
  riem.matrix.function(x, "log")
  riem.matrix.frechet(x, x, "log")
}
#> torch_tensor
#>  1  0
#>  0  1
#> [ CPUDoubleType{2,2} ]
```
