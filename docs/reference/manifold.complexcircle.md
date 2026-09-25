# Complex Phases and Unitary Matrices

Complex Phases and Unitary Matrices

## Usage

``` r
manifold.complexcircle(p)

manifold.unitary(p)
```

## Arguments

- p:

  Ambient vector or matrix dimension.

## Value

A complex manifold with real Hermitian inner products and real scalar
objectives. \`complexcircle\` represents p independent unit-modulus
phases. \`unitary\` represents square matrices U with U\* U = I.

## Details

Use complex64 or complex128 points with float32 or float64 losses,
respectively. Automatic Hessians are qualified for complex Euclidean
space, sphere and circle. Complex frames currently provide first-order
geometry. Circle logarithms exclude antipodes; Grassmann logarithms
exclude the cut locus.

## Examples

``` r
if(torch::torch_is_installed()) {
  M <- manifold.complexcircle(3)
  x <- riem.random(M,device="cpu")
  P <- riem.problem(M,function(x) -x$real$sum())
  riem.optimize(P,x,device="cpu")$objective
  riem.random(manifold.unitary(2),device="cpu")
}
#> torch_tensor
#> ℹ Use `$real` or `$imag` to print the contents of this tensor.
#> [ CPUComplexDoubleType{2,2} ]
```
