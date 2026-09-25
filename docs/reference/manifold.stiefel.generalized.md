# Generalized Orthogonality Geometries

Generalized Orthogonality Geometries

## Usage

``` r
manifold.stiefel.generalized(p, k, B)

manifold.grassmann.generalized(p, k, B)
```

## Arguments

- p, k:

  Ambient and frame dimensions.

- B:

  Fixed symmetric positive-definite torch tensor of size p by p.

## Value

A geometry with constraint X' B X=I and metric tr(U' B V).

## Details

Whitening uses the transpose of the Cholesky factor. B must already
share the point's device and dtype. Grassmann points identify frames
differing by right orthogonal transformations. B is copied at
construction. The whitening isometry also supplies exact ambient Hessian
conversion.

## Examples

``` r
if (torch::torch_is_installed()) {
  B <- torch::torch_eye(3, dtype = torch::torch_float64())
  M <- manifold.stiefel.generalized(3, 2, B)
  riem.random(M)
}
#> torch_tensor
#> -0.9530 -0.2964
#> -0.2953  0.9550
#> -0.0682  0.0064
#> [ CPUDoubleType{3,2} ]
```
