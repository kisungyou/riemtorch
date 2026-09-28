# Rotations and Rigid Motions

Rotations and Rigid Motions

## Usage

``` r
manifold.rotation(p)

manifold.rigidmotion(d)
```

## Arguments

- p:

  Rotation matrix size.

- d:

  Spatial dimension of a rigid motion.

## Value

A rotation manifold SO(p), or a named product \`rotation\`,
\`translation\`.

## Details

SO(p) uses the embedded Frobenius metric, polar retraction, exact
ambient Hessian conversion, and the matrix-exponential geodesic. Rigid
motions use the product of that metric and the Euclidean translation
metric; this is a Riemannian product, not a bi-invariant SE(d) metric.

## Examples

``` r
if (torch::torch_is_installed()) {
  riem.random(manifold.rotation(3))
  riem.random(manifold.rigidmotion(2))
}
#> $rotation
#> torch_tensor
#> -0.6762  0.7367
#> -0.7367 -0.6762
#> [ CPUDoubleType{2,2} ]
#> 
#> $translation
#> torch_tensor
#>  1.2751
#>  0.7991
#> [ CPUDoubleType{2} ]
#> 
```
