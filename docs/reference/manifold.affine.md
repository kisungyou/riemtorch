# Affine Subspaces and Orthogonal Matrices

Affine Subspaces and Orthogonal Matrices

## Usage

``` r
manifold.affine(basis, offset = NULL)

manifold.orthogonal(p)
```

## Arguments

- basis:

  Matrix with orthonormal columns spanning the affine directions.

- offset:

  Vector locating the affine subspace; defaults to zero.

- p:

  Dimension of an orthogonal matrix.

## Value

A Frobenius geometry. The orthogonal group includes both determinant
signs.

## Examples

``` r
if(torch::torch_is_installed()) {
  B <- torch::torch_eye(3,dtype=torch::torch_float64())[,1:2]
  M <- manifold.affine(B)
  riem.belongs(M,riem.random(M,device="cpu"))
  manifold.orthogonal(3)
}
#> <riem_manifold> orthogonal  | metric: frobenius  | dimension: 3 
```
