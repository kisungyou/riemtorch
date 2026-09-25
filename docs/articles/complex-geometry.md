# Optimization with complex geometry

Complex points use a real Hermitian metric, Re(trace(U\* V)). Objectives
must return real scalars: float64 for complex128 points, float32 for
complex64. Torch’s conjugate gradient convention is checked using
independent real and imaginary perturbations. Complex Euclidean, sphere
and phase geometries support automatic Hessians; complex frame and
unitary geometries currently qualify only first-order optimization.
Matrix factorizations at repeated singular values are not advertised as
automatically twice differentiable.

``` r

M <- manifold.complexcircle(4)
x <- riem.random(M,device="cpu")
P <- riem.problem(M,function(z) -z$real$sum())
riem.check.gradient(P,x)$checks
#>                  check        error    tolerance status note
#> 1 directional_gradient 7.088234e-13 0.0003162278   pass
riem.check.hessian(P,x)$checks
#>               check        error   tolerance status note
#> 1 hessian_linearity 1.057707e-17 0.001581139   pass     
#> 2  hessian_symmetry 2.325126e-17 0.001581139   pass     
#> 3    hessian_taylor 3.358129e-09 0.001581139   pass
fit <- riem.optimize(P,x,device="cpu")
fit$objective
#> [1] -4
riem.belongs(M,fit$point)
#> [1] TRUE
```

Complex Stiefel uses the embedded Euclidean metric. Complex Grassmann
uses orthonormal frames and objectives must be invariant to unitary
basis changes. Its logarithm is local and rejects the cut locus.

``` r

G <- manifold.grassmann(4,2,field="complex")
X <- riem.random(G,device="cpu")
U <- riem.tangent(G,X,torch_randn_like(X))*0.05
Y <- riem.exp(G,X,U)
(riem.log(G,X,Y)-U)$abs()$max()
#> torch_tensor
#> 4.129306918073458e-16
#> [ CPUDoubleType{} ]
riem.belongs(manifold.unitary(2),riem.random(manifold.unitary(2),device="cpu"))
#> [1] TRUE
```
