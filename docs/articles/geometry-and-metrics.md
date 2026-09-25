# Geometry, metrics and representations

Constructors specify a space, metric and representation. Multi-metric
families require an explicit choice. SPD tangents are symmetric
matrices; choosing AIRM changes gradient conversion, not just the
reported distance.

``` r

X <- torch_diag(torch_tensor(c(2, 3), dtype = torch_float64()))
G <- torch_eye(2, dtype = X$dtype)
M <- manifold.spd(2, "airm")
riem.egrad2rgrad(M, X, G)       # X G X
#> torch_tensor
#>  4  0
#>  0  9
#> [ CPUDoubleType{2,2} ]
riem.inner(M, X, G, G)
#> torch_tensor
#> 0.3611111111111111
#> [ CPUDoubleType{} ]
riem.capabilities(M)
#>      operation value first_derivative second_derivative native_batch
#> 1      belongs  TRUE            FALSE             FALSE        FALSE
#> 2      tangent  TRUE             TRUE              TRUE         TRUE
#> 3        inner  TRUE             TRUE              TRUE         TRUE
#> 4  egrad2rgrad  TRUE             TRUE              TRUE         TRUE
#> 5         retr  TRUE             TRUE              TRUE         TRUE
#> 6    transport  TRUE             TRUE             FALSE         TRUE
#> 7  ehess2rhess  TRUE            FALSE             FALSE         TRUE
#> 8          exp  TRUE             TRUE             FALSE         TRUE
#> 9          log  TRUE             TRUE             FALSE         TRUE
#> 10      sqdist  TRUE             TRUE             FALSE         TRUE
#> 11     project  TRUE            FALSE             FALSE         TRUE
#> 12      random  TRUE            FALSE             FALSE         TRUE
```

Leading batch axes are independent geometry evaluations. They are never
inferred to be factors of a joint optimization problem. Every input must
have identical batch axes; implicit broadcasting is deliberately
rejected.

``` r

S <- manifold.sphere(3)
batch <- riem.random(S, batch_shape = c(2, 4))
u <- riem.tangent(S, batch, torch_randn_like(batch))
riem.inner(S, batch, u, u)$shape
#> [1] 2 4
stopifnot(all(riem.belongs(S, riem.retr(S, batch, u, step = 0.1))))
```

Grassmann frame objectives must be invariant under changes of
orthonormal basis. PSD factor objectives must depend on Y Y’. Kendall
shape objectives must be invariant under rotations. Low-rank and shape
constructors document their regular strata. Repair through
[`riem.project()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
is explicit; a solver never repairs a failed trial invisibly. The
installed geometry ledger maps all source rows:

``` r

head(riem.support("geometry"), 8)
#>    id    source_geometry          constructor                       metric
#> 1 G01          Euclidean   manifold.euclidean                    euclidean
#> 2 G02             Sphere      manifold.sphere                        round
#> 3 G03    SphereExtrinsic      manifold.sphere                        round
#> 4 G04            Oblique     manifold.oblique                 column_round
#> 5 G05 ProbabilitySimplex manifold.multinomial                   fisher_rao
#> 6 G06       PoincareBall  manifold.hyperbolic poincare; negative curvature
#> 7 G07        Hyperboloid  manifold.hyperbolic  lorentz; negative curvature
#> 8 G08              Torus       manifold.torus                  flat_angles
#>                     representation   status
#> 1                           tensor verified
#> 2                      unit vector verified
#> 3 identity embedding; alias of G02 verified
#> 4               p x k unit columns verified
#> 5      positive probability vector verified
#> 6             d-vector inside ball verified
#> 7         d+1 vector positive time verified
#> 8              d angles modulo 2pi verified
#>                                                     evidence
#> 1   test-geometry-contracts; test-solver-evidence: euclidean
#> 2      test-geometry-contracts; test-solver-evidence: sphere
#> 3      test-geometry-contracts; test-solver-evidence: sphere
#> 4     test-geometry-contracts; test-solver-evidence: oblique
#> 5 test-geometry-contracts; test-solver-evidence: multinomial
#> 6    test-geometry-contracts; test-solver-evidence: poincare
#> 7 test-geometry-contracts; test-solver-evidence: hyperboloid
#> 8       test-geometry-contracts; test-solver-evidence: torus
#>                                            space_equivalence
#> 1 distinct metric/space family or parameterized construction
#> 2 distinct metric/space family or parameterized construction
#> 3                                                        G02
#> 4 distinct metric/space family or parameterized construction
#> 5 distinct metric/space family or parameterized construction
#> 6 distinct metric/space family or parameterized construction
#> 7                                                        G06
#> 8 distinct metric/space family or parameterized construction
```
