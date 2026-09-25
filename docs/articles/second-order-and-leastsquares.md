# Hessians and least-squares problems

Exact Hessian-vector products include the metric connection correction.
A sphere quadratic illustrates why projecting an ambient Hessian is
insufficient: a constant-on-sphere objective has zero intrinsic Hessian.

``` r

M <- manifold.sphere(3)
P <- riem.problem(M, function(x) torch_sum(x^2)/2)
x <- torch_tensor(c(1, 0, 0), dtype = torch_float64())
u <- torch_tensor(c(0, 1, 0), dtype = torch_float64())
riem.hessian(P, x, u)
#> torch_tensor
#>  0
#>  0
#>  0
#> [ CPUDoubleType{3} ]
```

Trust regions and Newton-CG use matrix-free exact HVPs. For unsupported
automatic geometry/objective combinations, provide `rhess(x,u)`
explicitly. Custom matrix logarithm/root operations support first
derivatives only and are rejected in automatic higher-order paths.

``` r

E <- manifold.euclidean(2)
P <- riem.problem(E, function(x) torch_sum((x - 1)^2)/2)
fit <- riem.optimize(P, torch_zeros(2, dtype = torch_float64()), "trust_regions")
fit
#> <riem_fit> trust_regions on euclidean 
#>  Objective: 0  | gradient norm: 0 
#>  Termination: converged_gradient  | iterations: 2
fit$diagnostics$subproblems
#> [[1]]
#> [[1]]$status
#> [1] "boundary"
#> 
#> [[1]]$iterations
#> [1] 1
#> 
#> [[1]]$predicted_reduction
#> [1] 0.9142136
#> 
#> [[1]]$ratio
#> [1] 1
#> 
#> 
#> [[2]]
#> [[2]]$status
#> [1] "residual_tolerance"
#> 
#> [[2]]$iterations
#> [1] 1
#> 
#> [[2]]$predicted_reduction
#> [1] 0.08578644
#> 
#> [[2]]$ratio
#> [1] 1
```

Least-squares problems use the selected metric adjoint J*, not an
unqualified coordinate transpose. The residual weights are fixed
positive semidefinite. Gauss–Newton and LM advertise the approximation
J* W J; LM adds lambda I in the tangent metric and adapts damping by
actual/predicted reduction.

``` r

P <- riem.problem.leastsquares(E, function(x) x - 1,
                              weights = torch_tensor(c(1, 3), dtype = torch_float64()))
fit <- riem.optimize(P, torch_zeros(2, dtype = torch_float64()), "levenberg_marquardt")
fit
#> <riem_fit> levenberg_marquardt on euclidean 
#>  Objective: 1.1764815e-20  | gradient norm: 1.535e-10 
#>  Termination: converged_gradient  | iterations: 4
stopifnot(fit$objective < 1e-10)
```
