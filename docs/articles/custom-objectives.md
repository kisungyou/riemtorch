# Custom tensor objectives

The solver only sees a scalar torch objective and a manifold. A new loss
does not require a new estimator. Here the exact minimizer is a
normalized vector.

``` r

M <- manifold.sphere(3)
a <- torch_tensor(c(1, 2, 3), dtype = torch_float64())
x <- torch_tensor(c(1, 0, 0), dtype = torch_float64())
P <- riem.problem(M, function(x, data) -torch_sum(x * data), data = a)
fit <- riem.optimize(P, x, "conjugate_gradient")
fit
#> <riem_fit> conjugate_gradient on sphere 
#>  Objective: -3.7416574  | gradient norm: 0 
#>  Termination: converged_gradient  | iterations: 1
as.numeric(fit$point)
#> [1] 0.2672612 0.5345225 0.8017837
stopifnot(abs(fit$objective + sqrt(14)) < 1e-7)
```

An ambient gradient and an intrinsic gradient are different callbacks.
The sphere gradient removes the radial component; the package performs
that conversion for an ambient callback.

``` r

analytic <- riem.problem(M, function(x, data) -torch_sum(x * data),
                        egrad = function(x, data) -data, data = a)
riem.evaluate(analytic, x)$gradient
#> torch_tensor
#>  0
#> -2
#> -3
#> [ CPUDoubleType{3} ]
```

Objectives must return one scalar tensor with the point’s dtype/device.
Data can be converted explicitly at the boundary, then all numerical
work stays in torch. Deterministic callbacks must not change batches,
dropout masks or other state inside a line search. Returned solutions
are detached from the optimization graph.
