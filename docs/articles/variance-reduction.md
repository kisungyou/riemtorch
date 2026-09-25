# Variance reduction and solver controls

A finite sum separates expensive full gradients from inexpensive
stochastic updates. SVRG uses a snapshot gradient; SRG transports a
recursive estimator. Only full gradients establish stationarity.

``` r

P <- riem.problem.finitesum(manifold.euclidean(1),
  function(x,i) ((x-i)^2)$sum()/2, n=5)
x <- torch_tensor(0,dtype=torch_float64())
fits <- lapply(c("svrg","srg"),function(method)
  riem.optimize(P,x,method,control=list(step_size=0.2,max_iterations=100),device="cpu"))
vapply(fits,function(f) as.numeric(f$point),numeric(1))
#> [1] 3 3
fits[[1]]$evaluations
#> $fn
#> [1] 200
#> 
#> $gradient
#> [1] 200
#> 
#> $hessian
#> [1] 0
#> 
#> $residual
#> [1] 0
#> 
#> $adjoint
#> [1] 0
#> 
#> $jvp
#> [1] 0
#> 
#> $terms
#> [1] 208
```

SVRG requires a local logarithm for its snapshot-to-current transport
path. It rejects unsupported geometries before evaluating the objective.
Work budgets and callbacks distinguish a requested stop from
convergence.

``` r

fit <- riem.optimize(P,x,control=list(callback=function(state) state$iteration>=1),device="cpu")
fit$termination
#> [1] "converged_gradient"
L <- riem.problem.leastsquares(manifold.euclidean(1),
  function(x) torch_cat(list(x-1,x-2,x-100)),loss="huber")
riem.optimize(L,x,"levenberg_marquardt",device="cpu")$point
#> torch_tensor
#>  2.0000
#> [ CPUDoubleType{1} ]
```
