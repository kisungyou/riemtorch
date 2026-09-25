# Constraints and intrinsic nonsmooth optimization

Equality constraints use h(x)=0; inequalities use g(x)\<=0. The
augmented Lagrangian reuses smooth inner solvers, tightens their
requested accuracy and updates multipliers. Its result reports primal
feasibility, Lagrangian gradient norm, complementarity, dual
feasibility, and every inner-solver status.

``` r

P <- riem.problem.constrained(manifold.euclidean(2),
  function(x) ((x-2)^2)$sum()/2,
  equality=function(x) x$sum()$reshape(1)-1,
  inequality=function(x) -x)
fit <- riem.optimize(P,torch_tensor(c(0.5,0.5),dtype=torch_float64()),
  "augmented_lagrangian",device="cpu")
fit$point
#> torch_tensor
#>  0.5000
#>  0.5000
#> [ CPUDoubleType{2} ]
fit$multipliers
#> $equality
#> torch_tensor
#>  1.5000
#> [ CPUDoubleType{1} ]
#> 
#> $inequality
#> torch_tensor
#>  0
#>  0
#> [ CPUDoubleType{2} ]
fit$kkt
#> $feasibility
#> [1] 1.386557e-07
#> 
#> $stationarity
#> [1] 4.710277e-15
#> 
#> $complementarity
#> [1] 0
#> 
#> $dual_feasibility
#> [1] 0
```

Automatic constraint adjoints use torch differentiation. Explicit metric
adjoints may be supplied as `equality_adjoint(x, weights)` and
`inequality_adjoint(x, weights)`. Registered data is passed consistently
to all callbacks. Inner failures, work budgets and user stops do not
count as KKT convergence. Even successful KKT tests are local
conditions, not global optimality.

For a smooth function f and nonsmooth h, the intrinsic proximal callback
must solve

``` math
\mathrm{prox}_{t h}(q)=\arg\min_{y\in M}\left(h(y)+\frac{d(y,q)^2}{2t}\right).
```

Projection after ambient shrinkage generally solves a different problem.
The unaccelerated method first takes an exponential gradient step,
applies that proximal map, then checks sufficient decrease. The norm of
the intrinsic gradient mapping is its residual; a small ambient gradient
is not used.

``` r

soft <- function(x,t) x$sign()*(x$abs()-t)$clamp(min=0)
P <- riem.problem.composite(manifold.positive(2),
  function(x) ((x$log()-2)^2)$sum()/2,
  nonsmooth=function(x) x$log()$abs()$sum(),
  prox=function(q,step) soft(q$log(),step)$exp())
fit <- riem.optimize(P,torch_ones(2,dtype=torch_float64()),device="cpu")
fit$point$log() # solution is c(1,1) in log coordinates
#> torch_tensor
#>  1
#>  1
#> [ CPUDoubleType{2} ]
fit$proximal_residual
#> [1] 0
```

Euclidean, affine, log-Euclidean positive/SPD, and hyperbolic geometries
and their products/scalings are initially qualified. Qualification
concerns the geometry contract, not arbitrary user proximal callbacks.
Other geometries require `qualified=FALSE` and remain experimental. The
caller must validate its prox against optimality conditions for its
particular h.

Cyclic proximal point takes lists of value functions and matching
proximal maps, without a separate smooth term. Default steps decay with
exponent 0.75. Its cycle displacement divided by the step is reported,
but a small cycle residual alone is not a stationarity certificate for
the sum.

``` r

C <- riem.problem.composite(manifold.euclidean(1),
  nonsmooth=list(function(x) x$square()$sum()/2,
                 function(x) ((x-2)^2)$sum()/2),
  prox=list(function(q,t) q/(1+t),function(q,t) (q+2*t)/(1+t)))
fit <- riem.optimize(C,torch_tensor(3,dtype=torch_float64()),
  "cyclic_proximal_point",control=list(max_iterations=50),device="cpu")
fit$point
#> torch_tensor
#>  1.0303
#> [ CPUDoubleType{1} ]
fit$termination
#> [1] "max_iterations"
```
