# Deterministic optimization and diagnostics

Orthogonal Procrustes is a nontrivial matrix objective with a known
solution. Neither the geometry nor the solver contains a special
Procrustes estimator.

``` r

M <- manifold.stiefel(5, 2, "euclidean")
A <- torch_randn(c(5, 2), dtype = torch_float64())
P <- riem.problem(M, function(X, data) -torch_sum(data * X), data = A)
fit <- riem.optimize(P, riem.random(M), "conjugate_gradient",
                     control = list(max_iterations = 150))
fit
#> <riem_fit> conjugate_gradient on stiefel 
#>  Objective: -2.9713442  | gradient norm: 1.656e-08 
#>  Termination: converged_gradient  | iterations: 42
fit$evaluations
#> $fn
#> [1] 88
#> 
#> $gradient
#> [1] 43
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
#> [1] 0
stopifnot(riem.belongs(M, fit$point))
reference <- riem.project(M, riem.to(A, device = fit$execution$device,
                                     dtype = fit$execution$dtype))
torch_sum((fit$point - reference)^2)$item()
#> [1] 6.777258e-17
```

Armijo backtracking rejects infeasible or nonfinite trials. Strong Wolfe
is available only where the true retraction-curve derivative is
supported. PR+ conjugate gradient restarts on loss of descent. L-BFGS
transports secant history and skips invalid curvature pairs; BB bounds
its spectral step lengths.

`converged_gradient` is distinct from `stopped_small_step`,
`max_iterations`, `line_search_failed`, `domain_error`,
`nonfinite_objective`, and `numerical_failure`. Always inspect
termination and the available gradient norm. A feasible iterate is
useful even when the maximum iteration budget was reached.

``` r

head(riem.support("solvers"))
#>               method                                                 variant
#> 1   steepest_descent                               Armijo retraction descent
#> 2 conjugate_gradient             transported PR+/FR/HS+/DY; descent restarts
#> 3   barzilai_borwein             BB1/BB2/alternating with safeguarded Armijo
#> 4              lbfgs transported limited-memory secants; curvature rejection
#> 5      trust_regions              Steihaug truncated CG and acceptance ratio
#> 6          newton_cg truncated CG; negative-curvature safeguard; line search
#>           requirements   status
#> 1 gradient; retraction verified
#> 2  gradient; transport verified
#> 3  gradient; transport verified
#> 4  gradient; transport verified
#> 5 exact Riemannian HVP verified
#> 6 exact Riemannian HVP verified
#>                                                reference
#> 1 inst/math/solvers.md; development/competitors/audit.md
#> 2 inst/math/solvers.md; development/competitors/audit.md
#> 3 inst/math/solvers.md; development/competitors/audit.md
#> 4 inst/math/solvers.md; development/competitors/audit.md
#> 5 inst/math/solvers.md; development/competitors/audit.md
#> 6 inst/math/solvers.md; development/competitors/audit.md
#>                                    evidence
#> 1 test-smooth-upgrade; test-solver-evidence
#> 2 test-smooth-upgrade; test-solver-evidence
#> 3 test-smooth-upgrade; test-solver-evidence
#> 4 test-smooth-upgrade; test-solver-evidence
#> 5 test-smooth-upgrade; test-solver-evidence
#> 6 test-smooth-upgrade; test-solver-evidence
fit$diagnostics$restarts
#> [1] 35
```

The four derivative-free searches and cubic regularization are
implemented but marked experimental. They have explicit algorithm
variants in the installed mathematical notes. Heuristics do not claim
gradient convergence.
