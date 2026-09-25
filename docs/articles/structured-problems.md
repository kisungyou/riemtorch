# Products, blocks and finite sums

Named products express joint parameters. Fixed metric weights affect
gradient conversion by reciprocal scaling, and are not loss weights.

``` r

M <- manifold.product(list(position = manifold.euclidean(2),
                          direction = manifold.sphere(3)), weights = c(2, 5))
x <- list(position = torch_tensor(c(2, -1), dtype = torch_float64()),
          direction = torch_tensor(c(1, 0, 0), dtype = torch_float64()))
P <- riem.problem(M, function(z) torch_sum(z$position^2)/2 - z$direction[2])
fit <- riem.optimize(P, x, "alternating_gradient",
                     control = list(max_iterations = 100))
fit
#> <riem_fit> alternating_gradient on product 
#>  Objective: -1  | gradient norm: 1.414e-05 
#>  Termination: max_iterations  | iterations: 100
stopifnot(riem.belongs(M, fit$point))
```

A finite-sum term callback identifies each observation by an index. The
loss normalization is explicit, including minibatch rescaling for full
sums.

``` r

observations <- torch_tensor(1:5, dtype = torch_float64())
P <- riem.problem.finitesum(manifold.euclidean(1),
  function(x, i, data) torch_sum((x - data[i])^2)/2,
  batch_fn = function(x, indices, data) {
    (x - data[indices])$square()$reshape(c(length(indices))) / 2
  },
  data = observations, n = 5, normalization = "mean",
  evaluation_batch_size = 2)
x <- torch_zeros(1, dtype = torch_float64())
fit <- riem.optimize(P, x, "stochastic_gradient", control = list(
  max_iterations = 30, batch_size = 5, step_size = 0.2))
fit
#> <riem_fit> stochastic_gradient on euclidean 
#>  Objective: 1.0000069  | gradient norm: 0.003714 
#>  Termination: max_iterations  | iterations: 30
fit$diagnostics$sampling
#> $normalization
#> [1] "mean"
#> 
#> $policy
#> [1] "without_replacement"
#> 
#> $batch_size
#> [1] 5
#> 
#> $evaluation_batch_size
#> [1] 2
#> 
#> $full_evaluation_every
#> [1] 0
#> 
#> $stationarity
#> [1] "full_gradient"
#> 
#> $intermediate
#> [1] "minibatch"
```

The stochastic solver evaluates minibatches after each update. Full
objective and gradient evaluations are chunked and occur initially and
finally. Set `full_evaluation_every` to a positive interval for periodic
stationarity checks; minibatch gradient norms alone are never reported
as convergence certificates. For a caller-controlled dataloader, use
[`riem.train()`](https://www.kisungyou.com/riemtorch/reference/riem.train.md)
with a native training optimizer.
