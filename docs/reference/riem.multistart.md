# Run Reproducible Multiple Starts on One Device

Run Reproducible Multiple Starts on One Device

## Usage

``` r
riem.multistart(
  problem,
  initials = NULL,
  starts = 5L,
  seed = 1L,
  method = "steepest_descent",
  control = list(),
  device = NULL,
  dtype = NULL
)
```

## Arguments

- problem:

  An optimization problem.

- initials:

  Optional list of feasible initial points.

- starts:

  Number of random starts when initials is NULL.

- seed:

  Integer initialization seed. NULL uses the current RNG stream.

- method, control:

  Solver and controls passed to \[riem.optimize()\].

- device, dtype:

  Execution requests resolved once for all starts.

## Value

A list of fits, the index of the best finite feasible fit, its fit, and
execution metadata. Failed fits remain in the result.

## Examples

``` r
if (torch::torch_is_installed()) {
  P <- riem.problem(manifold.sphere(3),function(x) -x[1])
  riem.multistart(P,starts=2,device="cpu")$best$objective
}
#> [1] -1
```
