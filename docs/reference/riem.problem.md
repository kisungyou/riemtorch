# Define a Scalar Manifold Optimization Problem

Define a Scalar Manifold Optimization Problem

## Usage

``` r
riem.problem(
  manifold,
  fn,
  egrad = NULL,
  rgrad = NULL,
  rhess = NULL,
  label = NULL,
  data = NULL,
  required_operations = NULL,
  value_rgrad = NULL,
  preconditioner = NULL
)
```

## Arguments

- manifold:

  Geometry specification.

- fn:

  Function of one point returning one finite scalar torch tensor.

- egrad:

  Optional ambient gradient callback \`egrad(x)\`.

- rgrad:

  Optional metric gradient callback \`rgrad(x)\`, mutually exclusive
  with \`egrad\`.

- rhess:

  Optional exact Riemannian Hessian callback \`rhess(x, u)\`.

- label:

  Optional problem label.

- data:

  Optional registered data. When supplied, callbacks receive it as an
  additional named \`data\` argument. Registered tensor trees can be
  moved with the problem by the execution driver.

- required_operations:

  Device operations required by callbacks; NULL conservatively probes
  all. See \[riem.devices()\].

- value_rgrad:

  Optional callback returning \`list(value, gradient)\` with a scalar
  value and a Riemannian gradient, evaluated together.

- preconditioner:

  Optional callback \`preconditioner(x, u)\` applying a positive
  definite self-adjoint tangent-space preconditioner.

## Value

A \`riem_problem\` holding callbacks and geometry; construction does not
evaluate callbacks or initialize devices.

## Details

Deterministic objectives must remain fixed during a solve. Freeze
batches, dropout and other random state before line searches. The
ordinary solver result is detached; differentiation through an entire
solve is not supported. Analytic derivatives must match the objective
and selected metric.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.sphere(3)
  a <- torch::torch_tensor(c(1, 2, 3), dtype = torch::torch_float64())
  problem <- riem.problem(M,
    function(x, data) -torch::torch_sum(x * data), data = a)
  fit <- riem.optimize(problem, riem.random(M))
  fit$objective
}
#> [1] -3.741657
```
