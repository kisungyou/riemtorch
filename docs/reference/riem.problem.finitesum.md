# Finite-Sum Problems with Explicit Normalization

Finite-Sum Problems with Explicit Normalization

## Usage

``` r
riem.problem.finitesum(
  manifold,
  fn,
  n,
  normalization = c("mean", "sum"),
  sampling = c("without_replacement", "with_replacement"),
  batch_fn = NULL,
  data = NULL,
  evaluation_batch_size = 1024L,
  value_rgrad = NULL,
  preconditioner = NULL,
  required_operations = NULL
)
```

## Arguments

- manifold:

  Geometry specification.

- fn:

  Term callback \`fn(x, i)\` returning the scalar loss for index i.

- n:

  Number of terms.

- normalization:

  \`"mean"\` or \`"sum"\` for the full objective.

- sampling:

  \`"without_replacement"\` or \`"with_replacement"\` per minibatch.

- batch_fn:

  Optional vectorized callback \`batch_fn(x, indices)\` returning one
  loss per requested index as a vector tensor. When supplied, stochastic
  and chunked full evaluations use it instead of calling \`fn\`
  repeatedly.

- data:

  Optional registered data passed to callbacks as a named \`data\`
  argument.

- evaluation_batch_size:

  Positive chunk size used for full objective and gradient evaluations.
  Chunk gradients are accumulated after detaching their computation
  graphs, bounding graph memory independently of \`n\`.

- value_rgrad:

  Optional fused \`function(x, indices)\` returning \`value\` and metric
  \`gradient\` for the requested minibatch, with the same mean/sum
  normalization as \`fn\`. Registered data is passed when present.

- preconditioner:

  Optional tangent preconditioner; see \[riem.problem()\].

- required_operations:

  Device operations required by callbacks.

## Value

A \`riem_finitesum\` problem; deterministic solvers evaluate all terms.

## Details

A minibatch gradient is the average of its terms, multiplied by n for
sum normalization. Independent minibatches use torch's RNG. Initial,
requested periodic and final diagnostics evaluate the full objective and
gradient in chunks; intermediate minibatch norms are not stationarity
certificates. Indices passed to callbacks are one-based R integers.

## Examples

``` r
if (torch::torch_is_installed()) {
  P <- riem.problem.finitesum(manifold.euclidean(1),
    function(x, i) torch::torch_sum((x - i)^2)/2, n = 3)
  x <- torch::torch_zeros(1, dtype = torch::torch_float64())
  riem.optimize(P, x, "stochastic_gradient",
                control = list(max_iterations = 10, batch_size = 3))
}
#> <riem_fit> stochastic_gradient on euclidean 
#>  Objective: 0.33333333  | gradient norm: 0 
#>  Termination: converged_gradient  | iterations: 2 
```
