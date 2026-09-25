# Least-Squares Objectives and Metric Adjoints

Least-Squares Objectives and Metric Adjoints

## Usage

``` r
riem.problem.leastsquares(
  manifold,
  residual,
  weights = NULL,
  jvp = NULL,
  adjoint = NULL,
  rhess = NULL,
  data = NULL,
  loss = "linear",
  loss_scale = 1,
  preconditioner = NULL,
  required_operations = NULL,
  value_rgrad = NULL
)
```

## Arguments

- manifold:

  Geometry specification.

- residual:

  Function \`residual(x)\` returning a one-dimensional tensor.

- weights:

  Optional nonnegative vector of diagonal residual weights or symmetric
  positive-semidefinite weight matrix, on the point's device/dtype.

- jvp:

  Optional residual differential callback \`jvp(x, u)\`.

- adjoint:

  Optional metric adjoint callback \`adjoint(x, v)\`. Both differential
  callbacks must be supplied together, or both omitted.

- rhess:

  Optional exact Hessian callback for the full objective.

- data:

  Optional registered data passed to callbacks as a named \`data\`
  argument.

- loss:

  Robust residual loss: linear, huber, soft_l1 or cauchy.

- loss_scale:

  Positive residual scale.

- preconditioner:

  Optional tangent preconditioner; see \[riem.problem()\].

- required_operations:

  Device operations required by callbacks.

- value_rgrad:

  Optional fused full-objective callback; see \[riem.problem()\].

## Value

A \`riem_leastsquares\` problem. Linear loss has objective 0.5 r' W r.
Robust losses apply scale squared times rho((r/scale)^2) componentwise
before diagonal weighting. The Gauss–Newton model uses nonnegative rho'
weights and omits rho” terms.

## Details

Without callbacks, reverse-mode differentiation computes the metric
adjoint, and reverse-over-reverse differentiation computes J u. This
needs differentiable backward rules for residual operations. GN and LM
use J\* W J, which is explicitly an approximation to the objective's
exact Hessian.

## Examples

``` r
if (torch::torch_is_installed()) {
  P <- riem.problem.leastsquares(manifold.euclidean(2), function(x) x - 1)
  x <- torch::torch_zeros(2, dtype = torch::torch_float64())
  riem.optimize(P, x, "levenberg_marquardt")
}
#> <riem_fit> levenberg_marquardt on euclidean 
#>  Objective: 2.3518606e-20  | gradient norm: 2.169e-10 
#>  Termination: converged_gradient  | iterations: 4 
```
