# Smooth Equality and Inequality Constrained Problems

Smooth Equality and Inequality Constrained Problems

## Usage

``` r
riem.problem.constrained(
  manifold,
  fn,
  equality = NULL,
  inequality = NULL,
  egrad = NULL,
  rgrad = NULL,
  equality_adjoint = NULL,
  inequality_adjoint = NULL,
  label = NULL,
  data = NULL,
  required_operations = NULL,
  preconditioner = NULL
)
```

## Arguments

- manifold:

  Geometry specification.

- fn:

  Function of one point returning one finite scalar torch tensor.

- equality, inequality:

  Optional functions returning real vector tensors. Equalities are
  h(x)=0; inequalities are g(x)\<=0. At least one is required.

- egrad:

  Optional ambient gradient callback \`egrad(x)\`.

- rgrad:

  Optional metric gradient callback \`rgrad(x)\`, mutually exclusive
  with \`egrad\`.

- equality_adjoint, inequality_adjoint:

  Optional metric adjoints \`function(x, weights)\` returning the
  Riemannian gradient of the weighted constraint sum. Omitted adjoints
  use automatic differentiation.

- label:

  Optional problem label.

- data:

  Optional registered data. When supplied, callbacks receive it as an
  additional named \`data\` argument. Registered tensor trees can be
  moved with the problem by the execution driver.

- required_operations:

  Device operations required by callbacks; NULL conservatively probes
  all. See \[riem.devices()\].

- preconditioner:

  Optional callback \`preconditioner(x, u)\` applying a positive
  definite self-adjoint tangent-space preconditioner.

## Value

A constrained \`riem_problem\` for \`method="augmented_lagrangian"\`.

## Details

Registered data is passed to every callback. The solver uses the
Powell–Hestenes–Rockafellar inequality penalty and smooth first-order
inner solvers. Feasibility, Lagrangian stationarity, complementarity,
nonnegative inequality multipliers and inner-solver status are reported.
A KKT stopping test does not certify a global optimum.

## Examples

``` r
if(torch::torch_is_installed()) {
  P <- riem.problem.constrained(manifold.euclidean(2),
    function(x) ((x-2)^2)$sum()/2, equality=function(x) x$sum()$reshape(1)-1)
  fit <- riem.optimize(P,torch::torch_zeros(2,dtype=torch::torch_float64()),
                       "augmented_lagrangian",device="cpu")
  fit$kkt
}
#> $feasibility
#> [1] 7.345558e-07
#> 
#> $stationarity
#> [1] 9.420555e-16
#> 
#> $complementarity
#> [1] 0
#> 
#> $dual_feasibility
#> [1] 0
#> 
```
