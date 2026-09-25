# Smooth Plus Nonsmooth Manifold Problems

Smooth Plus Nonsmooth Manifold Problems

## Usage

``` r
riem.problem.composite(
  manifold,
  fn = NULL,
  nonsmooth,
  prox,
  egrad = NULL,
  rgrad = NULL,
  label = NULL,
  data = NULL,
  required_operations = NULL,
  qualified = TRUE
)
```

## Arguments

- manifold:

  Geometry specification.

- fn:

  Function of one point returning one finite scalar torch tensor.

- nonsmooth:

  Function returning h(x), a real scalar tensor (positive infinity can
  encode infeasibility), or a list of such functions for cyclic proximal
  point. In that case \`fn\` must be NULL.

- prox:

  Function \`prox(q, step)\` solving exactly argmin_y
  h(y)+d(y,q)^2/(2\*step), or a matching list for cyclic proximal point.

- egrad:

  Optional ambient gradient callback \`egrad(x)\`.

- rgrad:

  Optional metric gradient callback \`rgrad(x)\`, mutually exclusive
  with \`egrad\`.

- label:

  Optional problem label.

- data:

  Optional registered data. When supplied, callbacks receive it as an
  additional named \`data\` argument. Registered tensor trees can be
  moved with the problem by the execution driver.

- required_operations:

  Device operations required by callbacks; NULL conservatively probes
  all. See \[riem.devices()\].

- qualified:

  Whether to require an initially qualified geometry. TRUE supports
  Euclidean, affine, log-Euclidean positive/SPD, hyperbolic and their
  products, powers and fixed metric scalings. FALSE permits experimental
  use on other geometries with exact exponential, logarithm and distance
  maps.

## Value

A composite \`riem_problem\` for \`"proximal_gradient"\` or
\`"cyclic_proximal_point"\`.

## Details

A proximal callback is a mathematical contract; ambient shrinkage
followed by projection generally does not satisfy it. Callbacks must use
the declared metric (including scale and product weights). The proximal
gradient solver uses an exponential gradient step and intrinsic prox,
with sufficient-decrease backtracking. Cyclic proximal point uses a
diminishing step sequence and reports a cycle residual, which alone is
not a stationarity certificate for a sum of nonsmooth functions.

## Examples

``` r
if(torch::torch_is_installed()) {
  P <- riem.problem.composite(manifold.euclidean(2),
    function(x) ((x-2)^2)$sum()/2, nonsmooth=function(x) x$abs()$sum(),
    prox=function(q,step) q$sign()*(q$abs()-step)$clamp(min=0))
  riem.optimize(P,torch::torch_zeros(2,dtype=torch::torch_float64()),
                "proximal_gradient",device="cpu")$point
}
#> torch_tensor
#>  1
#>  1
#> [ CPUDoubleType{2} ]
```
