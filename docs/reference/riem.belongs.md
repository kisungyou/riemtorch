# Inspect and Operate on Manifold Points

Metric-aware tensor operations. No operation silently repairs input
points. \`riem.project()\` is the explicit repair boundary. Tangents use
the same point axes and representation as points. Batch axes must match
exactly.

## Usage

``` r
riem.belongs(manifold, x, tol = NULL)

riem.tangent(manifold, x, u)

riem.istangent(manifold, x, u, tol = NULL)

riem.inner(manifold, x, u, v)

riem.norm(manifold, x, u)

riem.egrad2rgrad(manifold, x, u)

riem.retr(manifold, x, u, step = 1)

riem.transport(manifold, x, u, y, v, step = 1)

riem.project(manifold, x)

riem.exp(manifold, x, u, step = 1)

riem.log(manifold, x, y)

riem.sqdist(manifold, x, y)

riem.dist(manifold, x, y)

riem.ehess2rhess(manifold, x, u, egrad, ehess)
```

## Arguments

- manifold:

  A geometry specification.

- x, y:

  Points (tensors or named product lists).

- tol:

  Nonnegative membership/tangency tolerance. \`NULL\` chooses a
  precision-aware default: 1e-7 for float64 and 1e-5 for float32.

- u, v:

  Tangent vectors; \`u\` is an ambient vector for projection or gradient
  conversion.

- step:

  Scalar multiplier of a retraction/transport step.

- egrad, ehess:

  Ambient gradient and ambient Hessian-vector product.

## Value

\`belongs\` and \`istangent\` return logical values, one per batch
point. \`inner\`, \`norm\`, \`dist\` and \`sqdist\` return tensors with
point axes reduced. Other operations return tensors or named product
lists of tensors.

## Details

Transport receives the path \`(x, step \* u, y)\` and vector \`v\`.
Exact exponential, logarithm, distance and Hessian conversion are
optional; unsupported operations fail explicitly. Sphere logarithms
reject antipodes. Distance itself is not differentiable on the diagonal;
use squared distance when an objective needs a derivative there.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- manifold.sphere(3)
  x <- torch::torch_tensor(c(1, 0, 0), dtype = torch::torch_float64())
  u <- torch::torch_tensor(c(0, 0.2, 0), dtype = x$dtype)
  y <- riem.retr(M, x, u)
  riem.belongs(M, y)
  riem.inner(M, x, u, u)
  riem.transport(M, x, u, y, u)
  riem.sqdist(M, x, y)
}
#> torch_tensor
#> 0.01 *
#>  3.8965
#> [ CPUDoubleType{1} ]
```
