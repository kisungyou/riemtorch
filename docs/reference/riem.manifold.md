# Define a Manifold Through Public Geometry Contracts

Create an extensible geometry specification. Kernels operate on one
point; the public operations apply them separately to leading batch
dimensions.

## Usage

``` r
riem.manifold(
  name,
  shape,
  dimension,
  metric,
  operations,
  capabilities = list(),
  specification = list(),
  batch_operations = list(),
  primitive_derivatives = integer(),
  required_operations = NULL,
  second_order_retraction = FALSE
)
```

## Arguments

- name:

  Descriptive geometry name.

- shape:

  Positive integer vector of point dimensions.

- dimension:

  Intrinsic dimension (a nonnegative integer).

- metric:

  Metric identifier.

- operations:

  Named list of functions. Required: \`belongs(x, tol)\`, \`tangent(x,
  u)\`, \`inner(x, u, v)\`, \`egrad2rgrad(x, u)\`, \`retr(x, u)\`.
  Optional: \`transport(x, u, y, v)\`, \`project(x)\`, \`random(x)\`,
  \`residual(x)\`, \`exp(x, u)\`, \`log(x, y)\`, \`sqdist(x, y)\`,
  \`ehess2rhess(x, u, egrad, ehess)\`. Here \`u\` in transport is the
  actual retraction step, and \`x\` in random is a zero tensor with
  requested placement.

- capabilities:

  Named list describing optional derivative capabilities. \`hessian\`
  declares exact ambient conversion, \`curve_derivative\` declares first
  derivatives of the retraction, and \`isometric_transport\` declares
  metric preservation. \`snapshot_transport\` declares that transport
  accepts an endpoint pair identified by a local logarithm, even when it
  is not a retraction inverse. Custom explicit transports default to
  FALSE; the default endpoint tangent projection is safe. Other defaults
  are conservative.

- specification:

  Named serializable list of geometry configuration.

- batch_operations:

  Optional named list of kernels that operate directly on leading batch
  axes. Missing kernels use the pointwise fallback. Built-in geometries
  provide native batch kernels for their common operations.

- primitive_derivatives:

  Named integer orders (0, 1 or 2) verified for each primitive. Omitted
  entries conservatively declare no derivative support.

- required_operations:

  Optional device probe operations; NULL uses all.

- second_order_retraction:

  Whether the retraction has verified order two.

## Value

An S3 \`riem_manifold\` object.

## Examples

``` r
if (torch::torch_is_installed()) {
  M <- riem.manifold("line", 1, 1, "euclidean", list(
    belongs = function(x, tol) TRUE,
    tangent = function(x, u) u,
    inner = function(x, u, v) torch::torch_sum(u * v),
    egrad2rgrad = function(x, u) u,
    retr = function(x, u) x + u))
  x <- torch::torch_tensor(2, dtype = torch::torch_float64())
  riem.retr(M, x, -x)
}
#> torch_tensor
#>  0
#> [ CPUDoubleType{1} ]
```
