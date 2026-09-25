# Move Tensor Trees and Device-Bound Geometry

Move a tensor tree, registered problem data, or built-in geometry to one
resolved device. Floating-point tensors use the resolved dtype; integer
and logical tensors retain their dtype.

## Usage

``` r
riem.to(x, device = NULL, dtype = NULL)
```

## Arguments

- x:

  A torch tensor, nested list, \`riem_manifold\`, or \`riem_problem\`.

- device, dtype:

  Device and floating-point dtype requests as described in
  \[riem.devices()\].

## Value

An object with the same structure and moved tensor values.

## Details

Generalized Stiefel and Grassmann geometries are rebuilt so that their
fixed SPD matrix and derived whitening operators share the target
placement. Product factors are converted recursively. Other built-in
geometries contain no device-bound state. Registered problem data and
least-squares weights are moved, but tensors captured privately inside a
callback remain caller-owned and must be registered as \`data\` instead.
Custom manifolds with tensors in their specification cannot be rebuilt
safely and produce an explicit error.

## Examples

``` r
if (torch::torch_is_installed()) {
  values <- list(x = torch::torch_ones(2, dtype = torch::torch_float64()),
                 index = torch::torch_tensor(1:2, dtype = torch::torch_int64()))
  moved <- riem.to(values, device = "cpu", dtype = "float32")
  moved$x$dtype
  moved$index$dtype
}
#> torch_Long
```
