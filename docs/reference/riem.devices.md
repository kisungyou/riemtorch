# Discover and Select Torch Execution Devices

\`riem.devices()\` reports the devices visible to the current R process
and checks whether they support riemtorch's core linear-algebra
operations at the requested precision. \`riem.device()\` applies the
package selection policy and returns one reusable torch device.

## Usage

``` r
riem.devices(dtype = NULL, operations = NULL)

riem.device(device = NULL, dtype = NULL, reference = NULL, operations = NULL)
```

## Arguments

- dtype:

  Optional floating-point torch dtype, or one of \`"float64"\`,
  \`"float32"\`, \`"float16"\`, \`"bfloat16"\`, \`"complex64"\`, or
  \`"complex128"\`. Discovery defaults to float64.

- operations:

  Required device operations: a subset of \`basic\`, \`cholesky\`,
  \`eigh\`, \`svd\`, \`qr\`, \`solve\`, and \`matrix_exp\`. NULL probes
  all.

- device:

  A device request. \`NULL\` uses, in order,
  \`getOption("riemtorch.device")\`, \`RIEMTORCH_DEVICE\`, and
  \`"auto"\`. Explicit \`"auto"\` ignores those defaults. Use \`"cpu"\`,
  \`"mps"\`, \`"cuda"\`, or a zero-based CUDA index such as \`"cuda:1"\`
  to override.

- reference:

  Optional tensor or tensor tree whose floating-point dtype is preserved
  when \`dtype\` is omitted.

## Value

\`riem.devices()\` returns a data frame. \`riem.device()\` returns a
\`torch_device\`.

## Details

Automatic selection tries the current visible CUDA device, the remaining
visible CUDA devices in index order, compatible MPS, and CPU. A small
computation probe checks tensor creation, autograd, decompositions, and
the matrix exponential. An automatic request skips incompatible
accelerators; an explicit unavailable or incompatible request is an
error. No package-load device initialization or runtime download is
performed.

## Examples

``` r
if (torch::torch_is_installed()) {
  riem.devices()
  riem.device("cpu")
}
#> torch_device(type='cpu') 
```
