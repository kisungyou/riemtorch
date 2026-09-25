# Save and Restore Training Parameters and Optimizer State

Save and Restore Training Parameters and Optimizer State

## Usage

``` r
riem.save(
  optimizer,
  path,
  model = NULL,
  training_state = NULL,
  include_rng = TRUE
)

riem.load(optimizer, path, device = NULL, model = NULL, restore_rng = FALSE)
```

## Arguments

- optimizer:

  An optimizer returned by \[optim_rsgd()\] or \[optim_radam()\].

- path:

  Checkpoint file path.

- model:

  Optional torch module whose parameters and buffers are included.

- training_state:

  Optional named list of progress and caller-owned state, such as epoch,
  global step, or data-order metadata.

- include_rng:

  Whether to save R, torch CPU and visible CUDA RNG states.

- device:

  Device on which to load tensor storage. \`NULL\` uses the existing
  optimizer parameter device.

- restore_rng:

  Whether to restore saved RNG states.

## Value

\`riem.save()\` invisibly returns the path. \`riem.load()\` invisibly
returns the optimizer after restoring its existing parameters and
states.

## Details

Checkpoints use torch-aware serialization and contain schema version,
named parameter mapping, parameter tensors, geometry/metric
configurations, factor structure, optimizer options and tensor states.
Version 2 optionally includes complete module state, training progress
and RNG state. Loading validates points, momentum and module state
before updating parameters. Create a matching module and optimizer on
the target device first. Parameter keys default to stable group/position
keys; explicit keys detect reordering.

Deterministic continuation assumes identical objectives and minibatch
order. Dataset position is caller-owned and can be placed in
\`training_state\`. Reproducible continuation requires matching runtimes
and data order; results across device types need not be bitwise
identical. Custom geometries must be reconstructed by the caller before
loading; callbacks are not deserialized.

## Examples

``` r
if (torch::torch_is_installed()) {
  x <- torch::nn_parameter(torch::torch_tensor(c(1, 0),
                          dtype = torch::torch_float64()))
  opt <- optim_rsgd(list(x), manifold = manifold.sphere(2))
  path <- tempfile(fileext = ".pt")
  riem.save(opt, path)
  riem.load(opt, path)
  unlink(path)
}
```
