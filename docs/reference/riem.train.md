# Train a Torch Module on One Resolved Device

Train a Torch Module on One Resolved Device

## Usage

``` r
riem.train(
  model,
  dataloader,
  loss,
  optimizer_factory,
  epochs = 1L,
  device = NULL,
  dtype = NULL,
  checkpoint = NULL,
  restore_rng = FALSE,
  ...,
  validation = NULL,
  callback = NULL,
  clip_norm = NULL,
  accumulate = 1L,
  scheduler = NULL,
  data_state = NULL
)
```

## Arguments

- model:

  A torch \`nn_module\` instance.

- dataloader:

  A torch dataloader.

- loss:

  Function \`loss(model, batch, ...)\` returning one finite scalar torch
  tensor.

- optimizer_factory:

  Function called as \`optimizer_factory(model)\` after the model has
  been moved to the selected device and dtype. It must return a torch
  optimizer, such as \[optim_rsgd()\] or \[optim_radam()\].

- epochs:

  Positive number of complete dataloader passes.

- device:

  Device request. \`NULL\` follows the package selection policy; use
  \`"cpu"\` to force CPU or, for example, \`"cuda:1"\` for a specific
  GPU.

- dtype:

  Optional floating-point dtype request. Existing model precision is
  preserved when it is omitted.

- checkpoint:

  Optional checkpoint path to restore after model placement and
  optimizer construction. This permits, for example, resuming a GPU
  checkpoint with \`device = "cpu"\`.

- restore_rng:

  Whether to restore RNG state from \`checkpoint\`.

- ...:

  Additional arguments passed to \`loss\`.

- validation:

  Optional \`function(model, epoch)\` returning finite numeric
  validation metrics; evaluated in evaluation mode without gradient
  recording.

- callback:

  Optional \`function(state)\` called after each epoch; return TRUE to
  stop. The state contains detached history, metrics and progress.

- clip_norm:

  Optional positive bound on the total Riemannian gradient norm.

- accumulate:

  Number of minibatches averaged into each optimizer update.

- scheduler:

  Optional \`function(optimizer, epoch, metrics)\` called after
  validation. Caller-owned scheduler state can be included in
  \`data_state\`.

- data_state:

  Optional list with \`get()\` and \`set(state)\` callbacks for
  epoch-boundary data-order or scheduler state. Restoration happens
  before the next iterator is created. Pass the returned
  \`training_state\` to \[riem.save()\] to resume progress together with
  model, optimizer and RNG state.

## Value

A \`riem_training_fit\` containing \`model\`, \`optimizer\`, per-epoch
\`history\`, total \`steps\`, and resolved \`execution\` metadata.

## Details

Device resolution and model placement happen once, before optimizer
construction. Each minibatch is moved immediately before use. Integer
index tensors and logical masks retain their dtype. Tensor arguments
supplied in \`...\` are moved once with the model. The mean reported for
an epoch is the unweighted mean of its scalar minibatch losses.

## Examples

``` r
if (torch::torch_is_installed()) {
  x <- torch::torch_tensor(matrix(c(0, 1, 1, 2), ncol = 1),
                           dtype = torch::torch_float64())$clone()
  y <- (x * 2)$clone()
  loader <- torch::dataloader(torch::tensor_dataset(x, y), batch_size = 2)
  net_type <- torch::nn_module("tiny_linear",
    initialize = function() self$weight <- torch::nn_parameter(
      torch::torch_zeros(1, 1, dtype = torch::torch_float64())),
    forward = function(z) z$matmul(self$weight))
  fit <- riem.train(net_type(), loader,
    loss = function(model, batch) (model(batch[[1]]) - batch[[2]])$square()$mean(),
    optimizer_factory = function(model) optim_rsgd(model$parameters, lr = 0.1),
    epochs = 2, device = "cpu")
  fit$history
}
#>   epoch     loss batches steps
#> 1     1 5.050000       2     2
#> 2     2 1.022625       2     4
```
