# Armijo Riemannian Training Optimizer

Armijo Riemannian Training Optimizer

## Usage

``` r
optim_rlinesearch(
  params,
  lr = 1,
  manifold = NULL,
  metric_weight = 1,
  backtrack = 0.5,
  armijo = 1e-04,
  max_backtracks = 25L
)
```

## Arguments

- params:

  Tensor, list of tensors, or torch parameter groups. A group can supply
  \`manifold\`, \`lr\`, \`metric_weight\`, and unique character
  \`keys\`. A missing manifold means Euclidean geometry with the
  parameter's shape.

- lr:

  Positive learning rate.

- manifold:

  Default geometry for each tensor parameter.

- metric_weight:

  Positive factor metric weight, shared by group parameters.

- backtrack:

  Step reduction factor in (0,1).

- armijo:

  Sufficient decrease coefficient in (0,1).

- max_backtracks:

  Maximum reductions before failure.

## Value

A torch optimizer. \`step(closure)\` requires a deterministic closure
that clears gradients, computes a real scalar loss, calls backward and
returns the loss. Failed searches restore all parameter values and
gradients.

## Details

Closures must not change buffers, random state, or external state. Each
search varies all parameter groups together. Sparse gradients are not
supported by this optimizer. Parameter identity is preserved.

## Examples

``` r
if(torch::torch_is_installed()) {
  x <- torch::nn_parameter(torch::torch_ones(2,dtype=torch::torch_float64()))
  opt <- optim_rlinesearch(list(x))
  opt$step(function() {opt$zero_grad();z<-x$square()$sum();z$backward();z})
}
```
