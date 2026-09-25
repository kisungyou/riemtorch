# Native Torch Riemannian Training Optimizers

Native Torch Riemannian Training Optimizers

## Usage

``` r
optim_rsgd(params, lr = 0.01, manifold = NULL, momentum = 0, metric_weight = 1)

optim_radam(
  params,
  lr = 0.001,
  manifold = NULL,
  betas = c(0.9, 0.999),
  eps = 1e-08,
  metric_weight = 1,
  amsgrad = FALSE
)

optim_radagrad(
  params,
  lr = 0.01,
  manifold = NULL,
  eps = 1e-08,
  metric_weight = 1
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

- momentum:

  RSGD coefficient in \[0,1).

- metric_weight:

  Positive factor metric weight, shared by group parameters.

- betas:

  RADAM first- and second-moment coefficients in \[0,1).

- eps:

  Positive constant outside the bias-corrected variance square root.

- amsgrad:

  Use the running maximum of uncorrected second moments.

## Value

A native torch optimizer with \`step()\`, \`zero_grad()\`,
\`state_dict()\` and \`load_state_dict()\`. Parameter identity is
preserved by in-place copies.

## Details

RSGD uses d=momentum\*m+grad, retracts -lr\*d, and transports d to the
new point. RADAM uses a transported first moment and one scalar second
moment of the weighted squared gradient norm per tensor factor
(independent states for factors of a power manifold). Sparse row
gradients are coalesced and use row-local bias-correction clocks;
untouched rows and states do not change. Both moments have standard bias
corrections. This is factorwise Riemannian Adam, not rectified Adam. It
does not perform coordinatewise ambient scaling or weight decay. Frozen
and missing-gradient parameters are skipped. A failing update validates
all parameters before committing any of them. Place tensors on their
device before optimizer construction. See \[riem.save()\] for complete
checkpoints.

## Examples

``` r
if (torch::torch_is_installed()) {
  x <- torch::nn_parameter(torch::torch_tensor(c(1, 0, 0),
                           dtype = torch::torch_float64()))
  opt <- optim_rsgd(list(list(params = list(x), manifold = manifold.sphere(3))),
                    lr = 0.1, momentum = 0.8)
  opt$zero_grad()
  (-x[2])$backward()
  opt$step()
  riem.belongs(manifold.sphere(3), x)
  adaptive <- optim_radam(list(x), lr = 0.01, manifold = manifold.sphere(3))
  adagrad <- optim_radagrad(list(x), lr = 0.01, manifold = manifold.sphere(3))
}
```
