# Mixed-parameter torch training and checkpoints

A native torch module can mix an orthogonal matrix with ordinary
parameters. This reconstruction model uses an orthogonal two-dimensional
feature frame and an unconstrained offset. Iris is a small, built-in
public-data example.

``` r

X <- torch_tensor(as.matrix(iris[, 1:4]), dtype = torch_float64())
X <- X / X$abs()$max()
model_type <- nn_module("orthogonal_reconstruction",
  initialize = function() {
    self$frame <- nn_parameter(riem.random(
      manifold.stiefel(4, 2, "euclidean"), device = "cpu"))
    self$offset <- nn_parameter(torch_zeros(4, dtype = torch_float64()))
  },
  forward = function(x) {
    z <- x - self$offset
    z$matmul(self$frame)$matmul(self$frame$t()) + self$offset
  })
model <- model_type()
M <- manifold.stiefel(4, 2, "euclidean")
opt <- optim_radam(list(
  list(params = list(model$frame), manifold = M, keys = "frame"),
  list(params = list(model$offset), keys = "offset")
), lr = 0.02)
losses <- numeric(20)
for (i in seq_along(losses)) {
  opt$zero_grad()
  loss <- (model(X) - X)$square()$mean()
  loss$backward()
  opt$step()
  losses[i] <- loss$item()
}
range(losses)
#> [1] 0.02490891 0.17598219
stopifnot(riem.belongs(M, model$frame))
```

Each parameter tensor is one adaptive factor. Its variance is a scalar
squared metric gradient norm, while first moments are transported.
Arbitrary ambient coordinatewise scaling is not used. Parameter identity
stays intact through in-place copies. Frozen/missing-gradient parameters
are skipped, and invalid candidate updates fail before any group is
committed.

``` r

path <- tempfile(fileext = ".pt")
riem.save(opt, path)
riem.load(opt, path)
unlink(path)
```

Create matching parameter groups on the final device before loading.
Named keys detect mismatched order. Version 2 checkpoints can also carry
model buffers, training progress, caller-owned data-order metadata, and
random-generator state.

For a complete dataloader loop,
[`riem.train()`](https://www.kisungyou.com/riemtorch/reference/riem.train.md)
resolves one suitable device, moves the module before constructing the
optimizer, and transfers each minibatch. Pass `device = "cpu"` to force
CPU on a GPU workstation, or a value such as `device = "cuda:1"` to
select a particular visible GPU. Omitting `device` uses the package
selection policy. The returned history and execution metadata make the
selected placement explicit.
