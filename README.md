# riemtorch

**riemtorch** is an R package for Riemannian optimization with
[torch](https://torch.mlverse.org/). Define an objective, choose a manifold,
and optimize using automatic differentiation on a compatible CPU or GPU.
Supported geometries include spheres, Stiefel and Grassmann manifolds,
positive-definite matrices, low-rank matrices, and products.

## Installation

Install from GitHub, with the R torch runtime:

```r
install.packages(c("remotes", "torch"))
if (!torch::torch_is_installed()) torch::install_torch()
remotes::install_github("kisungyou/riemtorch")
```

## Example

Find the unit vector most aligned with a given direction:

```r
library(torch)
library(riemtorch)

sphere <- manifold.sphere(3)
target <- torch_tensor(c(1, 2, 3), dtype = torch_float64())
x0 <- torch_tensor(c(1, 0, 0), dtype = torch_float64())

problem <- riem.problem(
  sphere,
  fn = function(x, data) -torch_sum(x * data),
  data = target
)
fit <- riem.optimize(problem, x0, method = "conjugate_gradient")
fit$point  # approximately c(0.2673, 0.5345, 0.8018)
```

Runs select a compatible device automatically. Use `device = "cpu"` to force
CPU execution or `device = "cuda:0"` to request a specific available GPU.

## Documentation

See the [documentation website](https://www.kisungyou.com/riemtorch/),
[worked examples](https://www.kisungyou.com/riemtorch/articles/examples.html),
and [function reference](https://www.kisungyou.com/riemtorch/reference/index.html).
Report problems through [GitHub issues](https://github.com/kisungyou/riemtorch/issues).

## License

GPL (>= 2).
