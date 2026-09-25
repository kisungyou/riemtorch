# riemtorch

RIEMANNIAN OPTIMIZATION IN R

Your objective. Its geometry.  
One torch workflow.

Optimize user-defined tensor objectives on spheres, matrix manifolds,
hyperbolic spaces, and products. Use automatic differentiation, choose a
solver, and run on a compatible CPU or GPU.

[Get
started](https://www.kisungyou.com/riemtorch/articles/getting-started.md)
[Explore
examples](https://www.kisungyou.com/riemtorch/articles/examples.md)

## A first optimization

Find the unit vector most aligned with a target. The sphere supplies the
geometry; the objective is an ordinary scalar torch expression.

``` r

library(torch)
library(riemtorch)

M <- manifold.sphere(3)
target <- torch_tensor(c(1, 2, 3), dtype = torch_float64())
x0 <- torch_tensor(c(1, 0, 0), dtype = torch_float64())
problem <- riem.problem(
  M, function(x, data) -torch_sum(x * data), data = target
)

fit <- riem.optimize(problem, x0, method = "conjugate_gradient")
fit$point  # approximately c(0.2673, 0.5345, 0.8018)
fit$termination
```

The run selects a compatible device automatically. Add `device = "cpu"`
to force CPU execution, including on a CUDA workstation. Registered
`data` moves with the problem. [Learn about devices and
precision](https://www.kisungyou.com/riemtorch/articles/applications-and-devices.md).

## Learn with public data

Explore ten worked examples in the [example
gallery](https://www.kisungyou.com/riemtorch/articles/examples.md), from
short introductions to complete applications. Every example includes
executable code, figures, and numerical checks.

[**Penguin measurements**Find a two-dimensional subspace and compare
deterministic and finite-sum
optimization.](https://www.kisungyou.com/riemtorch/articles/example-penguin-subspace.md)
[**Industrial measurements**Compare quadratic and Huber regression on
observations from an ammonia
plant.](https://www.kisungyou.com/riemtorch/articles/example-stackloss-regression.md)
[**Missing air-quality readings**Fit a compact low-rank matrix and
evaluate predictions on held-out
observations.](https://www.kisungyou.com/riemtorch/articles/example-airquality-completion.md)
[**Covariance geometry**Average covariance matrices from observed time
series using two geometric
metrics.](https://www.kisungyou.com/riemtorch/articles/example-covariance-means.md)
[**Tree-volume constraints**Encode nonnegative scaling exponents and
check the constrained
optimum.](https://www.kisungyou.com/riemtorch/articles/example-tree-constraints.md)
[**Sparse car-data regression**Write an exact proximal operator and
trace coefficients across penalty
strengths.](https://www.kisungyou.com/riemtorch/articles/example-mtcars-proximal.md)

The data are built into R or bundled locally with attribution. See [data
sources and
reproducibility](https://www.kisungyou.com/riemtorch/articles/example-data-sources.md)
for citations, preprocessing, and downloads.

## Choose your workflow

| I want to… | Start with |
|:---|:---|
| Define my own loss | [Custom objectives](https://www.kisungyou.com/riemtorch/articles/custom-objectives.md) |
| Choose a manifold and metric | [Geometry and metrics](https://www.kisungyou.com/riemtorch/articles/geometry-and-metrics.md) |
| Compare smooth solvers | [Deterministic optimization](https://www.kisungyou.com/riemtorch/articles/deterministic-optimization.md) |
| Check gradients or Hessians | [Derivative diagnostics](https://www.kisungyou.com/riemtorch/articles/checking-derivatives.md) |
| Work with large finite sums | [Variance reduction](https://www.kisungyou.com/riemtorch/articles/variance-reduction.md) |
| Train a torch module | [Torch training](https://www.kisungyou.com/riemtorch/articles/torch-training.md) |
| Add constraints or a proximal term | [Constrained and nonsmooth problems](https://www.kisungyou.com/riemtorch/articles/constrained-nonsmooth.md) |
| Find a function | [Function reference](https://www.kisungyou.com/riemtorch/reference/index.md) |

## Installation

Install the package from GitHub and set up the R torch runtime:

``` r

install.packages(c("remotes", "torch"))
if (!torch::torch_is_installed()) torch::install_torch()
remotes::install_github("kisungyou/riemtorch")
```

To install a local checkout instead, open `riemtorch.Rproj` and use
**Build → Install Package** in RStudio.

riemtorch uses R torch as its numerical backend and contains no compiled
extension of its own. The [getting-started
guide](https://www.kisungyou.com/riemtorch/articles/getting-started.md)
walks through installation, your first solve, and interpretation of the
result.

## Know what is supported

`riem.capabilities(M)` describes the operations supported by a geometry.
[`riem.support()`](https://www.kisungyou.com/riemtorch/reference/riem.support.md)
exposes the evidence tables. Solver convergence, derivative support, and
hardware qualification are separate questions; consult [numerical
limitations](https://www.kisungyou.com/riemtorch/articles/numerical-limitations.md)
when selecting a method.

Manifold means and PCA here are worked optimization examples. For a
collection of statistical methods on manifolds, see the author’s
[Riemann package](https://www.kisungyou.com/Riemann/).

## Build this documentation locally

With `riemtorch.Rproj` open in RStudio:

``` r

install.packages("pkgdown")  # only needed once
pkgdown::build_site()
```

The site is written to `docs/` and opens in your browser in an
interactive session. You can also open `docs/index.html` directly. No
deployment is needed.
