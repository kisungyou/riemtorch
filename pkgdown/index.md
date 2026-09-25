# riemtorch

<div class="rt-intro">
<p class="rt-eyebrow">RIEMANNIAN OPTIMIZATION IN R</p>
<p class="rt-lead">Your objective. Its geometry.<br>One torch workflow.</p>
<p>Optimize user-defined tensor objectives on spheres, matrix manifolds,
hyperbolic spaces, and products. Use automatic differentiation, choose a solver,
and run on a compatible CPU or GPU.</p>
<p class="rt-actions"><a class="btn btn-primary" href="articles/getting-started.html">Get started</a>
<a class="btn btn-outline-primary" href="articles/examples.html">Explore examples</a></p>
</div>

## A first optimization

Find the unit vector most aligned with a target. The sphere supplies the geometry;
the objective is an ordinary scalar torch expression.

```r
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

The run selects a compatible device automatically. Add `device = "cpu"` to
force CPU execution, including on a CUDA workstation. Registered `data` moves
with the problem. [Learn about devices and precision](articles/applications-and-devices.html).

## Learn with public data

Explore ten worked examples in the [example gallery](articles/examples.html),
from short introductions to complete applications. Every example includes
executable code, figures, and numerical checks.

<div class="rt-example-grid">
<a class="rt-example" href="articles/example-penguin-subspace.html"><strong>Penguin measurements</strong><span>Find a two-dimensional subspace and compare deterministic and finite-sum optimization.</span></a>
<a class="rt-example" href="articles/example-stackloss-regression.html"><strong>Industrial measurements</strong><span>Compare quadratic and Huber regression on observations from an ammonia plant.</span></a>
<a class="rt-example" href="articles/example-airquality-completion.html"><strong>Missing air-quality readings</strong><span>Fit a compact low-rank matrix and evaluate predictions on held-out observations.</span></a>
<a class="rt-example" href="articles/example-covariance-means.html"><strong>Covariance geometry</strong><span>Average covariance matrices from observed time series using two geometric metrics.</span></a>
<a class="rt-example" href="articles/example-tree-constraints.html"><strong>Tree-volume constraints</strong><span>Encode nonnegative scaling exponents and check the constrained optimum.</span></a>
<a class="rt-example" href="articles/example-mtcars-proximal.html"><strong>Sparse car-data regression</strong><span>Write an exact proximal operator and trace coefficients across penalty strengths.</span></a>
</div>

The data are built into R or bundled locally with attribution. See
[data sources and reproducibility](articles/example-data-sources.html) for
citations, preprocessing, and downloads.

## Choose your workflow

| I want to… | Start with |
|:--|:--|
| Define my own loss | [Custom objectives](articles/custom-objectives.html) |
| Choose a manifold and metric | [Geometry and metrics](articles/geometry-and-metrics.html) |
| Compare smooth solvers | [Deterministic optimization](articles/deterministic-optimization.html) |
| Check gradients or Hessians | [Derivative diagnostics](articles/checking-derivatives.html) |
| Work with large finite sums | [Variance reduction](articles/variance-reduction.html) |
| Train a torch module | [Torch training](articles/torch-training.html) |
| Add constraints or a proximal term | [Constrained and nonsmooth problems](articles/constrained-nonsmooth.html) |
| Find a function | [Function reference](reference/index.html) |

## Installation

Install the package from GitHub and set up the R torch runtime:

```r
install.packages(c("remotes", "torch"))
if (!torch::torch_is_installed()) torch::install_torch()
remotes::install_github("kisungyou/riemtorch")
```

To install a local checkout instead, open `riemtorch.Rproj` and use
**Build → Install Package** in RStudio.

riemtorch uses R torch as its numerical backend and contains no compiled extension
of its own. The [getting-started guide](articles/getting-started.html) walks through
installation, your first solve, and interpretation of the result.

## Know what is supported

`riem.capabilities(M)` describes the operations supported by a geometry.
`riem.support()` exposes the evidence tables. Solver convergence, derivative
support, and hardware qualification are separate questions; consult
[numerical limitations](articles/numerical-limitations.html) when selecting a method.

Manifold means and PCA here are worked optimization examples. For a collection of
statistical methods on manifolds, see the author's
[Riemann package](https://www.kisungyou.com/Riemann/).

## Build this documentation locally

With `riemtorch.Rproj` open in RStudio:

```r
install.packages("pkgdown")  # only needed once
pkgdown::build_site()
```

The site is written to `docs/` and opens in your browser in an interactive session.
You can also open `docs/index.html` directly. No deployment is needed.
