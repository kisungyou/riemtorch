# riemtorch

Riemannian optimization of **your own torch objective**. Choose a geometry,
define a scalar loss, and optimize tensors or named products. No Rcpp compiler,
Python runtime, or external numerical solver is needed by this package.

## Installation

Install R torch and its runtime explicitly once:

```r
install.packages("torch")
torch::install_torch()
```

Then install this source package using `R CMD INSTALL riemtorch`, or
`install.packages("riemtorch_0.1.0.tar.gz", repos = NULL, type = "source")`.
Loading riemtorch does not download dependencies or initialize an accelerator.
The tested reference environment is R 4.5.2, torch 0.17.0, libtorch 2.8.0,
CPU float64 on macOS arm64. The torch runtime is a separate dependency; the
riemtorch source itself contains no compiled extension.

## Devices and precision

Every managed optimization or training run resolves one device before it starts.
Automatic selection tries compatible visible CUDA devices, compatible MPS, then
CPU. It preserves the initial point or model precision; it does not silently
downcast float64 to reach an accelerator.

```r
riem.devices()                         # visible devices and compatibility probes
fit <- riem.optimize(problem, x0)      # automatic selection
fit <- riem.optimize(problem, x0, device = "cpu")
fit <- riem.optimize(problem, x0, device = "cuda:1")
options(riemtorch.device = "cpu")      # session default
```

Resolution follows an explicit function argument, the `riemtorch.device`
option, the `RIEMTORCH_DEVICE` environment variable, then `"auto"`. An explicit
`device = "auto"` bypasses configured defaults. Unavailable explicit devices
are errors; automatic discovery skips incompatible accelerators. Numerical or
memory failures after a run begins do not trigger a silent restart elsewhere.
The result's `execution` field records the request, selected device, dtype and
runtime versions. CUDA and MPS support is runtime-checked; neither was available
in the local CPU reference environment.

## A complete custom-objective solve

```r
library(torch)
library(riemtorch)
torch_manual_seed(1)
M <- manifold.sphere(3)
a <- torch_tensor(c(1, 2, 3), dtype = torch_float64())
x0 <- torch_tensor(c(1, 0, 0), dtype = torch_float64())
problem <- riem.problem(M, function(x, data) -torch_sum(x * data), data = a)
fit <- riem.optimize(problem, x0, "conjugate_gradient")
fit
fit$point                         # a / sqrt(14)
fit$constraint_residuals
fit$evaluations
fit$execution
```

`fn` must return a scalar torch tensor. Explicit `egrad`, `rgrad` and `rhess`
callbacks are also supported. Registered `data` moves with the problem and is
passed to objective and derivative callbacks; tensors captured privately by a
closure remain caller-owned. Deterministic objectives must be fixed during a
solve. Product leaves are named tensors; unequal metric weights are handled in
gradient conversion. Geometry kernels support leading batch axes; one solve
optimizes one point or product, not a batch of independent problems.

## Geometry coverage

All 29 retained guideline rows have first-order implementations and fixtures,
including source aliases; the Riemann spdk Bures factor representation and reflection-inclusive landmark
quotient are additional audited variants. Families include Euclidean, sphere, oblique,
Fisher--Rao simplex, both hyperbolic models, torus, both Stiefel metrics,
Grassmann frames/projectors, generalized frames/subspaces, rotations/rigid
motions, AIRM/LERM/Bures SPD, fixed-rank rectangular/PSD matrices, elliptope,
spectrahedron, three correlation metrics, Kendall shape, and weighted products.

Use `riem.support("geometry")` for the inventory and
`riem.capabilities(M)` for primitive derivative qualifications. First-order
readiness does not imply exact logarithms or Hessians on every geometry. Matrix
log/root custom derivatives are first-order only. Rank/shape manifolds are
restricted to the smooth strata documented in their help pages.

Native leading-batch kernels cover Euclidean, sphere, oblique, torus, Stiefel,
both Grassmann representations, and all three full-rank SPD metrics. Custom
manifolds may supply optional batch kernels and otherwise use the pointwise
contract. Exact exp/local-log maps are available for oblique, Fisher--Rao
simplex, and Poincare-ball and hyperboloid models, subject to their documented
domains.

## Solver coverage

The deterministic core includes steepest descent, PR+/FR/HS+/DY conjugate gradient,
BB1/BB2/alternating steps, transported L-BFGS, trust regions, Newton-CG, Gauss-Newton and LM.
Finite-sum SGD and cyclic product-block gradients are available. Approximate
cubic regularization, particle swarm, local manifold Nelder-Mead, Stiefel
annealing and Grassmann MACG search are implemented as **experimental** methods.
Their mathematical variants and stopping rules are explicit. No heuristic
returns a fabricated stationarity certificate.

Automatic exact HVPs are tested for Euclidean, sphere, oblique, flat torus,
Euclidean and canonical Stiefel, Grassmann frames/projectors, generalized Stiefel/Grassmann,
rotations, AIRM SPD and weighted products whose factors all support conversion,
provided the objective supports double backward. Other geometries accept exact
`rhess` callbacks. Strong Wolfe requires the derivative of the actual retraction
curve. See `riem.support("solvers")` and the concrete combinations in
`riem.support("support")`.

Finite-sum problems accept a vectorized `batch_fn(x, indices, data)`. Full
initial/final diagnostics are accumulated in bounded chunks, and SGD evaluates
only minibatches between them unless `full_evaluation_every` requests a periodic
full check. Intermediate minibatch gradient norms are labelled and never treated
as full stationarity certificates.

## Torch training

`optim_rsgd()` and `optim_radam()` work in ordinary torch loops. `riem.train()`
adds a managed loop that selects a device once, moves the module before the
optimizer is constructed, and moves every minibatch before evaluating the loss.

```r
x <- torch_tensor(matrix(c(0, 1, 1, 2), ncol = 1),
                  dtype = torch_float64())$clone()
y <- (x * 2)$clone()
loader <- dataloader(tensor_dataset(x, y), batch_size = 2)
tiny_linear <- nn_module("tiny_linear",
  initialize = function() self$weight <- nn_parameter(
    torch_zeros(1, 1, dtype = torch_float64())),
  forward = function(z) z$matmul(self$weight))
fit <- riem.train(tiny_linear(), loader,
  loss = function(model, batch)
    (model(batch[[1]]) - batch[[2]])$square()$mean(),
  optimizer_factory = function(model)
    optim_rsgd(model$parameters, lr = 0.05),
  epochs = 2, device = "cpu")
fit$history
```

`optim_radam()` adapts one scalar variance per declared tensor factor (one per repeated power factor), with
transported first moments. Updates preserve parameter identity and validate all
groups before committing. Version-two checkpoints may include module parameters
and buffers, epoch/step or data-order metadata, and R/torch RNG states. Restore a
GPU-created checkpoint on CPU by constructing the model and optimizer there and
loading with `device = "cpu"`; cross-device continuation is not promised to be
bitwise identical.

## Local documentation website

Open `riemtorch.Rproj` in RStudio and run:

```r
install.packages("pkgdown")  # one-time setup
pkgdown::build_site()
```

This builds the complete website in `docs/`, including function examples,
the package vignettes, a getting-started guide, and ten illustrated examples.
Four introductory examples cover PCA, robust regression, SPD means, and matrix
completion. Six public-data applications use penguin measurements, industrial
stack loss, air quality, stock-index covariances, tree volumes, and car data.
Dataset downloads are not needed during the build. A working torch runtime is
required; see the installation steps
above. RStudio supplies Pandoc. Use `pkgdown::preview_site()` to reopen the local
site, or open `docs/index.html` directly. See
[the website maintenance guide](pkgdown/README.md) for source locations and
command-line prerequisites.

## Development and evidence

Roxygen sources generate `man/` and `NAMESPACE`. Public help topics have small
runnable examples guarded by `torch_is_installed()`. Fifteen executed vignettes
cover objectives, geometry, solvers, structured problems, training, second-order
methods, extensions, applications/device selection and numerical limitations.
The tests verify
metric duality, retraction consistency, quotient invariance, analytic optima,
Hessians, adjoints, native batching, device placement, scalable finite sums,
failure states, parameter identity and checkpoints.

```r
roxygen2::roxygenise(".")
testthat::test_local(".")
```

From the parent directory use `R CMD build riemtorch` and
`R CMD check --as-cran riemtorch_0.1.0.tar.gz`. The optional benchmark script in
`development/benchmarks/` compares matched accuracy, retains failed runs, and
separates objective/gradient costs from total solve time. The installed
`math/geometry.md` and `math/solvers.md` record formulas, domains and references.
The implementation report records actual local checks and their limits.

## Competitor upgrade

Public numerical checks are available through `riem.check.manifold()`,
`riem.check.gradient()`, `riem.check.hessian()`, and `riem.check.adjoint()`.
`riem.hessian.spectrum()` returns Ritz estimates and residuals.
`riem.support("competitors")` records evidence and remaining qualifications.

Smooth solver additions include preconditioned Newton/trust-region subproblems,
CG/BB variants, SVRG/SRG, robust least squares and `riem.multistart()`.
Callbacks, time/evaluation budgets and fused value/Riemannian-gradient callbacks
are optional. `manifold.power()` represents repeated factors within one point.
Compact fixed-rank points use `representation="svd"` and `riem.materialize()`
can read selected entries without constructing a dense matrix.

Training also offers AMSGrad, `optim_radagrad()`, `optim_rlinesearch()`, sparse
row updates and epoch-level validation, accumulation, metric clipping and
scheduler/data-state hooks. New optimizer layouts are versioned independently
from the checkpoint container, and earlier layouts remain readable.

New real geometries include positive tensors, affine subspaces, orthogonal
matrices and positive doubly stochastic matrices. Complex Euclidean, sphere,
phase, Stiefel, Grassmann and unitary geometries use real Hermitian metrics.
Their objectives remain real scalars. Complex frame Hessians remain unqualified.

`riem.problem.constrained()` adds equality and inequality constraints with an
augmented-Lagrangian solver and explicit KKT reporting.
`riem.problem.composite()` adds intrinsic proximal-gradient and cyclic proximal
point methods. User proximal callbacks must solve the stated manifold-distance
subproblem. A cycle residual alone does not establish stationarity.

The eight-family benchmark suite records 48 seeded riemtorch/Pymanopt/SciPy runs,
with configurations, failures, evaluation counts, sampled peak process memory
and timings in `development/benchmarks/upgrade/`. These are small correctness
comparisons, not a package-wide performance ranking. Memory experiments separate
detached diagnostic values, gradients and HVPs from full differentiable values,
whose retained computation graphs necessarily grow with the number of terms.
CUDA/MPS and nonlocal operating systems remain awaiting actual qualification;
see `development/qualification/` for the reproducible runner and CI workflow.
