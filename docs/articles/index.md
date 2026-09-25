# Articles

### Start here

Install the package, define an objective, and inspect your first result.

- [Get started with
  riemtorch](https://www.kisungyou.com/riemtorch/articles/getting-started.md):

  Define a constrained tensor objective, solve it, check the result, and
  choose CPU or GPU execution.

- [Example
  gallery](https://www.kisungyou.com/riemtorch/articles/examples.md):

  Ten complete optimization examples, from first steps to public-data
  applications, constraints, and sparse solutions.

- [Data sources and
  reproducibility](https://www.kisungyou.com/riemtorch/articles/example-data-sources.md):

  Where the example data come from, how they are prepared, and how to
  reproduce the website without dataset downloads.

### Foundations

Four small optimization problems with plots and independent checks.

- [Find a principal direction on the
  sphere](https://www.kisungyou.com/riemtorch/articles/example-sphere-pca.md):

  A complete optimization example with iris data, a convergence plot,
  and an independent eigenvector check.

- [Fit a line with a robust
  loss](https://www.kisungyou.com/riemtorch/articles/example-robust-regression.md):

  Compare ordinary least squares and Huber loss using the same residual
  function and verify both answers independently.

- [Average positive-definite matrices
  geometrically](https://www.kisungyou.com/riemtorch/articles/example-spd-mean.md):

  Optimize an affine-invariant midpoint, visualize covariance ellipses,
  and check a closed-form matrix reference.

- [Complete a partially observed low-rank
  matrix](https://www.kisungyou.com/riemtorch/articles/example-matrix-completion.md):

  Recover missing entries with compact rank-two factors and a Riemannian
  L-BFGS solver.

### Public-data applications

Work with biological, industrial, environmental, and time-series
measurements.

- [Find a two-dimensional penguin
  subspace](https://www.kisungyou.com/riemtorch/articles/example-penguin-subspace.md):

  Optimize a Grassmann subspace for Palmer Penguins measurements,
  compare with PCA, and try a finite-sum solver.

- [Robust regression on industrial
  measurements](https://www.kisungyou.com/riemtorch/articles/example-stackloss-regression.md):

  Fit quadratic and Huber regression to stackloss data, inspect weights,
  and verify both optima with independent base R calculations.

- [Reconstruct missing environmental
  measurements](https://www.kisungyou.com/riemtorch/articles/example-airquality-completion.md):

  Fit compact low-rank factors to New York air-quality measurements,
  with a leakage-free holdout and a simple imputation baseline.

- [Average covariance matrices
  geometrically](https://www.kisungyou.com/riemtorch/articles/example-covariance-means.md):

  Compare log-Euclidean and affine-invariant averages of covariance
  matrices from European stock-index returns.

### Constraints and sparsity

Express modeling assumptions and penalties through optimization
callbacks.

- [Fit a constrained tree-volume
  model](https://www.kisungyou.com/riemtorch/articles/example-tree-constraints.md):

  Use equality and inequality constraints for a log-volume model, then
  verify the solution by eliminating one slope.

- [Sparse regression with a proximal
  operator](https://www.kisungyou.com/riemtorch/articles/example-mtcars-proximal.md):

  Follow a five-penalty Lasso path with an exact Euclidean proximal map
  and verify it using coordinate descent and optimality conditions.

### Geometry and objectives

- [Geometry, metrics and
  representations](https://www.kisungyou.com/riemtorch/articles/geometry-and-metrics.md):
- [Custom tensor
  objectives](https://www.kisungyou.com/riemtorch/articles/custom-objectives.md):
- [Products, blocks and finite
  sums](https://www.kisungyou.com/riemtorch/articles/structured-problems.md):
- [Compact low-rank
  optimization](https://www.kisungyou.com/riemtorch/articles/compact-lowrank.md):
- [Optimization with complex
  geometry](https://www.kisungyou.com/riemtorch/articles/complex-geometry.md):

### Solvers and numerical checks

- [Deterministic optimization and
  diagnostics](https://www.kisungyou.com/riemtorch/articles/deterministic-optimization.md):
- [Hessians and least-squares
  problems](https://www.kisungyou.com/riemtorch/articles/second-order-and-leastsquares.md):
- [Checking geometry and
  derivatives](https://www.kisungyou.com/riemtorch/articles/checking-derivatives.md):
- [Variance reduction and solver
  controls](https://www.kisungyou.com/riemtorch/articles/variance-reduction.md):
- [Constraints and intrinsic nonsmooth
  optimization](https://www.kisungyou.com/riemtorch/articles/constrained-nonsmooth.md):

### Training and execution

- [Mixed-parameter torch training and
  checkpoints](https://www.kisungyou.com/riemtorch/articles/torch-training.md):
- [Sparse embeddings and resumable
  training](https://www.kisungyou.com/riemtorch/articles/sparse-training.md):
- [Applications and Portable Device
  Execution](https://www.kisungyou.com/riemtorch/articles/applications-and-devices.md):

### Extending the package and understanding limits

- [Adding a geometry without changing a
  solver](https://www.kisungyou.com/riemtorch/articles/extending-riemtorch.md):
- [Numerical domains and
  reproducibility](https://www.kisungyou.com/riemtorch/articles/numerical-limitations.md):
