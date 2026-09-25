# Package index

## Define and solve a problem

User-defined scalar objectives, structured losses, and optimization
drivers.

- [`riemtorch`](https://www.kisungyou.com/riemtorch/reference/riemtorch-package.md)
  [`riemtorch-package`](https://www.kisungyou.com/riemtorch/reference/riemtorch-package.md)
  : Optimization on Riemannian Manifolds with Torch
- [`riem.problem()`](https://www.kisungyou.com/riemtorch/reference/riem.problem.md)
  : Define a Scalar Manifold Optimization Problem
- [`riem.problem.finitesum()`](https://www.kisungyou.com/riemtorch/reference/riem.problem.finitesum.md)
  : Finite-Sum Problems with Explicit Normalization
- [`riem.problem.leastsquares()`](https://www.kisungyou.com/riemtorch/reference/riem.problem.leastsquares.md)
  : Least-Squares Objectives and Metric Adjoints
- [`riem.problem.constrained()`](https://www.kisungyou.com/riemtorch/reference/riem.problem.constrained.md)
  : Smooth Equality and Inequality Constrained Problems
- [`riem.problem.composite()`](https://www.kisungyou.com/riemtorch/reference/riem.problem.composite.md)
  : Smooth Plus Nonsmooth Manifold Problems
- [`riem.optimize()`](https://www.kisungyou.com/riemtorch/reference/riem.optimize.md)
  : Optimize a User-Defined Manifold Objective
- [`riem.multistart()`](https://www.kisungyou.com/riemtorch/reference/riem.multistart.md)
  : Run Reproducible Multiple Starts on One Device
- [`riem.evaluate()`](https://www.kisungyou.com/riemtorch/reference/riem.evaluate.md)
  [`riem.hessian()`](https://www.kisungyou.com/riemtorch/reference/riem.evaluate.md)
  : Evaluate Values and Metric Derivatives

## Choose a geometry

### Vectors, tensors, and curved spaces

- [`manifold.euclidean()`](https://www.kisungyou.com/riemtorch/reference/manifold.euclidean.md)
  : Euclidean Tensor Geometry
- [`manifold.sphere()`](https://www.kisungyou.com/riemtorch/reference/manifold.sphere.md)
  : Round Sphere Geometry
- [`manifold.oblique()`](https://www.kisungyou.com/riemtorch/reference/manifold.oblique.md)
  [`manifold.multinomial()`](https://www.kisungyou.com/riemtorch/reference/manifold.oblique.md)
  : Oblique Matrices and the Fisher–Rao Simplex
- [`manifold.hyperbolic()`](https://www.kisungyou.com/riemtorch/reference/manifold.hyperbolic.md)
  : Hyperbolic Ball and Hyperboloid
- [`manifold.torus()`](https://www.kisungyou.com/riemtorch/reference/manifold.torus.md)
  : Flat Torus in Angle Coordinates
- [`manifold.positive()`](https://www.kisungyou.com/riemtorch/reference/manifold.positive.md)
  : Positive Tensors with a Log-Euclidean Metric
- [`manifold.complexcircle()`](https://www.kisungyou.com/riemtorch/reference/manifold.complexcircle.md)
  [`manifold.unitary()`](https://www.kisungyou.com/riemtorch/reference/manifold.complexcircle.md)
  : Complex Phases and Unitary Matrices

### Frames, matrix groups, and subspaces

- [`manifold.stiefel()`](https://www.kisungyou.com/riemtorch/reference/manifold.stiefel.md)
  [`manifold.grassmann()`](https://www.kisungyou.com/riemtorch/reference/manifold.stiefel.md)
  : Stiefel and Grassmann Frame Geometries
- [`manifold.stiefel.generalized()`](https://www.kisungyou.com/riemtorch/reference/manifold.stiefel.generalized.md)
  [`manifold.grassmann.generalized()`](https://www.kisungyou.com/riemtorch/reference/manifold.stiefel.generalized.md)
  : Generalized Orthogonality Geometries
- [`manifold.rotation()`](https://www.kisungyou.com/riemtorch/reference/manifold.rotation.md)
  [`manifold.rigidmotion()`](https://www.kisungyou.com/riemtorch/reference/manifold.rotation.md)
  : Rotations and Rigid Motions
- [`manifold.affine()`](https://www.kisungyou.com/riemtorch/reference/manifold.affine.md)
  [`manifold.orthogonal()`](https://www.kisungyou.com/riemtorch/reference/manifold.affine.md)
  : Affine Subspaces and Orthogonal Matrices

### Positive-definite, low-rank, and constrained matrices

- [`manifold.spd()`](https://www.kisungyou.com/riemtorch/reference/manifold.spd.md)
  : Symmetric Positive-Definite Matrix Geometry
- [`manifold.spdk()`](https://www.kisungyou.com/riemtorch/reference/manifold.spdk.md)
  : Fixed-Rank Positive-Semidefinite Geometry
- [`manifold.fixedrank()`](https://www.kisungyou.com/riemtorch/reference/manifold.fixedrank.md)
  : Fixed-Rank Rectangular Matrix Geometry
- [`manifold.elliptope()`](https://www.kisungyou.com/riemtorch/reference/manifold.elliptope.md)
  [`manifold.spectrahedron()`](https://www.kisungyou.com/riemtorch/reference/manifold.elliptope.md)
  : Fixed-Rank Elliptope and Spectrahedron
- [`manifold.correlation()`](https://www.kisungyou.com/riemtorch/reference/manifold.correlation.md)
  : Full-Rank Correlation Geometries
- [`manifold.doublystochastic()`](https://www.kisungyou.com/riemtorch/reference/manifold.doublystochastic.md)
  : Positive Doubly Stochastic Matrix Geometry
- [`manifold.landmark()`](https://www.kisungyou.com/riemtorch/reference/manifold.landmark.md)
  : Kendall Landmark Shape Geometry

### Combine and scale geometries

- [`manifold.product()`](https://www.kisungyou.com/riemtorch/reference/manifold.product.md)
  : Weighted Product Geometry
- [`manifold.power()`](https://www.kisungyou.com/riemtorch/reference/manifold.power.md)
  [`manifold.scaled()`](https://www.kisungyou.com/riemtorch/reference/manifold.power.md)
  : Repeated and Scaled Manifold Geometries

## Operate on manifold points

- [`riem.random()`](https://www.kisungyou.com/riemtorch/reference/riem.random.md)
  : Generate Feasible Initial Points
- [`riem.belongs()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.tangent()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.istangent()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.inner()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.norm()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.egrad2rgrad()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.retr()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.transport()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.project()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.exp()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.log()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.sqdist()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.dist()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  [`riem.ehess2rhess()`](https://www.kisungyou.com/riemtorch/reference/riem.belongs.md)
  : Inspect and Operate on Manifold Points
- [`riem.materialize()`](https://www.kisungyou.com/riemtorch/reference/riem.materialize.md)
  : Materialize or Extract Entries from a Compact Matrix Point
- [`riem.chart()`](https://www.kisungyou.com/riemtorch/reference/riem.chart.md)
  [`riem.chart.inverse()`](https://www.kisungyou.com/riemtorch/reference/riem.chart.md)
  : Correlation Chart Coordinates
- [`riem.matrix.function()`](https://www.kisungyou.com/riemtorch/reference/riem.matrix.function.md)
  [`riem.matrix.frechet()`](https://www.kisungyou.com/riemtorch/reference/riem.matrix.function.md)
  : Stable Symmetric Matrix Functions

## Check derivatives and capabilities

- [`riem.check.manifold()`](https://www.kisungyou.com/riemtorch/reference/riem.check.manifold.md)
  [`riem.check.gradient()`](https://www.kisungyou.com/riemtorch/reference/riem.check.manifold.md)
  [`riem.check.hessian()`](https://www.kisungyou.com/riemtorch/reference/riem.check.manifold.md)
  [`riem.check.adjoint()`](https://www.kisungyou.com/riemtorch/reference/riem.check.manifold.md)
  : Check a Manifold or User-Supplied Derivatives
- [`riem.hessian.spectrum()`](https://www.kisungyou.com/riemtorch/reference/riem.hessian.spectrum.md)
  : Estimate Extreme Riemannian Hessian Eigenvalues
- [`riem.capabilities()`](https://www.kisungyou.com/riemtorch/reference/riem.capabilities.md)
  : Inspect Primitive-Level Geometry Capabilities
- [`riem.support()`](https://www.kisungyou.com/riemtorch/reference/riem.support.md)
  : Inspect Published Geometry and Solver Evidence
- [`riem.manifold()`](https://www.kisungyou.com/riemtorch/reference/riem.manifold.md)
  : Define a Manifold Through Public Geometry Contracts

## Train torch models

- [`optim_rsgd()`](https://www.kisungyou.com/riemtorch/reference/optim_rsgd.md)
  [`optim_radam()`](https://www.kisungyou.com/riemtorch/reference/optim_rsgd.md)
  [`optim_radagrad()`](https://www.kisungyou.com/riemtorch/reference/optim_rsgd.md)
  : Native Torch Riemannian Training Optimizers
- [`optim_rlinesearch()`](https://www.kisungyou.com/riemtorch/reference/optim_rlinesearch.md)
  : Armijo Riemannian Training Optimizer
- [`riem.train()`](https://www.kisungyou.com/riemtorch/reference/riem.train.md)
  : Train a Torch Module on One Resolved Device
- [`riem.schedule()`](https://www.kisungyou.com/riemtorch/reference/riem.schedule.md)
  : Learning-Rate Schedules
- [`riem.save()`](https://www.kisungyou.com/riemtorch/reference/riem.save.md)
  [`riem.load()`](https://www.kisungyou.com/riemtorch/reference/riem.save.md)
  : Save and Restore Training Parameters and Optimizer State

## Devices and precision

- [`riem.devices()`](https://www.kisungyou.com/riemtorch/reference/riem.devices.md)
  [`riem.device()`](https://www.kisungyou.com/riemtorch/reference/riem.devices.md)
  : Discover and Select Torch Execution Devices
- [`riem.to()`](https://www.kisungyou.com/riemtorch/reference/riem.to.md)
  : Move Tensor Trees and Device-Bound Geometry
