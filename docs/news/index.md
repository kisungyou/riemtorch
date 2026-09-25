# Changelog

## riemtorch 0.1.0

Initial release, consolidating the implementation and documentation
developed under the earlier internal milestone version numbers.

### Geometry and derivatives

- Provide a torch-only R implementation with explicit metric variants,
  weighted products, repeated and scaled manifolds, and the audited
  Riemann spdk factor representation.
- Include positive, affine, orthogonal and positive doubly stochastic
  geometries, together with dense and compact fixed-rank
  representations.
- Support complex Euclidean, sphere, phase, Stiefel, Grassmann and
  unitary geometry with real Hermitian metrics and precision-preserving
  losses.
- Provide native leading-batch kernels for common vector, frame,
  Grassmann and SPD geometries, plus a batch-kernel contract for custom
  manifolds.
- Include qualified ambient-to-Riemannian Hessian conversions, frame
  maps, and exact exponential or local logarithm maps for supported
  geometries. State cut-locus, positivity, rank and derivative
  restrictions explicitly.

### Problems and optimization

- Provide scalar, finite-sum and least-squares problems with registered
  data, automatic differentiation and explicit metric derivative
  contracts.
- Include deterministic, stochastic, second-order, block and
  experimental search solvers with termination reasons and evaluation
  diagnostics.
- Support work and time budgets, detached callbacks, fused
  value/gradient callbacks, preconditioning, conjugate-gradient and
  Barzilai–Borwein variants, SVRG/SRG, robust least squares and multiple
  starts.
- Stream full finite-sum gradients and Hessian-vector products with
  detached accumulators; support vectorized minibatches and configurable
  full checks.
- Support smooth equality and inequality constraints through augmented
  Lagrangian optimization with multipliers and explicit KKT residuals.
- Support composite problems, intrinsic proximal gradient and cyclic
  proximal point methods with documented manifold proximal-map
  contracts.

### Devices and torch training

- Discover compatible CUDA, MPS or CPU devices automatically with
  operation- specific probes while preserving requested precision. Allow
  explicit CPU and indexed CUDA overrides, and move registered data with
  the problem.
- Provide Riemannian SGD/momentum, Adam/AMSGrad, factorwise AdaGrad and
  transactional Armijo training optimizers that preserve parameter
  identity.
- Coalesce sparse embedding gradients and update only touched rows with
  local adaptive clocks; maintain independent state for repeated
  manifold factors.
- Include managed module/minibatch placement, validation, metric
  clipping, averaged gradient accumulation, scheduler hooks and
  resumable data order.
- Support portable parameter/state checkpoints, module parameters and
  buffers, progress metadata and R/torch RNG state. Retain legacy
  checkpoint readers and optimizer-state schema 1 alongside schema 2.

### Diagnostics and documentation

- Provide manifold, gradient, Hessian and residual-adjoint checks, and
  matrix-free Hessian spectrum estimates with residuals and estimation
  limits.
- Include roxygen help and executable examples, fifteen package
  vignettes, coverage ledgers, mathematical notes, analytic fixtures and
  extension tests.
- Add a local pkgdown website with ten worked examples, including six
  public-data applications with figures and independent numerical
  checks. Bundle the attributed Penguins snapshot for builds without
  dataset downloads.
- Include reproducible Pymanopt/SciPy comparisons, memory measurements,
  cross-platform CI and GPU qualification tools. Hardware support
  remains qualified separately from device discovery; local reference
  evidence is CPU based.
