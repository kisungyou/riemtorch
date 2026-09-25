# Competitor evidence and experimental solver audit

The source lock distinguishes inspected documentation from repository revisions
locked for future reproduction. A current HEAD is not represented as the version
that generated stable documentation. External packages are not runtime dependencies.
Compare metrics and representations, not constructor counts. The original geometry
ledger includes aliases; 31 rows are not 31 distinct spaces.

The existing five experimental methods remain experimental after reviewing their
implementations and re-running their numerical fixtures:

* Cubic regularization minimizes a cubic model approximately, initialized by a
  Cauchy step; this is not an exact cubic subproblem solver.
* Particle swarm transports velocities but uses local or aligned displacements;
  it supplies no stationarity certificate.
* Nelder-Mead builds a tangent simplex, uses local centroids and retracts proposals;
  it is not a globally affine simplex construction on curved spaces.
* Stiefel annealing uses left orthogonal proposals and a cooling Metropolis rule;
  it is an optimization heuristic, not a stationary-distribution sampler.
* Grassmann MACG uses elite sampling and shrinkage shape fitting; covariance
  adaptation is representation-specific and provides no global optimum guarantee.

The tests check feasible iterates, objective progress, failure accounting and
absence of fabricated gradient convergence. Cross-package timing comparisons are
separate from these contract tests. Unsupported primitive derivatives remain
explicit, including custom matrix-function double backward.

## Executed upgrade evidence

The final upgrade implements public diagnostics; CG/BB variants and preconditioned
second-order models; variance reduction; robust least squares; compact low-rank
and additional real geometries; complex first-order geometries; adaptive, sparse
and line-search training; and intrinsic constrained/proximal problems. The
installed competitor ledger names metrics, representations, algorithm variants,
evidence and restrictions individually.

Actual external solver execution is limited to Pymanopt 2.2.1 and SciPy 1.13.1
in the upgrade benchmarks, plus the original Riemann reference fixtures. Locked
Git HEADs for other competitors are provenance metadata, not a claim that those
commits were installed or benchmarked. Their official capability documentation
was inspected. The benchmark environment records exact installed versions.

Compact SVD points deliberately use a full small core. This preserves the same
embedded metric while avoiding diagonal-SVD coordinate ambiguity at repeated
singular values. Sparse Adam deliberately uses per-row bias-correction clocks;
that is a documented lazy-update variant rather than a claim of bitwise Geoopt
compatibility. Cyclic proximal point deliberately does not convert a small cycle
displacement into a stationarity claim. The projected robust Gauss–Newton model
is positive semidefinite and explicitly omits loss second derivatives.

The CPU evidence does not qualify CUDA/MPS kernels or Linux/Windows installations.
The supplied platform workflow and GPU runner must produce their own logs before
those statuses change. No package upload, CRAN submission or remote workflow run
was performed.
