# riemtorch competitor upgrade: historical implementation and verification

This report records the internal development version **riemtorch 0.6.0** and its
0.3, 0.4, 0.5 and 0.6 implementation stages. The current package was subsequently
renumbered to **0.1.0**; see [current version verification](version-0.1.0.md).
Version labels, test results, archive identities and hashes below describe the
original builds. The obsolete top-level `riemtorch-validation` scratch directory
has been removed; paths into it in preserved check logs are historical. No CRAN
submission or remote deployment was performed.

## Implemented scope

- **0.3:** competitor capability/source ledger; manifold, gradient, Hessian and
  residual-adjoint diagnostics; matrix-free spectral estimates with residuals;
  detached streaming values/gradients/HVPs; explicit primitive capabilities;
  operation-specific cached device probes and execution metadata.
- **0.4:** fused value/gradient and preconditioner callbacks; work/time budgets,
  detached iteration/final callbacks; CG/BB variants; preconditioned tCG;
  SVRG/SRG; sequential multistart; robust least squares; power/scaled, positive,
  affine, orthogonal and positive doubly stochastic geometry; compact fixed rank;
  exact frame maps and canonical-Stiefel Hessian conversion.
- **0.5:** AMSGrad, AdaGrad and transactional Armijo training optimizers;
  independent power-factor state; coalesced lazy sparse updates; validation,
  clipping, accumulation, scheduler and epoch/data-order hooks; versioned
  checkpoint layouts with legacy readers; six selected complex geometries.
- **0.6:** equality/inequality problems and augmented Lagrangian with multipliers
  and KKT diagnostics; intrinsic proximal gradient and cyclic proximal point;
  independent numerical comparisons, memory measurements and qualification tools.

Existing positional arguments and defaults remain intact. Standalone low-level
Torch optimizers preserve parameter identity. Managed runs resolve one device,
preserve requested precision, accept explicit CPU/CUDA overrides and do not
silently change devices after computation begins.

## Release checks

| Version | Passing expectations | Standard package check |
|---|---:|---|
| 0.3.0 | 864 | 0 errors, 0 warnings, 0 notes |
| 0.4.0 | 940 | 0 errors, 0 warnings, 0 notes |
| 0.5.0 | 1,018 | 0 errors, 0 warnings, 0 notes |
| 0.6.0 | 1,150 | 0 errors, 0 warnings, 0 notes |

The 0.6 archive installs successfully into a clean staged library. Its examples,
15 vignettes, namespace/load/unload checks, dependency checks, documentation
signatures and PDF manual all pass. There are 76 exported functions and 46 help
pages; 45 pages contain executable examples. The package overview is the sole
page without an example. The package itself has no compiled extension and uses
R torch as its only installed numerical backend.

Final CRAN-style check: **0 errors, 0 warnings, 3 notes**. The notes are
administrative/environmental: new-submission status; an unreachable service for
verifying the system clock; and the system's old HTML Tidy, which prevents
optional HTML validation. None reports a package implementation, test, example,
vignette or manual-generation failure. The exact messages are saved in
`releases/0.6.0-check-as-cran.log`.

Source archive: `/Users/kisung/Desktop/develop/R-devel/riemtorch_0.6.0.tar.gz`.
SHA256: `eb5c3be79322f8228e3fb2c3f3f1676ab426239be469f80751d2b9e3c6d7a7b2`.

An integrity comparison found all **119 source-bearing R, Rd, test, vignette,
coverage, mathematical-note and top-level files byte-identical** between the
working package and archive. R CMD build adds normal build metadata and rendered
vignettes. `../releases/manifest.csv` records all four archives and hashes.

## Independent evidence

The eight-family suite completed **48 runs**: three shared seeded instances for
each of riemtorch and Pymanopt/SciPy. Cases cover eigenvector/PCA objectives,
Procrustes, matrix completion, SPD log coordinates, Huber least squares,
hyperbolic embeddings, phase synchronization and constrained optimization.
All runs meet the stated objective-agreement, gradient/KKT and feasibility
thresholds of 1e-6. The largest objective disagreement is about 5.92e-7 in a
constrained case whose feasibility error is within tolerance. Unconstrained
objective disagreement is below 1e-10. Algorithms differ and problems are small;
these results do not establish package-wide speed parity or global convergence.
Failures, evaluation counts, timings, transfer costs, shared inputs and sampled
RSS are retained in `../benchmarks/upgrade/evidence/`. Executed references are
Pymanopt 2.2.1 and SciPy 1.13.1; other competitor documentation/revisions are
recorded separately without pretending those libraries were executed. Python
and competitor packages are isolated development tools, not R package dependencies.
Each earlier release also has nine passed analytic-accuracy CPU benchmark runs
in `../releases/<version>/`.

With chunk size 64 and 256 parameters, increasing finite-sum terms from 1,024 to
65,536 leaves sampled incremental gradient/HVP RSS around 14–18 MB. Diagnostic
values reach about 44 MB; full differentiable value graphs grow to about 171 MB.
Graph-retaining values are not advertised as bounded-memory evaluations. RSS
includes runtime/allocator effects and sampling can miss brief peaks.

## Qualification limits

The executed platform is macOS arm64, R 4.5.2, torch 0.17.0, libtorch 2.8.0.
The complete suite runs on CPU float64, with targeted float32/complex64 tests.
The CPU qualification runner passes automatic selection, explicit CPU override,
both float64 and float32, checkpoint loading and continuation. CPU/GPU
agreement remains part of the unexecuted accelerator qualification.

**CUDA, MPS, Linux and Windows remain awaiting actual hardware/platform
validation.** The reproducible GPU runner tests automatic and indexed CUDA,
forced CPU, numerical agreement, synchronized timings, transfers and checkpoint
migration. The CI workflow covers Linux/macOS/Windows. Neither tool converts an
unexecuted configuration into a verified support claim.

The original five experimental methods keep that label: approximate cubic
regularization, transported particle swarm, local Nelder–Mead, Stiefel annealing
and Grassmann MACG. Complex frame/unitary automatic Hessians remain unsupported.
Cut loci and numerical rank/positivity boundaries are explicit. Spectral Hessian
estimates report residuals without certifying unseen extrema. User proximal maps
must solve their documented manifold-distance subproblem; a cycle residual alone
does not establish stationarity. A legacy shared power variance is replicated
when migrating to independent factor state, so subsequent adaptation follows
the new layout rather than promising bitwise replay of the former algorithm.

Distributed training, automatic mixed precision, sampling, statistical estimator
catalogs, full-solver differentiation and specialized symplectic/tensor manifolds
remain outside the agreed scope.
