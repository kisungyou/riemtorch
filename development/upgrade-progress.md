# Competitor upgrade implementation

Baseline: riemtorch 0.2.0, 841 passing expectations. Preserve all existing APIs.

## Stages

- [x] 0.3: diagnostics, explicit primitive metadata, streaming evaluation,
  operation-specific devices, competitor inventory and numerical audit.
- [x] 0.4: solver controls, preconditioning, variance reduction, robust losses,
  repeated/scaled and additional real geometries, compact fixed rank, maps.
- [x] 0.5: adaptive/sparse/line-search optimizers, managed training controls,
  complex manifolds, checkpoint compatibility.
- [x] 0.6: constrained and composite problems, augmented Lagrangian and proximal
  solvers, documentation, benchmarks and qualification tooling.

Each stage requires regression and new numerical tests, regenerated roxygen,
source build, installation, examples, vignettes and a recorded package check.
External OS/GPU runs require an available host; do not substitute skipped tests
for hardware evidence. No package publication is part of this task.

0.3.0: standard package check Status OK; 864 passing expectations.

0.4.0: standard package check Status OK, including all examples and vignettes.

0.5.0: standard package check Status OK, including the complex and sparse training vignettes.

0.6.0: 1,150 passing expectations; clean standard check; CRAN-style check has no errors/warnings and three explained environment/administrative notes. See verification/upgrade-report.md.
