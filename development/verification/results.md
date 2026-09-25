# Historical validation results

The current package version is **0.1.0** following the version reset. See
[current version verification](version-0.1.0.md) for its checks. The historical
0.6.0 development version had 1,150 passing expectations and a clean standard
package check; [the upgrade report](upgrade-report.md) preserves that evidence
and its platform limitations. All results below describe their original builds.

The obsolete top-level `riemtorch-validation` scratch directory has been removed.
Archive paths into that directory below record their original locations; those
archives are no longer available there. The archived 0.1.0 build from 2026-09-17
is distinct from the current package using the reset version number.

## riemtorch 0.2.0 — 2026-09-20

* Source build: passed, including nine executed vignette HTML files.
* Fresh staged installation from the source archive: passed.
* Standard `R CMD check`: **0 errors, 0 warnings, 0 notes** (`Status: OK`).
* Installed test suite: **841 passing expectations, 0 failures, 0 warnings,
  0 skips**. See `testthat-0.2.0.Rout`.
* Installed examples, documentation completeness/signatures, rebuilt vignettes,
  package load/unload, namespace and PDF manual checks: all passed.
* Roxygen documentation regeneration: passed. All 33 user-facing help topics
  have executable examples; the package overview is the sole topic without one.
* An installed-package smoke test passed device discovery, registered-data
  optimization with a forced CPU device and the device support ledger.
* The portable device benchmark driver completed its sphere, Stiefel and SPD
  CPU float64 cases against the staged 0.2.0 installation.

Historical source archive location (removed during scratch-directory cleanup):
`../../../../riemtorch-validation/riemtorch_0.2.0.tar.gz`

Archive size: **124,862 bytes**

SHA256:
`baf551b4a386e79a7f56c1b762689e6da177130f47e0789f135e8d18e53832fd`

An integrity comparison confirmed that all 88 package files under `R`, `man`,
`tests`, `vignettes`, `inst/coverage` and `inst/math`, plus the principal
top-level metadata files, agree byte-for-byte with the source archive.

The tested environment is CPU float64 on macOS arm64 with R 4.5.2, torch 0.17.0
and libtorch 2.8.0. CPU float32 has targeted device and solver coverage. CUDA
and MPS were unavailable locally, so their support is runtime-discovered and
conditionally tested rather than claimed as local hardware certification.

Final `R CMD check --as-cran`: **0 errors, 0 warnings, 2 notes**. The notes are:

1. CRAN incoming feasibility identifies this as a **new submission**.
2. The system HTML Tidy binary is too old for optional HTML validation; that
   validation was skipped. PDF/manual generation and all Rd checks passed.

Neither note concerns package implementation, examples, tests or vignettes.
The standard and CRAN-style logs are `R-CMD-check-0.2.0.log` and
`R-CMD-check-as-cran-0.2.0.log`. Online dependency and metadata checks were run
with network access. No CRAN submission was made.

## Archived riemtorch 0.1.0 — 2026-09-17

* Source build: passed, including eight executed vignette HTML files.
* Staged installation from the source archive: passed.
* Standard `R CMD check`: **0 errors, 0 warnings, 0 notes** (`Status: OK`).
* Installed test suite: **653 passing expectations, 0 failures, 0 warnings,
  0 skips**. See `testthat.Rout`.
* Installed examples, documentation completeness/signatures, rebuilt vignettes,
  package load/unload, namespace and PDF manual checks: all passed.
* Roxygen documentation regeneration: passed. All 30 user-facing help topics
  have executable examples; the package overview is the sole topic without one.
* Nine matched-accuracy benchmark runs: all reached the stated objective,
  stationarity and constraint thresholds. See `../benchmarks/results.csv`.

Historical source archive location (removed during scratch-directory cleanup):
`../../../../riemtorch-validation/riemtorch_0.1.0.tar.gz`

SHA256:
`25e7b824397f52f0259dfbf534938489b0716aef8335a909b27f6b7196fbaf10`

The R, man, tests, coverage and vignette source files in this archive agree with
the working source. R CMD build adds only standard DESCRIPTION build metadata.
This package has no compiled extension (`NeedsCompilation: no`); torch/libtorch
is the external numerical runtime and was installed explicitly before testing.

The tested environment is CPU float64 on macOS arm64 with R 4.5.2, torch 0.17.0
and libtorch 2.8.0. Device support beyond this environment remains unverified.

Final `R CMD check --as-cran`: **0 errors, 0 warnings, 2 notes**. The notes are:

1. CRAN incoming feasibility identifies this as a **new submission**.
2. The system HTML Tidy binary is too old for optional HTML validation; that
   validation was skipped. PDF/manual generation and all Rd checks passed.

Neither note concerns package implementation, examples, tests or vignettes.
The full final log is `R-CMD-check-as-cran.log`; online dependency and metadata
checks were run with network access. No CRAN submission was made.
