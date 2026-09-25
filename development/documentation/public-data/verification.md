# Public-data gallery verification

This historical report predates the package version reset to 0.1.0. Its recorded
versions, measurements and archive identity are preserved. See
[current version verification](../../verification/version-0.1.0.md) for the
renumbered package.

Verified on 2026-09-25 with R 4.5.2, riemtorch 0.6.0, torch 0.17.0,
pkgdown 2.2.1, RStudio Pandoc 3.8.3, and macOS arm64.

## Delivered

Six public-data applications accompany the four existing foundation examples.
The gallery groups all ten by application and reports dataset, geometry,
difficulty, and measured CPU render time. The source/reproducibility guide
includes provenance, units, missingness, preprocessing, citations, and the
Penguins download, notice, checksum, and extraction instructions.

All application calculations run with fixed R and torch seeds, CPU float64,
and one torch thread. Automatic, explicit CPU, and indexed CUDA options are
shown without executing accelerator alternatives. No exported functions,
package numerical dependencies, or installed vignette behavior were changed.

## Execution and timing

Every worked example was rendered independently twice in fresh R processes.
All six public-data result vectors matched exactly: maximum scaled numerical
difference was zero, against the documented tolerance of 1e-8. The base-R
`download.file()` entry point was blocked during these standalone renders;
data loading used only the bundled CSV and `datasets::` objects.

Final complete standalone render times, including startup, independent checks,
figures, and HTML rendering:

| Application | Run 1 (seconds) | Run 2 (seconds) | Mean (seconds) |
|:--|--:|--:|--:|
| Penguin subspace | 10.836 | 10.536 | 10.686 |
| Industrial robust regression | 8.346 | 8.877 | 8.611 |
| Air-quality completion | 8.750 | 9.820 | 9.285 |
| Covariance means | 9.545 | 9.031 | 9.288 |
| Constrained tree model | 12.040 | 11.550 | 11.795 |
| Proximal sparse regression | 17.788 | 18.021 | 17.904 |

The six applications together took 67.305 and 67.835 seconds.
All were below the 30-second per-article target. Building these six final
articles through pkgdown took 87.467 seconds, below the three-minute added
application-build target. These are reference-machine observations, not runtime
guarantees. The initial complete site build, including all older guides and
reference examples, took 410.824 seconds.

## Independent numerical evidence

- Penguins: projector error against `prcomp()` was 4.49e-8; the subspace captured
  0.881568 of standardized variance. Rotating the basis changed the objective
  by only 1.11e-16. The deliberately short SVRG comparison exhausted its step
  budget and is explicitly reported as unconverged.
- Stack loss: quadratic coefficients passed the `lm()` comparison. Huber
  coefficients differed from the independently minimized base-R objective by
  1.73e-8; the independent Huber gradient norm was 6.02e-8.
- Air quality: training/holdout disjointness, at least two training entries per
  row, training-only scaling, and dense/compact objective agreement passed.
  Standardized holdout RMSE was 0.9740 versus 0.8641 for column means. This
  worse aggregate prediction result is retained and explained; outperforming
  the baseline is not an acceptance requirement.
- Covariances: the independent spectral log-Euclidean mean differed by
  1.19e-15. Affine-invariant stationarity was 8.84e-9. Both means and all
  shrinkage covariance blocks passed positive-definiteness checks.
- Trees: the reduced one-dimensional reference differed by 4.76e-7 in
  coefficients. Feasibility was 2.24e-8 and stationarity 5.16e-8;
  complementarity and dual-feasibility residuals were zero.
- Sparse regression: all five penalties passed the independent coordinate
  descent and Lasso optimality checks. The largest coefficient discrepancy
  was 9.97e-5 and the largest KKT residual was 7.97e-6.

Full deterministic outputs are in `numeric-metrics.csv` and
`numeric-results.rds`; source fingerprints are in `source-checksums.json`.
Individual render output and logs are in `run-1/` and `run-2/`.

## Website and package checks

The ordinary `pkgdown::build_site()` completed from the RStudio project root,
installing the source into pkgdown's temporary library and executing reference
examples and articles. The terminal supplied RStudio's Pandoc path; no custom
wrapper, startup file, or library-path override was needed. After plot layout
adjustments and final timing measurements, all six applications, the gallery,
and the data-source guide were regenerated, along with search and LLM indexes.

The generated site has 80 HTML pages, including 28 articles and 46 help topics.
The local validator checked 2,783 local links/assets, including 593 HTML
fragments and 48 CSS references. All nine new figures have descriptive
alternative text and were visually inspected. The missingness legend,
covariance value annotations, and tree plot aspect ratio were corrected during
that inspection. Navigation and search were also exercised in the browser;
searching for Penguins opened the new subspace article.

All four downloadable files returned HTTP 200 from the local preview and
matched the generated files. The 15,241-byte Penguins CSV matches its source
and pinned SHA-256:

`f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93`

`R CMD build` passed. Standard `R CMD check`, including the PDF manual,
finished with **Status: OK**, with zero check errors, warnings, or notes.
Tests, help examples, installed vignettes, and vignette rebuilding passed.
Repository-index requests were unavailable in the restricted check environment;
installed dependencies were present, and the dependency check passed. Those
connection messages did not become check warnings or notes.

Archive inspection confirmed that website sources, data, generated pages,
pkgdown configuration, and development evidence stay outside the installed
package. All 15 installed HTML vignettes remain. The checked source archive is
`/tmp/riemtorch-public-package/riemtorch_0.6.0.tar.gz`; earlier release archives
were not replaced.

## Reproduce locally

Open `riemtorch.Rproj` in RStudio and run `pkgdown::build_site()`.
Use `pkgdown::preview_site()` for a local HTTP preview with working search.
No dataset download or installation of `palmerpenguins` is needed. pkgdown
may still retrieve ordinary theme assets or package metadata.

To refresh the independent measurements, run
`Rscript development/documentation/verify-public-examples.R` from the project
root with Pandoc available. Rebuild the site, then run
`python3 development/documentation/validate-public-site.py` to validate local
links, fragments, assets, figure descriptions, and data files.

The missing public-URL notice remains intentional. A deployment address has
not been selected, and no publication was performed.
