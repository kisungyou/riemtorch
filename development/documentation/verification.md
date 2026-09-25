# Local pkgdown site verification

Verified on 2026-09-25 with R 4.5.2, pkgdown 2.2.1, torch 0.17.0,
and the Pandoc supplied with RStudio on macOS arm64.

## Result

The ordinary `pkgdown::build_site()` command completed from the directory
containing `riemtorch.Rproj`, using the default R library. No custom build
wrapper, project startup file, or library-path override is required.
The shell verification supplied `RSTUDIO_PANDOC` because this terminal does
not inherit RStudio's Pandoc discovery; RStudio supplies it automatically.
The build installed the current package into pkgdown's temporary library and
ran its reference examples and articles in fresh processes.

The site contains 72 HTML pages, including 46 help topics and 20 articles:
the 15 package vignettes, a getting-started guide, and four new worked examples.
Seven plots across the new articles have descriptive alternative text.

The worked examples cover sphere PCA, robust regression, an affine-invariant
SPD midpoint, and compact fixed-rank matrix completion. They execute on CPU
float64 with explicit seeds. Reference comparisons include base R eigenvectors,
least-squares and Huber fits, and an independently computed SPD midpoint.
The matrix-completion example also reports errors on held-out entries.

## Checks performed

- Two complete standard site builds passed. The homepage was subsequently
  regenerated after correcting two article-index anchor links.
- All 2,667 local HTML links, fragment targets, scripts, stylesheets, images,
  and CSS asset references resolved across the 72 pages.
- The local HTTP preview displayed the homepage and example figures correctly.
  Searching for "Huber" returned the worked example and the least-squares
  function reference.
- `R CMD build` passed, rebuilding all 15 installed vignettes.
- `R CMD check --no-manual` finished with `Status: OK`: zero check errors,
  warnings, or notes. This includes tests, help examples, and vignette rebuilding.
  Repository-index requests during this check could not access the network in
  the restricted session; installed dependencies were available and the
  dependency check passed.
- The source archive contains no generated website, pkgdown configuration,
  or website-only article sources; all 15 installed vignette HTML files remain.

Logs and the link-check result are stored beside this report. The existing
release archives outside the package directory were not replaced.

## Expected metadata notice

A public `url` has intentionally not been chosen. pkgdown's initial sitrep
therefore reports `url is missing`; this does not prevent the local build.
Its separate publication-readiness check, `pkgdown::check_pkgdown()`, requires
the public URL and should be run after that address is selected. No deployment
workflow or public hosting configuration has been created.

## Rebuild and preview

Open `riemtorch.Rproj` in RStudio, then run `pkgdown::build_site()`.
The result is `docs/index.html`. Use `pkgdown::preview_site()` to reopen an
HTTP preview with working local search. The site's maintenance instructions
are in `pkgdown/README.md`.
