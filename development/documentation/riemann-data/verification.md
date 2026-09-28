# Riemann-data documentation verification

Verified on 2026-09-25 with riemtorch 0.1.0, R 4.5.2, torch 0.17.0,
pkgdown 2.2.1, RStudio Pandoc 3.8.3, and macOS arm64.

## Gallery placement

The existing gallery now contains thirteen worked examples. Hand alignment
joins Foundations (five examples); cities and ERP join Public-data applications
(six examples). Constraints and sparsity retains its two examples. There is no
separate Riemann gallery group. The homepage retains three columns at desktop
width, and all gallery groups retain two columns.

## Data and reproducibility

The three complete R objects are bundled as compressed RDS snapshots from
Riemann 0.1.7, commit `e847692576f8b250c8140e1343b145f72caa379f`.
The extraction script checks source-file fingerprints and object identity.
Re-extraction reproduced the bundled files byte for byte. The source and
generated-download SHA-256 checks pass for all three snapshots.

Attribution, original sources, licensing, and extraction instructions appear on
the data-source page and in the downloadable notice. ERP is described as 216
transformed MEG epochs from one participant, including prototype and trial
features; it is not described as raw EEG from 216 participants.

Each new article ran twice independently in fresh R sessions with fixed seeds,
CPU float64, and one torch thread. The usual base-R download entry point was
blocked during these renders. Data loading uses the bundled snapshots only;
Riemann need not be installed. All recorded numerical results matched exactly
between runs, within the prescribed relative tolerance of 1e-8.

| Article | Run 1 (seconds) | Run 2 (seconds) | Mean (seconds) |
|:--|--:|--:|--:|
| Hands | 4.969 | 3.882 | 4.426 |
| Cities | 2.139 | 2.187 | 2.163 |
| ERP | 2.433 | 2.271 | 2.352 |

These times include process startup, complete rendering, plots, and independent
checks. The combined means total 8.941 seconds; each article is below the
30-second target. They are reference-machine measurements, not guarantees.

## Independent numerical checks

- Hands: all five rotation fits converged. The maximum rotation discrepancy
  against determinant-corrected base-R SVD was 1.13e-7; the maximum objective
  discrepancy was 9.38e-15. Orthogonality, positive determinant, and preprocessing
  invariance checks passed.
- Cities: both spherical fits converged. The independent population-weighted
  and equal-city gradient norms were 1.50e-9 and 5.06e-10. Base-R two-angle
  optimization, unit-norm, objective-agreement, and inactive-clamping checks
  passed.
- ERP: the exact reduced objective uses all 216 covariance matrices. The
  log-Euclidean solution differed from independent base-R eigendecompositions
  by 8.66e-15 relatively. Stationarity was 3.12e-14, and full-versus-reduced
  objective error was 5.33e-15. Positive definiteness and common-scale distance
  invariance passed.

See `timings.csv`, `numeric-results.rds`, `environment.txt`, and the standalone
HTML outputs and logs in `run-1/` and `run-2/`.

## Website checks

Ordinary `pkgdown::build_site()` completed successfully from the project root,
including temporary package installation, reference examples, and articles.
RStudio's Pandoc directory was supplied to the terminal; no project startup
hook or custom build wrapper was required. A final introductory-link wording
change was rebuilt separately. `pkgdown::check_pkgdown()` reported no problems.

The local validator checked 131 HTML files (including redirects), 2,917 local
links and assets, 614 fragment links, and 48 CSS asset references. All checks
passed, including ten downloadable resources and descriptive alternative text
for all four new figures. Details are in `site-check.json`.

Browser inspection confirmed the existing gallery groups, two-column panels,
nine homepage application links in three desktop columns, new thumbnails, and
local article navigation. Search indexes include the new articles. pkgdown's
existing search behavior uses canonical public URLs: selecting an unpublished
article from search leads to its future public address, so local inspection
of new pages should use the gallery links until publication.

This change adds website-only articles, snapshots, and development evidence.
It changes no exported functions, numerical implementation, installed vignette,
or package dependency. The existing package exclusions cover these additions.
No commit, push, or publication was performed.

## Repeat

Open `riemtorch.Rproj` in RStudio and run `pkgdown::build_site()`.
The commands in `pkgdown/README.md` rerun the three independent examples and
refresh their timing rows, then validate the generated site. Existing timing
rows for the other ten examples are preserved during a targeted run.
