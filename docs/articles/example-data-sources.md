# Data sources and reproducibility

The [example
gallery](https://www.kisungyou.com/riemtorch/articles/examples.md)
combines small synthetic problems with applications using public
observations. All data needed for the examples are available locally:
examples use R’s `datasets` package and bundled snapshots from Palmer
Penguins and Riemann. Building the site does not install a data package
or download a dataset. Riemann supplies observations for three examples;
their numerical optimization uses riemtorch and R torch.

## Data inventory

| Dataset | Source and measurements | Missing data and preparation |
|:---|:---|:---|
| [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/reference/penguins.html) | 344 penguins; bill length/depth and flipper length in millimeters, mass in grams. | Complete cases of these four measurements only, leaving 342 rows. Standardize the measurements. Species is used only to color a plot. Missing sex does not remove a row. |
| [Stack loss](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/stackloss.html) | 21 days of ammonia-plant operation: air flow, water temperature, acid concentration, and stack loss. | No missing values. Standardize the predictors; document the loss threshold in response units. These are observational measurements, not known erroneous values. |
| [Air quality](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/airquality.html) | 153 days in New York in 1973. Ozone in ppb, solar radiation in Langleys, wind in mph, temperature in degrees Fahrenheit. | Ozone has 37 missing values and solar radiation has 7. Exclude Month/Day from the matrix. Separate naturally missing values from a seeded holdout of observed entries; derive means and scales from training entries. |
| [European stock indices](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/EuStockMarkets.html) | 1,860 business-time observations of DAX, SMI, CAC, and FTSE indices, supplied to R by Erste Bank AG. | Convert to percentage log returns. Use 15 complete, nonoverlapping blocks of 120 returns; omit the final 59 returns. Apply 5% covariance shrinkage before geometric averaging. |
| [Black cherry trees](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/trees.html) | 31 trees: diameter in inches, height in feet, and timber volume in cubic feet. | No missing values. The column named `Girth` actually measures diameter. Log-transform the three variables; nonnegative exponents summing to three are a modeling assumption. |
| [Motor Trend cars](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/mtcars.html) | 32 cars. Response: miles per US gallon. Predictors: displacement (cubic inches), horsepower, rear-axle ratio, weight (1,000 pounds), and quarter-mile time (seconds). | No missing values in the selected columns. Center `mpg`, standardize the five predictors, and recover the intercept from the centering transformation. No test-set prediction claim is made. |
| [US cities](https://www.kisungyou.com/riemtorch/articles/example-cities-center.md) ([`Riemann::cities`](https://kisungyou.com/Riemann/reference/cities.html)) | 60 historical city locations and population counts, described as January 2006 data; latitude/longitude in degrees and unit Cartesian coordinates. Derived by Riemann from [`maps::us.cities`](https://rdrr.io/pkg/maps/man/us.cities.html). | No missing locations or populations. Compare equal-city weights with population weights normalized to sum to one. The selected cities are not the entire US population. |
| [Hand landmarks](https://www.kisungyou.com/riemtorch/articles/example-hands-alignment.md) ([`Riemann::hands`](https://kisungyou.com/Riemann/reference/hands.html)) | 40 left-hand configurations from four people, with 56 corresponding planar landmarks per configuration. Physical coordinate units are unspecified. | No missing landmarks. Use the first six poses of person 1; remove translation and positive scale. Additional rotations are explicitly introduced for the demonstration and removed by SO(2) alignment. Reflections are excluded. |
| [MEG covariance features](https://www.kisungyou.com/riemtorch/articles/example-erp-covariances.md) ([`Riemann::ERP`](https://kisungyou.com/Riemann/reference/ERP.html)) | 216 matrices of size 32 by 32, from one participant’s epochs after supervised xDAWN filtering and prototype augmentation. Labels LA, LV, RA, RV have counts 57, 57, 53, 49. | No missing matrices or labels. Apply a common positive rescaling to the very small original values, retaining every matrix. The 32 coordinates are derived features, not original sensor channels. |

The foundation PCA example uses R’s `iris` measurements, the
hand-alignment example uses observed landmarks, and the other three
foundation examples use fixed synthetic matrices or seeded simulated
data. Consult
[`help("iris", package = "datasets")`](https://rdrr.io/r/datasets/iris.html)
and each R dataset’s help page for the original bibliographic
references. The examples use the copies distributed with R rather than
fetching another version from a website.

## The bundled penguin snapshot

- [Download
  penguins.csv](https://www.kisungyou.com/riemtorch/articles/data/penguins.csv).
- [Download the attribution and CC0
  notice](https://www.kisungyou.com/riemtorch/articles/data/NOTICE-penguins.txt).
- [Download the extraction
  script](https://www.kisungyou.com/riemtorch/articles/data/extract-penguins.R).

The CSV is an unchanged copy of `extdata/penguins.csv` from
`palmerpenguins` **0.1.1**. Its SHA-256 checksum is:

``` text
f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93
```

The observations were collected by Kristen B. Gorman and the Palmer
Station Antarctica Long Term Ecological Research program, and curated
for R by Allison Marie Horst, Alison Presmanes Hill, and Kristen B.
Gorman. The upstream project distributes these data under [CC0
1.0](https://allisonhorst.github.io/palmerpenguins/LICENSE.html). No
penguin artwork is redistributed here.

For attribution, cite Horst, Hill, and Gorman (2020),
[palmerpenguins](https://doi.org/10.5281/zenodo.3960218), and Gorman,
Williams, and Fraser (2014), [the original ecological
study](https://doi.org/10.1371/journal.pone.0090081). The notice links
the original species-level data releases as well.

The five R datasets are accessed through `datasets::` and are not copied
into riemtorch. R’s distribution terms and the source references in
their help pages remain applicable; the penguin dataset’s CC0 notice
does not cover those separate datasets.

To recreate the penguin snapshot, obtain `palmerpenguins` version 0.1.1
and run the extraction script from the package project. It checks the
package version and file checksum before copying the CSV. This is a
maintenance operation; neither that package nor the extraction script is
needed to build the site.

``` r

source("vignettes/articles/data/extract-penguins.R")
```

## The bundled Riemann snapshots

The three complete dataset objects come from **Riemann 0.1.7**, commit
[`e847692576f8b250c8140e1343b145f72caa379f`](https://github.com/kisungyou/Riemann/tree/e847692576f8b250c8140e1343b145f72caa379f).
The original `.rda` files were loaded and saved as version-2 RDS files
without changing values, labels, ordering, or attributes. No
preprocessing is hidden in these snapshots; each article shows its own
transformations.

- [Download
  cities](https://www.kisungyou.com/riemtorch/articles/data/riemann-cities.rds),
  [hands](https://www.kisungyou.com/riemtorch/articles/data/riemann-hands.rds),
  and
  [ERP](https://www.kisungyou.com/riemtorch/articles/data/riemann-ERP.rds).
- [Download source and snapshot
  checksums](https://www.kisungyou.com/riemtorch/articles/data/riemann-snapshots.csv).
- [Download attribution and license
  notices](https://www.kisungyou.com/riemtorch/articles/data/NOTICE-Riemann.txt).
- [Download the extraction
  script](https://www.kisungyou.com/riemtorch/articles/data/extract-riemann.R).

The pinned Riemann release declares **MIT + file LICENSE**, with
copyright 2020 Kisung You; the bundled notice includes that license and
the original data-source credits. This describes the pinned release, not
a later development version. The separate Palmer Penguins CC0 notice
does not apply to these Riemann snapshots.

The hand data are attributed to Stegmann and Gomez (2002), *A Brief
Introduction to Statistical Shape Analysis*, Technical University of
Denmark. The source help does not identify an additional public-domain
dedication for those data. The alignment example keeps handedness;
Riemann’s `wrap.landmark()` uses an orthogonal quotient that also
identifies reflections, so the two conventions must not be compared as
if they were identical.

For ERP, the provenance follows the reconstruction documented in
Riemann’s September 2026 data-source correction and the [historical
pyRiemann
recipe](https://github.com/pyRiemann/pyRiemann/blob/176e766f540bd4c7846f38573165fc3d27fc69ca/examples/ERP/plot_embedding_EEG.py).
The recipe selects **MEG and excludes EEG**. Its separate 72-epoch
training split supplies the supervised spatial filters; the packaged
matrices describe the remaining 216 epochs. The first 16 coordinates are
class-prototype features and the last 16 are filtered-trial features.
These are repeated measurements from **one participant**, not 216
independent participants. The older package help’s
EEG/channel/participant wording is superseded by this reconstruction;
the stored numerical data are unchanged.

The underlying MNE sample dataset’s [OpenNeuro record, ds000248 version
1.2.4](https://doi.org/10.18112/openneuro.ds000248.v1.2.4), declares
CC0. Cite Gramfort et al.
[(2013)](https://doi.org/10.3389/fnins.2013.00267) and
[(2014)](https://doi.org/10.1016/j.neuroimage.2013.10.027) for MNE. The
notice links the pinned metadata and keeps these source terms distinct
from the package license. Neither the original recordings nor upstream
code are redistributed in riemtorch.

To recreate the snapshots, obtain a local checkout of the pinned Riemann
source and run the following maintenance commands. The script checks the
source-file hashes and verifies that the extracted R objects are
unchanged. Riemann need not be installed; neither this checkout nor the
script is needed for an ordinary website build.

``` r

source("vignettes/articles/data/extract-riemann.R")
extract_riemann_snapshots("../Riemann")
```

## Reproduce the examples

Open `riemtorch.Rproj` in RStudio and run
[`pkgdown::build_site()`](https://pkgdown.r-lib.org/reference/build_site.html).
The new applications use CPU float64, one torch thread, and explicit
R/torch seeds. Each article executes numerical assertions against an
independent calculation or a clearly stated geometric condition. A
failed assertion stops the build.

The gallery’s [recorded
runtimes](https://www.kisungyou.com/riemtorch/articles/data/example-runtimes.csv)
measure complete standalone article rendering, including initialization,
plots, and Pandoc, on the documented reference machine. They are not
solver benchmarks. Fresh processes are used for two repetitions of each
new application; deterministic numeric results are compared with
tolerance `1e-8` after scaling by `pmax(1, abs(result))`. Wall-clock
times and hardware metadata are excluded from that comparison.

No dataset network connection is needed. pkgdown itself may still
retrieve its standard theme assets or package metadata; this is separate
from data loading. The articles show explicit `device = "auto"`,
`device = "cpu"`, and indexed CUDA alternatives. Only CPU results are
used for the reproducibility checks.

## What the checks establish

PCA is checked by its subspace projector, since basis vectors are not
unique. Regression and proximal solutions are compared with independent
base-R calculations. Constrained fits report feasibility and optimality
residuals. SPD examples verify positive definiteness and the relevant
metric conditions.

For air quality, only deliberately withheld observations have known
answers. The example reports its errors alongside a column-mean baseline
even if it does not improve on that baseline. Nothing in the example
establishes the accuracy of predictions for measurements that were
originally missing.

Small illustrative datasets support learning and numerical verification.
They do not establish general predictive accuracy, causal conclusions,
or large-scale performance.
