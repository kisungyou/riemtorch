# Maintain the riemtorch documentation site

Open `riemtorch.Rproj` in RStudio. From its console, run:

```r
pkgdown::build_site()
```

The standard build installs the current source in a temporary library, executes
reference examples and articles in fresh R processes, and writes the site to
`docs/`. An interactive build opens the result in a browser. To reopen it, use
`pkgdown::preview_site()` or open `docs/index.html`.

## One-time prerequisites

```r
install.packages(c("torch", "pkgdown", "knitr", "rmarkdown"))
torch::install_torch()
```

The first build may download and cache pkgdown's standard theme assets.
RStudio supplies Pandoc. Outside RStudio, install Pandoc or set `RSTUDIO_PANDOC`
to the directory containing its executable. The project does not set library
paths or machine-specific environment variables. A working torch runtime is
required to execute the tutorials; the build does not install it automatically.

## Files to edit

- `pkgdown/_pkgdown.yml`: navigation, article groups, reference groups, theme.
- `pkgdown/index.md`: the website homepage; `README.md` remains the package README.
- `pkgdown/extra.css`: the homepage layout, alongside the locally bundled theme assets.
- `vignettes/articles/*.Rmd`: website-only getting-started and worked examples.
- `vignettes/*.Rmd`: the existing package vignettes, also shown on the website.
- `R/*.R`: roxygen documentation for function pages. After changing it, run
  `roxygen2::roxygenise()` before rebuilding the site.
- `NEWS.md`: the changelog.

The ten worked examples use seeded, small CPU problems and base R plots.
They execute during an ordinary site build. The installed package vignettes
retain their existing behavior, including automatic device selection where
demonstrated. Reference examples are enabled.

Six public-data applications use five built-in R datasets and the bundled
`vignettes/articles/data/penguins.csv` snapshot. Its attribution, CC0 notice,
checksum, and extraction instructions are supplied alongside it and on the
data-source page. `palmerpenguins` is needed only to recreate the snapshot,
not to build the website. No dataset is downloaded during a build.

The gallery reads measured article render times from
`vignettes/articles/data/example-runtimes.csv`. To refresh these measurements
and compare independent repeated renders, run
`Rscript development/documentation/verify-public-examples.R` from the project
directory. That development script writes its results under
`development/documentation/public-data/` and updates the gallery's timing CSV.
Rebuild the site after refreshing the measurements. Then run
`python3 development/documentation/validate-public-site.py` to check generated
local links, fragment targets, figure descriptions, and downloadable files.
The public-data verification report and logs are in
`development/documentation/public-data/verification.md`.

The generated `docs/` directory and website-only sources are excluded from
package source archives. `docs/` is ignored by Git because it is reproducible
build output. The package's installed help and fifteen package vignettes remain
part of standard package builds.

## Local preview and later publication

All internal navigation is relative. The homepage, examples, and reference work
when opening `docs/index.html` directly. Browser search loads its index via HTTP;
use `pkgdown::preview_site()` for a full local preview including search.

There is deliberately no deployment workflow, analytics, or invented public
repository URL. When a hosting location is chosen, add `url:` to
`pkgdown/_pkgdown.yml` and configure the desired deployment separately. A missing
canonical URL is expected for this local-only site. The build's initial sitrep
reports `url is missing`, but the site still builds successfully. The separate
`pkgdown::check_pkgdown()` publication-readiness check also requires this URL;
use it after choosing the public address.
