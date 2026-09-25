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
- `docs/.nojekyll`: tells GitHub Pages to serve the generated site without a
  Jekyll build. Ordinary site rebuilds preserve this committed file.
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
package source archives. Commit `docs/` to Git after rebuilding: GitHub Pages
will publish that directory from `main`. The package's installed help and
fifteen package vignettes remain part of standard package builds.

## Local preview

All internal navigation is relative. The homepage, examples, and reference work
when opening `docs/index.html` directly. Browser search loads its index via HTTP;
use `pkgdown::preview_site()` for a full local preview including search.

The canonical URL in `pkgdown/_pkgdown.yml` is
`https://www.kisungyou.com/riemtorch/`. This supplies publication metadata while
preserving local previews. Setting this URL does not publish the website.

## First publication on GitHub Pages

The following steps are the maintainer's publication procedure. Preparing the
local files does not commit, push, or enable GitHub Pages; Pages is not yet
enabled for this repository at the time these instructions were prepared.

1. Open `riemtorch.Rproj` and run `pkgdown::build_site()` in RStudio. Check the
   local preview and confirm that `docs/index.html` and `docs/.nojekyll` exist.
   The empty `.nojekyll` file is included in the repository. If recreating a
   deleted `docs/` directory, run `file.create("docs/.nojekyll")` after the build.
2. Review, commit, and push the source changes and complete generated `docs/`
   directory to the repository's `main` branch.
3. Open [Settings > Pages](https://github.com/kisungyou/riemtorch/settings/pages).
   Under **Build and deployment**, choose **Deploy from a branch**, then branch
   **main** and folder **/docs**, and click **Save**.
4. Leave the project's **Custom domain** field empty. The account's existing
   `kisungyou.github.io` site already uses `www.kisungyou.com`, which project
   sites inherit. Do not add a `CNAME` file or change DNS for this project.
   Ensure **Enforce HTTPS** is enabled when the option is available.
5. Wait for the Pages deployment to finish in the repository's **Actions** tab,
   then check the homepage, example gallery, search, and a data download at
   [the public site](https://www.kisungyou.com/riemtorch/). Set the repository's
   **About > Website** field to this address.

The requested address `https://kisungyou.com/riemtorch/` redirects to
`https://www.kisungyou.com/riemtorch/`, just as the existing Riemann and
T4transport sites do. No change to the account site's custom domain is needed.

For later updates, rebuild locally, review the results, and commit and push the
updated source and `docs/` files. GitHub publishes the committed output; no R or
torch installation is needed on its Pages server. Keep `docs/.nojekyll` in the
published directory and avoid adding a second workflow that deploys a different
copy of the site.

See GitHub's instructions for [publishing from a branch](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
and [custom-domain inheritance](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages),
and pkgdown's [site configuration reference](https://pkgdown.r-lib.org/reference/build_site.html).
