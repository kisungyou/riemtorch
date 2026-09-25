# Version 0.1.0 and validation-folder cleanup

The package version is now **0.1.0**. DESCRIPTION, source-archive instructions,
release notes, and current maintenance reports have been updated. NEWS.md
consolidates the implemented features into one initial-release entry.

Removed `/Users/kisung/Desktop/develop/riemtorch-validation`, the obsolete
top-level scratch directory containing old checks and reference installations.
No active package, test, documentation-build, or development script depended
on that directory. References remaining in old check logs are historical.

The new source archive is
`/Users/kisung/Desktop/develop/R-devel/riemtorch_0.1.0.tar.gz`.
SHA-256: `65dc817b35f1df4cddde93fd5e063a4356c5dc15d6a56e6ce467cd452d5f6569`.

Source building passed and rebuilt all 15 installed vignettes. The archive's
DESCRIPTION and NEWS identify version 0.1.0. Its 108 R, Rd, test and package
vignette source files match the immediately preceding 0.6.0 archive byte for
byte. Website-only files and development evidence remain excluded.

## Completed verification

- Standard `R CMD check` finished with **Status: OK**: zero check errors,
  warnings or notes. All 1,150 test expectations passed, with no failures,
  warnings or skips. Examples, all 15 vignettes, and the PDF manual passed.
- The ordinary `pkgdown::build_site()` completed successfully. All 80 HTML
  pages display version 0.1.0, including the gallery, citation and news pages.
  No stale 0.6.0 labels remain in generated HTML.
- All 2,778 local links and assets passed validation, including fragment
  targets, the nine public-data figures and four downloads. The local HTTP
  preview was restarted and the gallery's 0.1.0 label confirmed in the browser.
- The removed scratch directory was not recreated by package or site builds.

The terminal supplied RStudio's Pandoc path for building; an RStudio project
session supplies Pandoc normally. Repository-index access was unavailable in
the restricted package-check environment, but installed dependencies were
present and the dependency check passed. The missing public-URL notice remains
intentional for this local-only site.

Build, check, test and site-validation evidence is retained in
[../releases/0.1.0/](../releases/0.1.0/). The release manifest records the new
archive and its checksum.
Historical test and benchmark records retain their original versions and
measurements; they have not been relabeled as new runs.
