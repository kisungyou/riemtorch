# Inspect Published Geometry and Solver Evidence

Inspect Published Geometry and Solver Evidence

## Usage

``` r
riem.support(
  ledger = c("geometry", "solvers", "support", "devices", "competitors")
)
```

## Arguments

- ledger:

  \`"geometry"\`, \`"solvers"\`, \`"support"\`, \`"devices"\`, or
  \`"competitors"\`.

## Value

A data frame from the installed coverage ledger. Status distinguishes
supported, experimental, planned and unsupported capabilities.

## Examples

``` r
head(riem.support("geometry"))
#>    id    source_geometry          constructor                       metric
#> 1 G01          Euclidean   manifold.euclidean                    euclidean
#> 2 G02             Sphere      manifold.sphere                        round
#> 3 G03    SphereExtrinsic      manifold.sphere                        round
#> 4 G04            Oblique     manifold.oblique                 column_round
#> 5 G05 ProbabilitySimplex manifold.multinomial                   fisher_rao
#> 6 G06       PoincareBall  manifold.hyperbolic poincare; negative curvature
#>                     representation   status
#> 1                           tensor verified
#> 2                      unit vector verified
#> 3 identity embedding; alias of G02 verified
#> 4               p x k unit columns verified
#> 5      positive probability vector verified
#> 6             d-vector inside ball verified
#>                                                     evidence
#> 1   test-geometry-contracts; test-solver-evidence: euclidean
#> 2      test-geometry-contracts; test-solver-evidence: sphere
#> 3      test-geometry-contracts; test-solver-evidence: sphere
#> 4     test-geometry-contracts; test-solver-evidence: oblique
#> 5 test-geometry-contracts; test-solver-evidence: multinomial
#> 6    test-geometry-contracts; test-solver-evidence: poincare
#>                                            space_equivalence
#> 1 distinct metric/space family or parameterized construction
#> 2 distinct metric/space family or parameterized construction
#> 3                                                        G02
#> 4 distinct metric/space family or parameterized construction
#> 5 distinct metric/space family or parameterized construction
#> 6 distinct metric/space family or parameterized construction
```
