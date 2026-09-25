# Weighted Product Geometry

Weighted Product Geometry

## Usage

``` r
manifold.product(factors, weights = NULL)
```

## Arguments

- factors:

  Nonempty, uniquely named list of manifold specifications.

- weights:

  Positive fixed metric weights; defaults to one per factor.

## Value

A \`riem_product\`, also inheriting from \`riem_manifold\`.

## Details

Points, tangents and gradients are identically named lists, including
nested products. Gradient conversion divides each factor by its metric
weight. All leaves of a solve must share device and dtype. Batches are
never treated as product factors.

## Examples

``` r
M <- manifold.product(list(position = manifold.euclidean(2),
                           direction = manifold.sphere(3)), c(2, 5))
M
#> <riem_manifold> product  | metric: weighted_product  | dimension: 4 
```
