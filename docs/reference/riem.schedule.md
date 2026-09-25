# Learning-Rate Schedules

Learning-Rate Schedules

## Usage

``` r
riem.schedule(
  initial,
  type = c("constant", "polynomial", "cosine"),
  decay = 0.01,
  power = 0.5,
  total_iterations = 100,
  minimum = 0
)
```

## Arguments

- initial:

  Positive initial rate.

- type:

  \`"constant"\`, \`"polynomial"\`, or \`"cosine"\`.

- decay:

  Nonnegative polynomial decay coefficient.

- power:

  Positive polynomial exponent.

- total_iterations:

  Positive cosine horizon.

- minimum:

  Nonnegative minimum rate, at most initial.

## Value

A function of the zero-based iteration number.

## Examples

``` r
schedule <- riem.schedule(0.1, "polynomial", decay = 0.01)
schedule(10)
#> [1] 0.09534626
```
