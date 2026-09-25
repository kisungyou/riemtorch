# Generate Feasible Initial Points

Generate Feasible Initial Points

## Usage

``` r
riem.random(manifold, batch_shape = integer(), dtype = NULL, device = NULL)
```

## Arguments

- manifold:

  A geometry specification.

- batch_shape:

  Optional positive integer batch dimensions.

- dtype:

  Optional floating-point torch dtype or supported dtype name. The
  default is the dtype of device-bound manifold state, or float64.

- device:

  Device request. The default follows \[riem.device()\], which
  automatically selects a compatible visible accelerator before CPU.

## Value

A tensor or named product. Random generation uses torch's RNG.

## Details

Most geometries project a Gaussian draw. This is an initialization law,
not a claim of uniformity. Sphere and orthogonal frames are isotropic.

## Examples

``` r
if (torch::torch_is_installed()) {
  torch::torch_manual_seed(1)
  riem.random(manifold.sphere(3))
}
#> torch_tensor
#>  0.9239
#>  0.3729
#>  0.0862
#> [ CPUDoubleType{3} ]
```
