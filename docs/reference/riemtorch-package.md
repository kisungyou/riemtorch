# Optimization on Riemannian Manifolds with Torch

Define a scalar tensor objective with \[riem.problem()\], choose a
geometry, and solve with \[riem.optimize()\]. Managed runs select one
compatible device automatically; pass \`device = "cpu"\` or an indexed
CUDA request to override selection. Registered problem data follows the
point to that device.

## Details

Geometry operations use leading batch axes followed by point axes.
Native batch kernels cover the common vector, frame, Grassmann and SPD
geometries; solvers optimize one point or named product rather than
reducing independent batched problems. \[riem.train()\] provides managed
module and minibatch placement, while \[optim_rsgd()\] and
\[optim_radam()\] remain available for ordinary torch loops. The
numerical reference platform is CPU float64 with torch 0.17.0; CUDA and
MPS support is qualified at runtime.

## See also

\[riem.devices()\], \[manifold.sphere()\], \[optim_rsgd()\],
\[riem.support()\]

## Author

**Maintainer**: Kisung You <kisung.you@outlook.com>
([ORCID](https://orcid.org/0000-0002-8584-459X))

Authors:

- Kisung You <kisung.you@outlook.com>
  ([ORCID](https://orcid.org/0000-0002-8584-459X))
