# Geometry contracts and mathematical review

The coverage unit is a metric, point representation, and optimization capability.
The installed CSV ledgers are the support claim; an unlisted solver/device pair
has not been certified. All dimensions are general, subject to the stated
regularity conditions. The test suite covers metric duality, tangent membership, retraction derivatives,
feasibility and custom-objective descent; the per-feature ledgers identify the
actual tested combinations. Tests establish
numerical evidence, not a proof of global convergence for arbitrary user losses.

## Vector and elementary geometries

G01 uses the Frobenius metric on arbitrary-shaped real tensors. G02/G03 use the
round unit sphere and its identity embedding. The tangent/gradient map is
`U-X<X,U>`. Retraction is normalization; transport is tangent projection, which is
not isometric. Exact sphere exp/log are optional, with log excluding antipodes.
G04 is a product of column spheres, with no reduction across leading batch axes.
It has columnwise exact exponential and local logarithm maps, and its Hessian
conversion applies the round-sphere connection correction to every column.
G05 has positive simplex coordinates and metric `sum(U*V/X)` (Fisher--Rao,
not one quarter of it). Its dual gradient is `X*(G-<X,G>)`; exponential-coordinate
normalization remains the default retraction. Its exact local exp/log maps pull
back those of the radius-two square-root sphere; an exponential leaving the
positive orthant is outside this coordinate domain and is rejected. Its distance
is the radius-two sphere distance. Underflow to zero is a rejected trial.
G06 has curvature `-c`, ball radius `1/sqrt(c)`, and conformal factor
`2/(1-c||X||^2)`. G07 has Lorentz signature (-,+,...,+), `<X,X>_L=-1/c`, and
positive time coordinate. Raising the time index before tangent projection is
essential for the metric dual gradient. Additive-ball and normalized-timelike
retractions have explicit local domains; both models also expose their exact
exp/log maps. G08 uses real angles modulo 2*pi and a
flat metric. Losses must be periodic if intended as functions on the torus.

## Orthogonality and group geometries

G09 canonical Stiefel metric is `<U,V>-<X'U,X'V>/2`; its dual gradient is
`G-X G' X`. G10 Euclidean Stiefel uses `G-X sym(X'G)`. Both use polar
retraction. G11 Grassmann horizontal tangents satisfy `X'U=0`, with gradient
`(I-XX')G`. Quotient losses must be basis invariant. G12 represents the same
space by `P=XX'`; metric is half the Frobenius inner product, gradient is twice
the tangent projection. This factor is necessary for equivalence with G11.
G13/G14 fix SPD B, impose `X' B X=I`, and use `<U,BV>`. Cholesky whitening
`W=chol(B)'` maps these isometrically to G10/G11. B is copied and must match the
point's device and dtype. The whitening isometry transfers the exact Hessian
conversions from G10/G11. G15 is SO(p) with the embedded Frobenius metric; polar
steps from a tangent stay in its connected component, and its Hessian conversion
is the Euclidean Stiefel correction. G16 is the product
SO(d) x R^d with that rotation metric and Euclidean translation metric, not a
claim of bi-invariance under SE(d) group multiplication.

## Positive definite and correlation geometries

G18 AIRM uses `tr(X^-1 U X^-1 V)`, hence `grad=X sym(G) X`. The polynomial
`X+U+U X^-1 U/2` is a globally positive-definite second-order retraction in exact
arithmetic. Floating-point loss of positivity is rejected. Transport is the
isometric congruence along the endpoint affine-invariant geodesic; that path
choice is explicitly distinct from the polynomial retraction path. The exact
Hessian conversion is
`X sym(H) X + sym(U sym(G) X)`. It requires an exact ambient HVP.

G17 LERM is the pullback by matrix log: `<Dlog_X U,Dlog_X V>`. The inverse
of Dlog_X is Dexp_log(X), so the metric dual gradient applies that inverse twice
to sym(G). Retraction and transport add/move in log coordinates. G19 uses
`0.5 tr(L_X(U)V)`, with `X L_X(U)+L_X(U)X=U`. Gradient is
`2(X sym(G)+sym(G)X)`. Retraction `(I+L_X U)X(I+L_X U)` is the local Bures
exponential; I+L_X U must stay positive definite.

Log/sqrt/invsqrt use first-order spectral divided-difference rules with continuous
limits at equal eigenvalues. In particular Dlog_I(U)=U. Higher derivatives of
these custom rules are **not supported**. The package's automatic HVP and residual
JVP entry points reject such operations rather than returning an incomplete
second derivative. Native matrix exponential supports its tested higher derivatives.
Direct external double-backward calls on custom kernels remain outside the API
contract. Geometry Frechet kernels are value-level differential operators.

G25 ECM uses strictly lower coordinates of `Theta(C)=diag(L)^-1 L`, with L
lower Cholesky. G26 LEC uses the finite nilpotent log of Theta. Inverse charts
form L L' and divide by the square roots of its diagonal. Their metrics are
Euclidean on these strictly lower coordinates, and exact chart transport is
isometric. G27 is the quotient of AIRM by positive diagonal congruences: for
zero-diagonal symmetric U, solve
`(I+C^-1 elementwise C) a=-diag(C^-1 U)` and lift U to `U+diag(a)C+C diag(a)`.
Apply the AIRM metric to lifts. Its gradient follows the adjoint of diagonal
normalization; it is not replaced by a Cholesky chart. Its retraction normalizes
the AIRM polynomial retraction of the horizontal lift.

## Low rank and shape

G20 embedded rank-k rectangular matrices use SVD tangent projection and truncated
SVD retraction. G21 PSD tangents have zero null-null block; spectral truncation is
a local retraction while the k positive eigenvalues stay separated from zero.
G22 rank-k Bures uses the rank-restricted Sylvester inverse, with null-null block
zero, and the same congruence step as G19 on its stated local domain.
The additional `representation="factor"` uses full-column-rank Y modulo O(k),
Frobenius horizontal tangents, and additive horizontal retraction. For horizontal
Z, `U=ZY'+YZ'` preserves the metric. This recovers Riemann's spdk factor metric;
it is distinct from G21's embedded metric. The factor's local step requires
`Y'(Y+Z)` positive definite. Quotient objectives must depend only on YY'.

G23 is the regular unit-diagonal rank-k PSD stratum, k>=2. Orthogonal tangent
projection subtracts rank-projected diagonal normals by solving their Gram
system. Singular constraint strata are rejected by the solve. G24 subtracts the
trace-normal component along the support projector. Both retract by spectral
rank truncation then diagonal/trace normalization; on tangent directions the
normalization derivative is identity. G28 uses centered unit k x p preshapes
modulo SO(p), restricted to full column rank. The explicit reflections=TRUE
variant instead identifies O(p), matching the pinned Riemann shape quotient. Horizontal projection removes the
rotation component by a skew Sylvester solve. The supported domain excludes
lower-rank singular shapes; k>p. G29 uses fixed positive weights: gradients and
Hessian operators divide factor j by w_j, while metric inner products multiply
by w_j. Exact product Hessians are available when every factor supplies a
verified exact conversion.

## Optional operations and tensor rules

Exact logs/distances are supplied only where documented. No Stiefel log is
required for first-order optimization. Spectral rank projections have forward
support; derivatives at repeated singular/eigenvalues are not claimed. Solvers
never differentiate those retractions unless a capability explicitly permits it.
Exact ambient-to-Riemannian Hessian conversion is implemented for Euclidean,
sphere, oblique, torus, Euclidean/canonical Stiefel, Grassmann frame/projector,
generalized Stiefel/Grassmann, rotation and AIRM SPD geometries. A weighted
product has an exact conversion only when every factor does. Geometries without
a verified conversion require a user-supplied exact `rhess` for second-order solvers. The availability table from `riem.capabilities()` is authoritative.

Geometry kernels accept leading batch dimensions, with exact matching batch
shapes and no implicit broadcasting. Native vectorized kernels cover Euclidean,
sphere, oblique, torus, Stiefel, Grassmann frame/projector, and AIRM, LERM and
Bures--Wasserstein SPD geometries. Other built-ins use the generic pointwise
fallback; custom manifolds may register batch operations. Generic solves still
take a single point or named product. R logical membership decisions may
synchronize device tensors; inner products retain tensor-valued batch results.
Matrix kernels retain dtype and device.

Managed runs resolve one device before computation. Registered problem data,
least-squares weights and device-bound generalized geometry constants move with
the initial point. Integer indices and logical masks keep their dtype. CPU
float64 is the reference validation platform. CUDA and MPS combinations are
reported only after runtime availability and operation probes; those probes do
not establish performance or cross-device bitwise equivalence.

## References and source audit

* Boumal (2023), *An Introduction to Optimization on Smooth Manifolds*,
  https://www.nicolasboumal.net/book/ (gradients, quotients, retractions, Hessians).
* Thanwerdas and Pennec (2023), *Theoretically and computationally convenient
  geometries on full-rank correlation matrices*, https://arxiv.org/abs/2201.06282.
* Thanwerdas and Pennec (2021), quotient-affine correlation geometry,
  https://arxiv.org/abs/2103.04621.
* GeoJAX geometry source at the revision in `coverage/source-lock.yml` supplied
  inventory and representation definitions. Formula derivations above and tests
  are independent implementation checks. The MIT notice is included under
  `inst/NOTICE-GeoJAX` for provenance.
* Riemann's spdk horizontal factor metric is Frobenius and its factor action is
  right orthogonal. It maps to the Bures factor variant, not embedded PSD.
  The pinned Riemann correlation implementation explicitly reports its quotient
  operations unavailable; riemtorch implements the independently specified
  quotient metric separately from ECM and LEC.
  Riemann's statistical exports are out of scope, not unfinished obligations.

## Power, scaled, compact and complex geometries

Power geometry reduces over a distinguished factor axis and keeps ordinary
batch axes separate. Fixed metric scale s multiplies the metric by s^2,
distance by s and gradients/Hessians by 1/s^2. Positive tensors use log
coordinates. Affine subspaces use an orthonormal basis. O(p) includes both
components of square real Stiefel. Positive doubly stochastic matrices use
sum(U*V/X), gauge-fixed row/column tangent projection and Sinkhorn retraction.

Compact fixed-rank points contain orthonormal U,V and a full invertible k-by-k
core S. Tangents store (Up,C,Vp), representing Up V' + U C V' + U Vp'. This is
the dense embedded Frobenius metric in a compact representation. Retraction
uses QR on concatenated factors and a small SVD. A full core avoids a singular
diagonal-only coordinate chart at repeated singular values. Numerical rank
thresholds reject points too close to the rank boundary. Compact manifold
diagnostics explicitly materialize matrices for their derivative comparison.

For canonical Stiefel, write g=G-X G'X and
Dg[U]=H-U G'X-X H'X-X G'U. With C=U X'g+g X'U the Hessian is the tangent
projection of Dg[U]-(C+XX'C)/2. Exact matrix-exponential maps permit independent
second directional differences. Polar retraction is not declared second order
for the canonical metric. Grassmann logarithms are local and reject the cut
locus; map values and differentiated map capabilities are recorded separately.

Complex geometries use conjugate transpose and Re(trace(U*V)). Complex128
points pair with float64 losses, complex64 with float32. The complex sphere is
the real round sphere in twice as many coordinates; phases are a tensor
representation of flat circular factors. Frame quotient losses must be
unitary-basis invariant. Complex Euclidean, sphere and circle have tested
automatic Hessians. Complex Stiefel/Grassmann/unitary remain first-order
qualified; SVD differentiation at repeated spectra is not advertised.
