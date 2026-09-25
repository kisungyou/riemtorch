# Source audit boundary

The planning baseline is Riemann e847692576f8b250c8140e1343b145f72caa379f and
GeoJAX 46210cdb5ea2fc6db1a5a15d24d5e3a759d0d025. The local Riemann checkout is
exactly that revision. Pinned GeoJAX geometry definitions were inspected with
git show; the current local checkout (488890e0598a56d1a87bc46f7f430c17f3afd489)
also supplied supplemental numerical-domain review. It is not substituted for
the locked baseline. No source package is imported by installed riemtorch code.

Material differences resolved:

* Riemann spdk uses a p x k factor and Frobenius horizontal metric. It is the
  Bures quotient factor representation, not embedded PSD. riemtorch implements
  both metrics and tests matrix/factor pushforward isometry.
* Riemann rotation tangents use body-skew coordinates. Multiplication X*Omega
  maps them isometrically to riemtorch's ambient rotation tangents, because X
  is orthogonal and both metrics are Frobenius.
* Riemann multinomial uses sum(U*V/X), with distance 2*acos(sum(sqrt(X*Y))).
  riemtorch preserves that scaling and uses a different, explicitly named
  exponential-coordinate retraction for optimization.
* Riemann's pinned correlation intrinsic operations explicitly error as
  unavailable. It therefore cannot serve as a numerical quotient oracle.
  riemtorch's affine quotient follows the horizontal-lift definition and tests
  independent metric duality; ECM and LEC remain separate chart metrics.
* Riemann's pinned landmark geometry identifies O(p), including reflections;
  GeoJAX KendallShape uses SO(p). `manifold.landmark(..., reflections=TRUE)`
  explicitly preserves the O(p) variant, while FALSE gives SO(p). They have the
  same local horizontal metric on the full-rank stratum but different global
  equivalence classes. Tests check reflection-gauge invariance separately.
* GeoJAX Grassmann projectors use a distinct representation. riemtorch applies
  the half-Frobenius metric so pushforward from frame tangents is isometric;
  this is an equivalence, not a new dimension count.

The implementation reviews metric formulas and numerically tests the local
contracts. It does not claim an independent external mathematical peer review.
Portable analytic fixtures remain primary; source-package results alone cannot
establish correctness. Source licenses and provenance notices are retained.

Supplemental portable distance fixtures were generated separately with the
preinstalled released Riemann **0.1.7**, not the pinned development revision.
The generator and session information are in development/reference-oracles.
They cover round sphere and noncommuting SPD AIRM/LERM distances; installed tests
read the frozen numeric fixtures without loading Riemann. This distinction is
intentional: source-formula audit and released-package comparison have separate
provenance. Analytic metric and derivative tests remain the primary evidence.
