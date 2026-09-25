# Solver and training contracts

Every solver uses the supplied geometry metric and objective. `riem_fit` records
configuration, objective, gradient norm when evaluated, constraint residuals,
iteration/evaluation counts, accepted/rejected steps, termination and history.
One `fn` count is one scalar objective evaluation, even when it consumes all
finite-sum terms; `terms` records that additional work. Gradient and HVP counts
are request counts, and nested value/gradient evaluations also increment their
own counters. `residual`, `jvp`, and `adjoint` record least-squares work. Rejected
steps may fail before objective evaluation. A failure retains the last valid
iterate and an explanatory message. `converged_gradient` is a stationarity
criterion under the selected metric, not a global certificate.

## First order

Steepest descent uses -grad and Armijo backtracking. Adaptive Armijo seeds the
next search at 1.5 times the last accepted scalar step. Fixed-step search is
explicit and does not impose descent. Strong Wolfe differentiates
`f(retr(x,t*d))` with respect to t; it never substitutes a transported direction
for the curve velocity. A bracket/bisection search enforces both inequalities.
Unsupported retraction derivative combinations are rejected.

The default conjugate-gradient variant is PR+: compare the new gradient to the old gradient
transported to the new point, divide its PR numerator by the old squared metric
norm, and truncate beta below at zero. Periodic and non-descent restarts are
counted. FR, HS+ and Dai–Yuan variants are also available. BB uses transported s
and y, BB1 <s,s>/<s,y> or BB2 <s,y>/<y,y> (optionally alternating), bounded steps,
and Armijo. L-BFGS transports every stored pair into the current tangent space,
recomputes curvature, drops invalid pairs, and applies the standard two-loop
recursion with scalar <s,y>/<y,y> scaling. Secant curvature skips are recorded.
These are transport-specific practical algorithms. Projection transport is not
claimed isometric, and no blanket quasi-Newton convergence theorem is asserted.

Stochastic gradient uses uniform independent minibatches. Sampling without
replacement is within each batch, not a persistent random epoch permutation.
Means average sampled terms; sums multiply that average by n. A vectorized
`batch_fn` returns one loss per requested index and avoids per-term R calls.
Full objective and gradient diagnostics are computed in bounded chunks initially
and finally; positive `full_evaluation_every` adds periodic full checks. Other
iterations evaluate the minibatch only, and their explicitly labelled gradient
norms cannot establish full stationarity. Chunking bounds each graph's memory but
does not change the total work of a requested full evaluation. Constant,
polynomial and cosine schedules are provided. Alternating gradient updates one
product factor in the specified cyclic order, with the complete objective used
for line search.

## Second order and residual problems

HVPs are matrix-free by default. Euclidean, round sphere, oblique, flat torus,
Euclidean/canonical Stiefel, Grassmann frame/projector, generalized Stiefel/Grassmann,
rotations, AIRM SPD and weighted products of supported factors have tested exact
ambient-to-Riemannian conversion. An explicit `rhess(x,u)` enables other
geometries. Analytic gradient callbacks require a corresponding explicit rhess
for second-order use. No finite-difference Hessian is silently substituted.
First-order custom spectral functions are blocked in automatic higher-order
paths. Exact connection corrections, metric self-adjointness and
second-order-retraction directional differences have independent fixtures.

Newton-CG uses truncated CG with negative-curvature termination, descent
fallback, and line search. Trust regions use Steihaug CG, boundary intersection,
actual/predicted reduction, and radius updates. Its model is the exact Hessian
quadratic. Cubic regularization adds sigma*||s||^3/3. Its subproblem starts at the
exact ray Cauchy minimizer, then performs backtracking gradient iterations in the
fixed tangent space. The recorded model-gradient residual states how accurately
it was solved; outer steps use actual/predicted reduction and adapt sigma.
This is an approximate ARC variant, marked experimental; no second-order
stationarity certificate or hard-case global subproblem claim is made.

For residual r and fixed PSD weight W, f=0.5*r'Wr. JVPs use native
reverse-over-reverse differentiation or explicit callbacks; adjoints convert
ambient pullbacks through the selected metric, including reciprocal product
weights. GN uses J*WJ and CG plus line search. LM adds lambda*I in the tangent
metric; its acceptance denominator uses the undamped GN model, while damping
controls the computed direction and is adapted by the reduction ratio. The
model is labeled `gauss_newton`, not an exact Hessian.

## Extended searches

Particle swarm owns one tangent velocity per particle, transports it after each
accepted retraction, and combines inertia with cognitive/social displacements.
Where an exact local log exists it is used; otherwise displacements are projected
ambient chords, with orthogonal alignment for quotient frames and SO alignment
for Kendall shapes. A failed local log (for example antipodes) or unavailable
local domain is reported. Independent random coefficients are scalar per
particle, preserving coordinate invariance within the declared representation.

Nelder--Mead creates a metric-orthonormal tangent simplex with dimension+1
vertices. Its centroid is a tangent average retracted from the best vertex.
Reflection/expansion/contraction use that centroid's displacement to the worst;
shrink moves other vertices toward the best. This is a local retraction-based
simplex variant, not a claim of a canonical intrinsic centroid or global
convergence. Termination by diameter is only `stopped_small_step`.

Stiefel annealing proposes exp(scale*skew(A))*X for an isotropic Gaussian square
A. The symmetric left-rotation proposal is reversible on the connected orbit;
Metropolis acceptance uses a geometric temperature schedule. Square Stiefel
frames remain in their initial determinant component. Grassmann MACG search
samples polar(A^(1/2) Z), selects elite frames and fits their MACG shape by the
fixed-point equation p/(k*N)*sum X(X' A^-1 X)^-1 X'. A trace-p constraint fixes
scale and an explicit 1e-6 ridge makes each fit positive definite. The ridge means
this is a regularized search distribution, not an exact unregularized MLE.
Fit residuals and shrinkage are reported. All four searches are experimental
heuristics and deliberately return gradient_norm=NA and converged=FALSE.

## Execution, training and checkpointing

`riem.optimize()` resolves one device and floating dtype before evaluating a
callback. Resolution uses the explicit request, the package option, the
environment variable, then automatic CUDA/MPS/CPU discovery. Automatic selection
skips devices that fail a core-operation probe at the preserved precision;
explicit unavailable or incompatible devices are errors. The problem's registered
data, weights and geometry state move with the point. Captured closure tensors do
not. Result metadata records the request, source, device, dtype and runtime.

Native torch optimizer subclasses update actual parameter objects by copy_.
Each tensor is a declared manifold factor. For metric weight w, use grad/w and
w*<grad/w,grad/w> in adaptive statistics. RSGD forms d=beta*m+grad, retracts
-lr*d, and transports d. There is no dampening, Nesterov or ambient weight decay.
Factorwise RADAM forms exponentially weighted first moment and a single scalar
squared metric norm per tensor, applies both bias corrections, and places eps
outside the square root. Its transported state respects quotient basis changes.
This implements Riemannian Adam's factorwise principle, not rectified Adam,
coordinatewise Euclidean Adam, or an AMSGrad convergence theorem. Missing
gradients and frozen parameters are skipped; finite candidate points and tangent
states are staged for *all* groups before any update is committed.

`riem.train()` provides the managed training boundary: it places the module
before invoking `optimizer_factory(model)`, transfers each minibatch, and records
per-epoch mean minibatch loss and execution metadata. Low-level optimizers retain
their normal torch contract and do not move already registered parameters. The
managed loop uses one device per run and does not provide distributed training
or automatic mixed precision.

Checkpoints use torch_save/torch_load, not ordinary R serialization of tensor
pointers. Parameter keys, geometry specifications, options and states are
validated on load. Schema version two may also contain module parameters and
buffers, caller-supplied epoch/step/data-order state, and R, torch CPU and visible
CUDA RNG states. Version-one parameter checkpoints remain readable. Create a
matching model and optimizer on the target device before loading; a CPU target
can restore tensors saved on CUDA. Geometry callbacks are reconstructed by the
program and are not loaded from disk. Deterministic continuation still requires
matching objective code, runtime behavior and data order, and cross-device
results need not be bitwise identical.

## Primary references

* Boumal (2023), https://www.nicolasboumal.net/book/.
* Absil, Mahony and Sepulchre (2008), *Optimization Algorithms on Matrix
  Manifolds*, https://press.princeton.edu/absil (matrix manifold optimization).
* Bécigneul and Ganea (2019), *Riemannian Adaptive Optimization Methods*,
  https://arxiv.org/abs/1810.00760 (factorwise adaptation; no theorem transfer is
  claimed for arbitrary retractions, transports and user objectives).
* Chikuse (1990), *The matrix angular central Gaussian distribution*,
  https://doi.org/10.1016/0047-259X(90)90050-R (MACG sampling/fitting model).
* R torch optimizer and autograd interfaces,
  https://torch.mlverse.org/docs/reference/optimizer and
  https://torch.mlverse.org/docs/articles/extending-autograd.html.
* Pinned GeoJAX and Riemann inventories in `coverage/source-lock.yml` establish
  the source algorithm target; they do not imply identical algorithm variants.

## Upgrade solver contracts

Optional value_rgrad callbacks share value/gradient work. Preconditioners apply
positive, metric-self-adjoint tangent operators in truncated CG. Callback states
are detached; final events are observational. Time limits apply between backend
operations. Primitive evaluation budgets exclude the separately reported term
count. Cached values are scoped to an accepted point and a fixed minibatch.

SVRG forms grad_i(x)-T(grad_i(snapshot)-full_grad(snapshot)). Its local-log path
requires an explicitly endpoint-compatible transport. SRG transports the previous
recursive estimate minus the previous sample gradient and adds the new sample
gradient; periodic full refreshes limit estimator drift. Stochastic estimates
are not used to claim full stationarity. Transport/retraction choices remain
part of each algorithm's qualification, not a universal convergence theorem.

Robust least squares uses 0.5 sum w_i c^2 rho((r_i/c)^2), with linear, Huber,
soft-L1 and Cauchy rho. Its positive-semidefinite Gauss–Newton operator uses
rho' weights and omits rho''; only linear loss accepts a full matrix weight.

AMSGrad takes a running maximum of uncorrected second moments, followed by the
usual bias correction. AdaGrad accumulates squared metric norms. Power factors
keep independent variances. Sparse row SGD/Adam coalesce repeated indices,
update only touched rows and preserve untouched state exactly; Adam uses
row-local clocks. Line-search optimizer failures restore parameters and
gradients, but closures must not mutate unrelated buffers or RNG.

Training accumulation averages minibatch gradients, including a partial final
group. Metric clipping uses the total weighted Riemannian norm. Validation runs
without gradient recording, scheduler hooks run after validation, and data-state
hooks restore caller-owned epoch data order before the next iterator. Optimizer
schema 2 supports earlier scalar variance layouts; a legacy power variance is
replicated across factors during migration. Subsequent updates use independent
factor adaptation, so that layout migration is not bitwise replay of the old
shared-variance algorithm.

Augmented Lagrangian uses lambda'h+rho||h||^2/2 for equalities and
(||max(0,mu+rho*g)||^2-||mu||^2)/(2*rho) for g<=0. Multipliers are updated after
inner solves; KKT residuals use the unpenalized Lagrangian. Inner failure and
budget stops remain distinct. Intrinsic proximal gradient applies prox_h after
an exponential smooth-gradient step and checks total sufficient decrease.
The callback must solve h(y)+d(y,q)^2/(2*t), not ambient shrinkage/projection.
Cyclic proximal point uses diminishing steps and reports cycle displacement/t;
that residual alone does not certify stationarity of a nonsmooth sum.

Further primary comparisons: R-SVRG (https://arxiv.org/abs/1702.05594), R-SRG
(https://proceedings.mlr.press/v80/kasai18a.html), and Manopt.jl's augmented
Lagrangian and intrinsic proximal-gradient documentation, linked in the installed
competitor ledger. Independent executed comparisons use Pymanopt 2.2.1 and SciPy;
they do not imply every compared library has been executed.
