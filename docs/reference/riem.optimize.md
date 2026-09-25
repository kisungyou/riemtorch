# Optimize a User-Defined Manifold Objective

Generic solvers share metric derivatives, feasibility checks and
diagnostics.

## Usage

``` r
riem.optimize(
  problem,
  initial,
  method = "steepest_descent",
  control = list(),
  device = NULL,
  dtype = NULL
)
```

## Arguments

- problem:

  A problem from \[riem.problem()\], \[riem.problem.leastsquares()\] or
  \[riem.problem.finitesum()\].

- initial:

  A feasible tensor or named product, without batch axes.

- method:

  Solver key; see Details.

- control:

  Named options. Common options: \`max_iterations\` (200),
  \`gradient_tolerance\` (1e-7), \`step_tolerance\` (1e-12),
  \`step_size\` (1), \`line_search\` (\`"armijo"\`,
  \`"adaptive_armijo"\`, \`"strong_wolfe"\`, \`"fixed"\`),
  \`max_linesearch\` (30), \`armijo\` (1e-4), \`wolfe\` (0.9),
  \`backtrack\` (0.5), \`min_step\` (1e-14), \`max_step\` (100), and
  \`history\` (TRUE). Execution controls: \`callback\` (NULL),
  \`max_time\` (Inf seconds), \`max_evaluations\` (Inf primitive
  evaluations, excluding the term counter). Callbacks receive detached
  iteration states and a final event; TRUE stops an iteration event.
  Final events are observational. Time limits are checked between
  operations and cannot interrupt a running backend kernel. \`cg_beta\`
  selects PR+, FR, HS+ or DY; \`bb_variant\` selects BB1, BB2 or
  alternating. \`epoch_length\` and \`snapshot_every\` default to 100
  for SVRG/SRG. Secant controls: \`memory\` (10), \`restart_every\`
  (50). Model controls: \`trust_radius\` (1), \`max_trust_radius\`
  (100), \`acceptance_ratio\` (0.1), \`inner_iterations\` (50),
  \`inner_tolerance\` (0.1), \`damping\` (0.01), \`regularization\` (1).
  Stochastic/block controls: \`batch_size\` (1),
  \`full_evaluation_every\` (0, meaning only initial and final full
  evaluations), \`schedule\` (NULL, constant step_size), \`blocks\`
  (NULL, cyclic factor names). Search controls: \`population\` (20),
  \`temperature\` (1), \`cooling\` (0.95), \`proposal_scale\` (0.2),
  \`inertia\` (0.5), \`cognitive\` (1), \`social\` (1),
  \`elite_fraction\` (0.25). Unknown options are errors.

- device:

  Execution device request. \`NULL\` follows \[riem.device()\]; use
  \`"cpu"\` to force CPU or a value such as \`"cuda:1"\` to select a
  GPU.

- dtype:

  Optional floating-point dtype. When omitted, the initial point's
  precision is preserved and automatic selection skips incompatible
  devices.

## Value

A \`riem_fit\`: point, objective, gradient_norm, constraint_residuals,
manifold, method, control, iterations, evaluations, accepted_steps,
rejected_steps, termination, converged, history, diagnostics and
message.

## Details

Constrained problems use \`augmented_lagrangian\`. Additional controls
are \`penalty\` (10), \`penalty_increase\` (5), \`penalty_contraction\`
(0.5), \`feasibility_tolerance\` and \`complementarity_tolerance\`
(1e-6), \`inner_method\` ("lbfgs"), \`inner_control\`
(list(max_iterations=100)), and \`inner_accuracy\` (0.1, tightened each
outer iteration). Composite problems use \`proximal_gradient\` or
\`cyclic_proximal_point\`, with \`proximal_tolerance\` (1e-7) and
\`cycle_power\` (0.75, in (0.5,1\]). Diminishing cyclic step sizes alone
cannot certify stationarity; a small cycle residual reports a stop
without a convergence claim.

First-order methods are \`steepest_descent\`, \`conjugate_gradient\`
(transported PR+ with descent restarts), \`barzilai_borwein\` (BB1), and
\`lbfgs\` (transported secants and curvature rejection). Quasi-Newton
secants and stored vectors move to the accepted point before reuse;
projection transports give practical variants without a universal
convergence guarantee.

\`trust_regions\` uses Steihaug truncated CG; \`newton_cg\` uses
truncated CG plus line search. \`adaptive_regularization_cubics\` uses a
Cauchy-initialized approximate cubic subproblem. These require an exact
\`rhess\` callback or verified automatic Hessian conversion.
\`gauss_newton\` and \`levenberg_marquardt\` require a least-squares
problem, use the metric adjoint, and report Gauss–Newton rather than
exact Hessian models. LM adapts damping with actual/predicted reduction.
None of these certifies a global optimum.

\`svrg\` and \`srg\` add finite-sum variance reduction. SVRG requires an
exact local logarithm to identify its snapshot transport path.
\`stochastic_gradient\` requires a finite sum. It uses chunked full
initial and final evaluations; positive \`full_evaluation_every\`
requests additional full checks. Other iterations report explicitly
labelled minibatch diagnostics, which cannot establish gradient
convergence. \`alternating_gradient\` updates one named product block
per iteration. Derivative-free methods are \`particle_swarm\`,
\`nelder_mead\`, \`stiefel_annealing\`, and \`grassmann_macg\`; see the
installed mathematical notes for their local movement and sampling
rules. They return no gradient-based convergence claim.

Strong Wolfe differentiates the actual retraction curve. It is enabled
only on geometries advertising that derivative. Small-step termination
is \`stopped_small_step\`, distinct from \`converged_gradient\`.
Rejected domain trials are counted; accepted iterates are never silently
repaired.

## Examples

``` r
if (torch::torch_is_installed()) {
  torch::torch_manual_seed(1)
  M <- manifold.sphere(3)
  a <- torch::torch_tensor(c(1, 2, 3), dtype = torch::torch_float64())
  P <- riem.problem(M, function(x, data) -torch::torch_sum(data*x), data = a)
  fit <- riem.optimize(P, riem.random(M, device = "cpu"),
                       "conjugate_gradient", device = "cpu")
  fit
  fit$point
}
#> torch_tensor
#>  0.2673
#>  0.5345
#>  0.8018
#> [ CPUDoubleType{3} ]
```
