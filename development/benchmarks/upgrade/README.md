# Independent numerical comparisons

Run from the package root with the development R library and an isolated Python
containing Pymanopt 2.2.1, NumPy 2.0.2, SciPy 1.13.1 and psutil:

```
python development/benchmarks/upgrade/run.py
python development/benchmarks/upgrade/memory.py
```

The fixed seeds 1, 2 and 3 produce shared CSV inputs. All implementations use the
same objectives, initial points, metrics and double precision; phase problems
use complex128 with real float64 losses. The suite covers an eigenvector/PCA
objective, Stiefel Procrustes, rank-two matrix completion, SPD log-Euclidean
optimization, Huber least squares, Poincare embedding alignment, phase
synchronization and equality/nonnegativity constraints. Comparator variants are
recorded explicitly. Different solvers and derivative implementations mean
these timings are not a same-algorithm speed comparison.

Pymanopt supplies actual Riemannian PR+ CG runs for four cases. SciPy supplies
independent optimizers for the other four, using the exact log-coordinate
isometry for log-Euclidean SPD and a separately coded hyperbolic distance.
Robust least squares uses the same per-residual Huber objective and scale.
Constrained runs compare ALM with SLSQP, and independently reconstruct KKT
stationarity for SLSQP's returned active set.

Acceptance: objective disagreement <= 1e-6, metric gradient/KKT stationarity
<= 1e-6, and feasibility <= 1e-6. All 48 recorded runs passed. The largest
objective difference is about 5.92e-7 on constrained cases: the ALM answer is
slightly infeasible within tolerance. Unconstrained objective differences are
below 1e-10. These small examples do not establish global reliability or
large-scale speed parity. Full rows, including stopping text and counts, are in
`evidence/results.csv`; failures would remain as explicit rows.

Timings exclude R/Python startup and Riemannian warmups. CPU operations are
synchronous. CPU-to-CPU transfer cost is zero. Each solve has its own process;
RSS is sampled every 20ms with interpreter/runtime overhead included. The
post-warmup baseline and peak are recorded; brief peaks may be missed. CUDA
memory and synchronization require the separate hardware runner.

The memory experiment holds parameter size (256) and chunk size (64) fixed while
increasing terms from 1,024 to 65,536. Values without a gradient graph, detached
full gradients and HVPs are measured separately. Full differentiable values
intentionally retain all chunk graphs and are not promised bounded memory.
Results in `evidence/memory.csv` show gradient/HVP incremental sampled RSS near
14–18 MB at both sizes; full differentiable values increase to about 171 MB.
Diagnostic values reached about 44 MB because process/allocator high-water marks
also include R temporaries. These are sampled process measurements, not an
exact tensor-allocation profiler or a proof of asymptotic bounds.
