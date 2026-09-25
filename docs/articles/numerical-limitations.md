# Numerical domains and reproducibility

CPU float64 is the reference validation platform. A managed run probes
visible devices and preserves its requested precision: compatible CUDA
devices are considered first, then compatible MPS, then CPU. This is
runtime qualification, not evidence that an accelerator is faster or
numerically identical. CUDA and MPS were unavailable in the local
reference environment. Use `device = "cpu"` for a reproducible host
override; an unavailable explicit accelerator is an error rather than a
request to fall back silently.

Batch geometry is supported with exact shape matching; independent
batched solves are not implicitly reduced to one loss. Native kernels
cover the most frequently used vector, frame, Grassmann and SPD
geometries. Explicit host scalar extraction is still needed for stopping
and validation and may synchronize an accelerator. An out-of-memory or
numerical failure after execution starts does not restart the run on
another device.

Smooth matrix functions should be differentiated as matrices, not
through arbitrary eigenvector gauges. The matrix-log custom backward is
finite at an identity spectrum.

``` r

X <- torch_eye(2, dtype = torch_float64(), requires_grad = TRUE)
value <- riem.matrix.function(X, "log")$sum()
value$backward()
X$grad
#> torch_tensor
#>  1  1
#>  1  1
#> [ CPUDoubleType{2,2} ]
```

These custom log/root backward rules are first-order only. An exact HVP
through them requires an analytic callback. Do not call external double
backward on these primitives and assume it implements the full matrix
second derivative. Rank projection and quotient charts have additional
domains in their help pages.

R torch 0.17.0 can promote a numeric scalar on the **left** of a
zero-dimensional tensor operation through the default float dtype. For
strict float64 objectives, write `tensor * 0.1` instead of
`0.1 * tensor`, or construct a matching torch constant. riemtorch’s
scalar metric arithmetic follows this convention.

An example based on R’s built-in iris data demonstrates the
registered-data boundary and an SPD objective. The data is converted
once; the execution driver moves the registered target with the point.
Tensors captured privately in a callback closure are not discoverable
and remain the caller’s responsibility.

``` r

X <- torch_tensor(as.matrix(iris[, 1:4]), dtype = torch_float64())
X <- X - X$mean(dim = 1, keepdim = TRUE)
C <- X$t()$matmul(X)/(nrow(iris) - 1) + torch_eye(4, dtype = X$dtype)*0.1
T <- riem.matrix.function(C, "log")
M <- manifold.spd(4, "lerm")
P <- riem.problem(M, function(x, data) {
  torch_sum((riem.matrix.function(x, "log") - data)^2) / 2
}, data = T)
fit <- riem.optimize(P, torch_eye(4, dtype = X$dtype), device = "cpu")
fit
#> <riem_fit> steepest_descent on spd 
#>  Objective: 3.754793e-30  | gradient norm: 2.739e-15 
#>  Termination: converged_gradient  | iterations: 1
stopifnot(fit$objective < 1e-10)
```

Finite-sum SGD performs full objective and gradient evaluations
initially and finally. Intermediate iterations use minibatches unless a
positive `full_evaluation_every` requests periodic full diagnostics.
`batch_fn` avoids per-term R calls, and chunked full evaluations bound
the size of each computation graph; they do not remove the total cost of
a requested full diagnostic.

Version-two checkpoints can include model parameters and buffers,
caller-owned progress or data-order metadata, and R, torch CPU and
visible CUDA RNG states. Exact continuation still depends on matching
objective code, runtime behavior and minibatch order. Results resumed
across CPU and CUDA need not be bitwise identical. Custom geometry
callbacks are reconstructed by the program and are never deserialized
from a checkpoint.
