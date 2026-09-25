# Applications and Portable Device Execution

``` r

library(torch)
library(riemtorch)
knitr::opts_chunk$set(collapse = TRUE, comment = "#>",
                      eval = torch::torch_is_installed())
if (torch::torch_is_installed()) {
  torch_manual_seed(12)
  torch_set_num_threads(1)
  device <- riem.device()
  device
}
```

    ## torch_device(type='cpu')

`riemtorch` selects one compatible device for each managed run. It
preserves the requested precision, so an accelerator that cannot execute
float64 is skipped rather than used with a silent conversion. Pass
`device = "cpu"` to reproduce a run on the host, or `device = "cuda:1"`
to select the second visible CUDA GPU.
[`riem.devices()`](https://www.kisungyou.com/riemtorch/reference/riem.devices.md)
reports the devices that the current R process can see and the result of
a small compatibility probe. Resolution follows an explicit argument,
`options(riemtorch.device = ...)`, `RIEMTORCH_DEVICE`, then automatic
discovery. An explicit `device = "auto"` asks for discovery even when a
default is set. Scheduler visibility restrictions are respected because
only devices exposed to the R process are considered.

``` r

riem.devices()
#>   device type index available compatible
#> 1    mps  mps    NA      TRUE      FALSE
#> 2    cpu  cpu    NA      TRUE       TRUE
#>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            reason
#> 1 Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead. Exception raised from empty_mps at /Users/runner/work/libtorch-mac-m1/libtorch-mac-m1/pytorch/aten/src/ATen/mps/EmptyTensor.cpp:45 (most recent call first): frame #0: c10::Error::Error(c10::SourceLocation, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>) + 56 (0x106446978 in libc10.dylib) frame #1: at::detail::empty_mps(c10::ArrayRef<long long>, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>, std::__1::optional<c10::MemoryFormat>) + 1184 (0x14f6835dc in libtorch_cpu.dylib) frame #2: at::native::empty_strided_mps(c10::ArrayRef<long long>, c10::ArrayRef<long long>, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>) + 140 (0x14f6a06c4 in libtorch_cpu.dylib) frame #3: at::_ops::empty_strided::redispatch(c10::DispatchKeySet, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>) + 172 (0x14bb75304 in libtorch_cpu.dylib) frame #4: at::_ops::empty_strided::call(c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>) + 340 (0x14bb74e60 in libtorch_cpu.dylib) frame #5: at::native::_to_copy(at::Tensor const&, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>, bool, std::__1::optional<c10::MemoryFormat>) + 2280 (0x14b396c94 in libtorch_cpu.dylib) frame #6: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>, bool, std::__1::optional<c10::MemoryFormat>) + 172 (0x14b83f72c in libtorch_cpu.dylib) frame #7: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>, bool, std::__1::optional<c10::MemoryFormat>) + 172 (0x14b83f72c in libtorch_cpu.dylib) frame #8: c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::DispatchKeySet, at::Tensor const&, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>, bool, std::__1::optional<c10::MemoryFormat>), &torch::autograd::VariableType::(anonymous namespace)::_to_copy(c10::DispatchKeySet, at::Tensor const&, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>, bool, std::__1::optional<c10::MemoryFormat>)>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>, bool, std::__1::optional<c10::MemoryFormat>>>, at::Tensor (c10::DispatchKeySet, at::Tensor const&, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>, bool, std::__1::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>, bool, std::__1::optional<c10::MemoryFormat>) + 1232 (0x14de2b960 in libtorch_cpu.dylib) frame #9: at::_ops::_to_copy::call(at::Tensor const&, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>, bool, std::__1::optional<c10::MemoryFormat>) + 348 (0x14b83f3ec in libtorch_cpu.dylib) frame #10: at::_ops::to_dtype_layout::call(at::Tensor const&, std::__1::optional<c10::ScalarType>, std::__1::optional<c10::Layout>, std::__1::optional<c10::Device>, std::__1::optional<bool>, bool, bool, std::__1::optional<c10::MemoryFormat>) + 360 (0x14ba0d03c in libtorch_cpu.dylib) frame #11: at::Tensor::to(c10::TensorOptions, bool, bool, std::__1::optional<c10::MemoryFormat>) const + 312 (0x13daf7a4c in liblantern.dylib) frame #12: _lantern_Tensor_to + 140 (0x13de664c0 in liblantern.dylib) frame #13: torch_tensor_cpp(SEXPREC*, Rcpp::Nullable<XPtrTorchDtype>, Rcpp::Nullable<XPtrTorchDevice>, bool, bool) + 4124 (0x1206d4ddc in torchpkg.so) frame #14: _torch_torch_tensor_cpp + 164 (0x1204eeb24 in torchpkg.so) frame #15: R_doDotCall + 1612 (0x100ec3a0c in libR.dylib) frame #16: bcEval_loop + 81268 (0x100f13bf4 in libR.dylib) frame #17: bcEval + 592 (0x100ef23d0 in libR.dylib) frame #18: Rf_eval + 556 (0x100ef1b6c in libR.dylib) frame #19: R_execClosure + 816 (0x100ef4730 in libR.dylib) frame #20: applyClosure_core + 164 (0x100ef3824 in libR.dylib) frame #21: Rf_eval + 1224 (0x100ef1e08 in libR.dylib) frame #22: do_set + 360 (0x100ef7fa8 in libR.dylib) frame #23: Rf_eval + 1012 (0x100ef1d34 in libR.dylib) frame #24: do_begin + 400 (0x100ef6fd0 in libR.dylib) frame #25: Rf_eval + 1012 (0x100ef1d34 in libR.dylib) frame #26: Rf_eval + 1012 (0x100ef1d34 in libR.dylib) frame #27: do_eval + 1352 (0x100ef8ec8 in libR.dylib) frame #28: bcEval_loop + 27244 (0x100f068ec in libR.dylib) frame #29: bcEval + 592 (0x100ef23d0 in libR.dylib) frame #30: Rf_eval + 556 (0x100ef1b6c in libR.dylib) frame #31: forcePromise + 232 (0x100ef2668 in libR.dylib) frame #32: Rf_eval + 660 (0x100ef1bd4 in libR.dylib) frame #33: do_withVisible + 64 (0x100ef9200 in libR.dylib) frame #34: do_internal + 400 (0x100f58490 in libR.dylib) frame #35: bcEval_loop + 27828 (0x100f06b34 in libR.dylib) frame #36: bcEval + 592 (0x100ef23d0 in libR.dylib) frame #37: Rf_eval + 556 (0x100ef1b6c in libR.dylib) frame #38: forcePromise + 232 (0x100ef2668 in libR.dylib) frame #39: Rf_eval + 660 (0x100ef1bd4 in libR.dylib) frame #40: forcePromise + 232 (0x100ef2668 in libR.dylib) frame #41: getvar + 408 (0x100f15a18 in libR.dylib) frame #42: bcEval_loop + 16936 (0x100f040a8 in libR.dylib) frame #43: bcEval + 592 (0x100ef23d0 in libR.dylib) frame #44: Rf_eval + 556 (0x100ef1b6c in libR.dylib) frame #45: R_execClosure + 816 (0x100ef4730 in libR.dylib) frame #46: applyClosure_core + 164 (0x100ef3824 in libR.dylib) frame #47: Rf_eval + 1224 (0x100ef1e08 in libR.dylib) frame #48: do_eval + 1352 (0x100ef8ec8 in libR.dylib) frame #49: bcEval_loop + 27244 (0x100f068ec in libR.dylib) frame #50: bcEval + 592 (0x100ef23d0 in libR.dylib) frame #51: Rf_eval + 556 (0x100ef1b6c in libR.dylib) frame #52: R_execClosure + 816 (0x100ef4730 in libR.dylib) frame #53: applyClosure_core + 164 (0x100ef3824 in libR.dylib) frame #54: Rf_eval + 1224 (0x100ef1e08 in libR.dylib) frame #55: do_begin + 400 (0x100ef6fd0 in libR.dylib) frame #56: Rf_eval + 1012 (0x100ef1d34 in libR.dylib) frame #57: R_execClosure + 816 (0x100ef4730 in libR.dylib) frame #58: applyClosure_core + 164 (0x100ef3824 in libR.dylib) frame #59: Rf_eval + 1224 (0x100ef1e08 in libR.dylib) frame #60: do_docall + 628 (0x100e903b4 in libR.dylib) frame #61: bcEval_loop + 27244 (0x100f068ec in libR.dylib) frame #62: bcEval + 592 (0x100ef23d0 in libR.dylib) 
#> 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            passed operations: basic, cholesky, eigh, svd, qr, solve, matrix_exp
#>          model                                  operations
#> 1  Apple Metal basic,cholesky,eigh,svd,qr,solve,matrix_exp
#> 2 Apple M1 CPU basic,cholesky,eigh,svd,qr,solve,matrix_exp
```

## Principal subspace estimation

The leading two-dimensional eigenspace of a covariance matrix is a point
on a Grassmann manifold. Registering the covariance as problem data lets
the execution driver move it with the initial point.

``` r

X <- scale(as.matrix(iris[, 1:4]), center = TRUE, scale = FALSE)
C <- torch_tensor(crossprod(X) / nrow(X), dtype = torch_float64())
G <- manifold.grassmann(4, 2)
pca_problem <- riem.problem(G,
  function(Q, data) -torch_trace(Q$t()$matmul(data)$matmul(Q)), data = C)
pca_fit <- riem.optimize(pca_problem, riem.random(G), "conjugate_gradient",
  control = list(max_iterations = 15))
pca_fit$objective
#> [1] -4.315626
pca_fit$constraint_residuals
#> [1] 3.398242e-16
```

## Low-rank matrix completion

The mask and observations are ordinary registered tensors. Changing the
loss does not require changing the fixed-rank geometry or solver.

``` r

truth <- torch_tensor(outer(1:5, c(1, -1, 2)) + outer(c(1, 0, -1, 2, 1),
  c(2, 1, -1)), dtype = torch_float64())
mask <- torch_tensor(matrix(c(1,1,0, 1,0,1, 0,1,1, 1,1,0, 1,0,1), 5, 3,
  byrow = TRUE), dtype = torch_float64())
F <- manifold.fixedrank(5, 3, 2)
completion <- riem.problem(F, function(Z, data) {
  observed <- data$mask * (Z - data$truth)
  torch_sum(observed^2) / 2
}, data = list(mask = mask, truth = truth))
completion_fit <- riem.optimize(completion, riem.random(F), "conjugate_gradient",
  control = list(max_iterations = 10))
completion_fit$constraint_residuals
#> [1] 1.645044e-15
```

## An affine-invariant SPD mean

The Fréchet mean minimizes a sum of squared affine-invariant distances.
This example uses only the general problem interface and SPD geometry.

``` r

S <- manifold.spd(2, "airm")
A <- torch_diag(torch_tensor(c(1, 4), dtype = torch_float64()))
B <- torch_diag(torch_tensor(c(9, 1), dtype = torch_float64()))
mean_problem <- riem.problem(S, function(X, data) {
  (riem.sqdist(S, X, data$A) + riem.sqdist(S, X, data$B)) / 2
}, data = list(A = A, B = B))
mean_fit <- riem.optimize(mean_problem, riem.random(S), "conjugate_gradient",
  control = list(max_iterations = 12))
mean_fit$point
#> torch_tensor
#>  3.0000e+00 -9.3832e-07
#> -9.3832e-07  2.0000e+00
#> [ CPUDoubleType{2,2} ]
```

## Orthogonality-constrained torch training

[`riem.train()`](https://www.kisungyou.com/riemtorch/reference/riem.train.md)
moves the module before constructing the optimizer, transfers each
minibatch, and retains the actual parameter objects used by the
optimizer. Here the learned linear map remains on a Stiefel manifold
after every update.

``` r

inputs <- torch_randn(12, 4, dtype = torch_float64())
teacher <- torch_eye(4, dtype = torch_float64())[, 1:2]
responses <- inputs$matmul(teacher)
loader <- dataloader(tensor_dataset(inputs, responses), batch_size = 4)
frame_geometry <- manifold.stiefel(4, 2, "euclidean")
orthogonal_module <- nn_module("orthogonal_module",
  initialize = function() {
    self$frame <- nn_parameter(riem.random(frame_geometry, device = "cpu"))
  },
  forward = function(x) x$matmul(self$frame))
trained <- riem.train(orthogonal_module(), loader,
  loss = function(model, batch)
    (model(batch[[1]]) - batch[[2]])$square()$mean(),
  optimizer_factory = function(model)
    optim_rsgd(list(model$frame), lr = 0.05, manifold = frame_geometry),
  epochs = 2)
riem.belongs(frame_geometry, trained$model$frame)
#> [1] TRUE
trained$execution
#> $requested
#> [1] "auto"
#> 
#> $source
#> [1] "default"
#> 
#> $device
#> [1] "cpu"
#> 
#> $dtype
#> [1] "float64"
#> 
#> $reason
#> [1] "automatic selection: passed operations: basic, cholesky, eigh, svd, qr, solve, matrix_exp"
#> 
#> $model
#> [1] "Apple M1 CPU"
#> 
#> $operations
#> [1] "basic"      "cholesky"   "eigh"       "svd"        "qr"        
#> [6] "solve"      "matrix_exp"
#> 
#> $R
#> [1] "4.5.2"
#> 
#> $torch
#> [1] "0.17.0"
#> 
#> $libtorch
#> [1] "2.8.0"
#> 
#> $cuda_runtime
#> [1] NA
#> 
#> $cuda_visible_devices
#> [1] NA
```

The selected device and dtype are stored in every managed result. For a
forced CPU run on a GPU workstation, specify the override on the run
itself:

``` r

cpu_fit <- riem.optimize(pca_problem, riem.random(G, device = "cpu"),
  "steepest_descent", control = list(max_iterations = 2), device = "cpu")
cpu_fit$execution
#> $requested
#> [1] "cpu"
#> 
#> $source
#> [1] "argument"
#> 
#> $device
#> [1] "cpu"
#> 
#> $dtype
#> [1] "float64"
#> 
#> $reason
#> [1] "explicit selection: passed operations: basic, cholesky, eigh, matrix_exp, qr, solve, svd"
#> 
#> $model
#> [1] "Apple M1 CPU"
#> 
#> $operations
#> [1] "basic"      "cholesky"   "eigh"       "matrix_exp" "qr"        
#> [6] "solve"      "svd"       
#> 
#> $R
#> [1] "4.5.2"
#> 
#> $torch
#> [1] "0.17.0"
#> 
#> $libtorch
#> [1] "2.8.0"
#> 
#> $cuda_runtime
#> [1] NA
#> 
#> $cuda_visible_devices
#> [1] NA
```
