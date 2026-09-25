.stop <- function(...) stop(..., call. = FALSE)
.scalar <- function(x) as.numeric(x$item())
.finite <- function(x) isTRUE(as.logical(torch::torch_isfinite(x)$all()$item()))
.default_tolerance <- function(x) {
  leaves <- .leaves(x)
  if (!length(leaves)) return(1e-7)
  reference <- leaves[[1L]]
  if (!inherits(reference, "torch_tensor") || !(reference$is_floating_point()||reference$is_complex()))
    return(1e-7)
  if (isTRUE(.real_dtype(reference$dtype) == torch::torch_float64())) return(1e-7)
  if (isTRUE(.real_dtype(reference$dtype) == torch::torch_float32())) return(1e-5)
  if (isTRUE(reference$dtype == torch::torch_float16())) return(5e-3)
  if (isTRUE(reference$dtype == torch::torch_bfloat16())) return(5e-2)
  1e-7
}
.positive_integer <- function(x, name = "dimension") {
  if (!is.numeric(x) || length(x) < 1L || anyNA(x) ||
      any(!is.finite(x) | x < 1 | x != floor(x)))
    .stop(name, " must contain positive integers")
  as.integer(x)
}
.number <- function(x, name, lower = 0, strict = TRUE) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
      (if (strict) x <= lower else x < lower)) .stop("Invalid ", name)
  x
}
.eye <- function(x, n = tail(x$shape, 1L))
  torch::torch_eye(n, dtype = x$dtype, device = x$device)
.mt <- function(x) x$transpose(-2, -1)
.sym <- function(x) (x + .mt(x)) / 2
.skew <- function(x) (x - .mt(x)) / 2
.dot <- function(x, y) {z<-torch::torch_sum(x$conj()*y);if(z$is_complex()) z$real else z}
.fnorm <- function(x) torch::torch_sqrt(.dot(x, x))
.diag <- function(x) torch::torch_diag(x)
.solve <- function(a, b) torch::linalg_solve(a, b)
.clone <- function(x, grad = FALSE) {
  if (is.list(x)) return(lapply(x, .clone, grad = grad))
  x$detach()$clone()$requires_grad_(grad)
}
.tree_map <- function(x, fn) {
  if (is.list(x)) lapply(x, .tree_map, fn = fn) else fn(x)
}
.tree_zip <- function(x, y, fn) {
  if (is.list(x)) {
    if (!is.list(y) || !identical(names(x), names(y)) || length(x) != length(y))
      .stop("Parameter trees must have identical named leaves")
    return(Map(function(a, b) .tree_zip(a, b, fn), x, y))
  }
  if (is.list(y)) .stop("Parameter trees must have identical named leaves")
  fn(x, y)
}
.scale <- function(x, a) .tree_map(x, function(z) z * a)
.add <- function(x, y, a = 1) .tree_zip(x, y, function(u, v) u + v * a)
.zeros <- function(x) .tree_map(x, torch::torch_zeros_like)
.leaves <- function(x) {
  if (!is.list(x)) return(list(x))
  unlist(lapply(x, .leaves), recursive = FALSE, use.names = FALSE)
}
.unflatten <- function(template, leaves) {
  index <- 0L
  .tree_map(template, function(x) { index <<- index + 1L; leaves[[index]] })
}
.tree_dot <- function(x, y) Reduce(`+`, .leaves(.tree_zip(x, y, .dot)))
.allfinite <- function(x) all(vapply(.leaves(x), .finite, logical(1)))
.shape_ok <- function(M, x, batch = TRUE) {
  if(inherits(M,"riem_svd")) return(.svd_shape(M,x))
  if (!inherits(x, "torch_tensor")) return(FALSE)
  s <- as.integer(x$shape); p <- length(M$shape)
  length(s) >= p && identical(tail(s, p), M$shape) && (batch || length(s) == p)
}
.check_tensor <- function(M, x, batch = TRUE) {
  if (!.shape_ok(M, x, batch)) .stop("Expected ", M$name, " point shape ",
    paste(M$shape, collapse = " x "), if (batch) " (optional leading batch axes)" else " without batch axes")
  if(inherits(M,"riem_svd")) return(invisible(x))
  if(!(x$is_floating_point()||x$is_complex())) .stop("Points must be floating-point or complex torch tensors")
  field<-if(!is.null(M$specification$field)) M$specification$field else "real"
  if(x$is_complex()&&field!="complex") .stop("Complex tensors require a complex manifold")
  if(!x$is_complex()&&field=="complex") .stop("Complex manifolds require complex tensor points")
  invisible(x)
}
.pointwise <- function(M, op, ...) {
  args <- list(...)
  x <- args[[1L]]; .check_tensor(M, x)
  s <- as.integer(x$shape); q <- length(M$shape)
  for (z in args) {
    .check_tensor(M, z)
    if (!identical(as.integer(z$shape), s)) .stop("Batch and point shapes must match exactly; broadcasting is not implicit")
    if (!(z$dtype == x$dtype) || !identical(z$device$type, x$device$type) ||
        !identical(z$device$index, x$device$index)) .stop("Tensor dtype and device must agree")
  }
  if (length(s) == q) return(do.call(op, args))
  batch <- head(s, -q); n <- prod(batch)
  flat <- lapply(args, function(z) z$reshape(c(n, M$shape)))
  out <- lapply(seq_len(n), function(i) do.call(op, lapply(flat, function(z) z$select(1, i))))
  if (is.logical(out[[1L]]) || is.numeric(out[[1L]])) return(array(unlist(out), dim = batch))
  torch::torch_stack(out, dim = 1)$reshape(c(batch, out[[1L]]$shape))
}
.polar <- function(x) x$matmul(riem.matrix.function(.mt(x)$matmul(x), "invsqrt"))
.normalize <- function(x) {
  n <- .fnorm(x)
  if (.scalar(n) <= 0) .stop("Cannot normalize a zero point")
  x / n
}
.jvp <- function(fn, x, u, create_graph = FALSE) {
  torch::with_enable_grad({
    z <- if (x$requires_grad) x else x$detach()$requires_grad_(TRUE)
    y <- fn(z)
    w <- torch::torch_zeros_like(y)$requires_grad_(TRUE)
    jt <- torch::autograd_grad(y, z, w, create_graph = TRUE)[[1L]]
    torch::autograd_grad(jt, w, u, create_graph = create_graph,
                         retain_graph = TRUE)[[1L]]
  })
}

.real_dtype <- function(dtype) {
  if(dtype==torch::torch_cdouble()) return(torch::torch_float64())
  if(dtype==torch::torch_cfloat()) return(torch::torch_float32())
  dtype
}
.adj <- function(x) x$transpose(-2,-1)$conj()
.herm <- function(x) (x+.adj(x))/2
