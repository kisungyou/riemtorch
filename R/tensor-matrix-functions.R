.derivative_context <- new.env(parent = emptyenv())
.derivative_context$second_order <- FALSE
.spectral <- function(x, type) {
  ev <- torch::linalg_eigh(.sym(x)); d <- ev[[1L]]; Q <- ev[[2L]]
  if (type != "exp" && .scalar(d$min()) <= 0) .stop("Matrix function requires a positive-definite matrix")
  f <- switch(type, log = torch::torch_log(d), sqrt = torch::torch_sqrt(d),
              invsqrt = torch::torch_rsqrt(d), exp = torch::torch_exp(d))
  list(value = (Q * f$unsqueeze(-2))$matmul(.mt(Q)), d = d, Q = Q)
}
.loewner <- function(d, type) {
  a <- d$unsqueeze(-1); b <- d$unsqueeze(-2)
  if (type == "sqrt") return(1 / (a$sqrt() + b$sqrt()))
  if (type == "invsqrt") return(-1 / (a$sqrt() * b$sqrt() * (a$sqrt() + b$sqrt())))
  delta <- a - b
  if (type == "log") {
    z <- delta / (a + b); small <- z$abs() < 1e-4
    safe_delta <- torch::torch_where(small, torch::torch_ones_like(delta), delta)
    return(torch::torch_where(small, (z^2/3 + z^4/5 + z^6/7 + 1)*2/(a+b),
                             (a$log()-b$log())/safe_delta))
  }
  gap <- delta$abs()
  small <- gap < 1e-5
  safe <- torch::torch_where(small, torch::torch_ones_like(gap), gap)
  top <- torch::torch_maximum(a,b)
  top$exp() * torch::torch_where(small, -gap/2 + gap^2/6 - gap^3/24 + 1,
                               -torch::torch_expm1(-gap)/safe)
}
.matrix_autograd <- function(x, type) {
  fun <- torch::autograd_function(
    forward = function(ctx, input) {
      e <- .spectral(input, type)
      ctx$save_for_backward(d = e$d, Q = e$Q)
      e$value
    },
    backward = function(ctx, grad_output) {
      s <- ctx$saved_variables
      qt <- .mt(s$Q)
      g <- .sym(s$Q$matmul(.loewner(s$d, type) *
                        qt$matmul(.sym(grad_output))$matmul(s$Q))$matmul(qt))
      list(input = g)
    }
  )
  fun(x)
}

#' Stable Symmetric Matrix Functions
#'
#' Matrix logarithm, exponential, square root and inverse square root, and
#' their Frechet differentials. Point axes are the last two dimensions.
#' @param x A symmetric tensor, positive definite except for `type = "exp"`.
#' @param type One of `"log"`, `"exp"`, `"sqrt"`, or `"invsqrt"`.
#' @param u A symmetric perturbation with the same shape as `x`.
#' @return A tensor with the same shape, dtype and device as `x`.
#' @details Logarithm and roots use a spectral forward evaluation and a
#' divided-difference backward rule with continuous repeated-eigenvalue limits.
#' Their custom backward supports first derivatives only; double backward is
#' not a supported operation. The Frechet helper is a value-level differential,
#' not a differentiable replacement for a second derivative. Exponential uses
#' torch's native matrix exponential, including its higher derivatives.
#' @examples
#' if (torch::torch_is_installed()) {
#'   x <- torch::torch_eye(2, dtype = torch::torch_float64())
#'   riem.matrix.function(x, "log")
#'   riem.matrix.frechet(x, x, "log")
#' }
#' @export
riem.matrix.function <- function(x, type = c("log", "exp", "sqrt", "invsqrt")) {
  type <- match.arg(type)
  if (.derivative_context$second_order && type != "exp") .stop("Matrix log/root kernels support first derivatives only; supply an analytic rhess or residual differential")
  if (!inherits(x, "torch_tensor") || length(x$shape) < 2L ||
      tail(x$shape, 1) != tail(x$shape, 2)[1]) .stop("x must have square matrix point axes")
  if (!.finite(x) || .scalar((x - .mt(x))$abs()$max()) > 1e-7)
    .stop("x must be finite and symmetric")
  if (type == "exp") torch::torch_matrix_exp(.sym(x)) else .matrix_autograd(x, type)
}

#' @rdname riem.matrix.function
#' @export
riem.matrix.frechet <- function(x, u, type = c("log", "exp", "sqrt", "invsqrt")) {
  type <- match.arg(type)
  if (!inherits(x, "torch_tensor") || !inherits(u, "torch_tensor") ||
      !identical(as.integer(x$shape), as.integer(u$shape)) ||
      length(x$shape) < 2L || tail(x$shape, 1) != tail(x$shape, 2)[1])
    .stop("x and u must have matching square matrix point axes")
  if (!(x$dtype == u$dtype) || !identical(x$device$type, u$device$type) ||
      !identical(x$device$index, u$device$index))
    .stop("x and u must have matching dtype and device")
  e <- .spectral(x, type); qt <- .mt(e$Q)
  .sym(e$Q$matmul(.loewner(e$d, type) *
    qt$matmul(.sym(u))$matmul(e$Q))$matmul(qt))
}
.sylvester <- function(x, u, rank = NULL) {
  e <- torch::linalg_eigh(.sym(x)); d <- e[[1]]; Q <- e[[2]]
  den <- d$unsqueeze(-1) + d$unsqueeze(-2)
  if (!is.null(rank)) {
    p <- tail(as.integer(d$shape), 1L)
    active <- torch::torch_arange(1, p, dtype = d$dtype, device = d$device) > (p - rank)
    keep <- active$unsqueeze(-1) | active$unsqueeze(-2)
    den <- torch::torch_where(keep, den, torch::torch_ones_like(den))
  }
  z <- .mt(Q)$matmul(u)$matmul(Q) / den
  if (!is.null(rank)) z <- z * keep$to(dtype = z$dtype)
  Q$matmul(z)$matmul(.mt(Q))
}
