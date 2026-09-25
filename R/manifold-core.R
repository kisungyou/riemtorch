#' Define a Manifold Through Public Geometry Contracts
#'
#' Create an extensible geometry specification. Kernels operate on one point;
#' the public operations apply them separately to leading batch dimensions.
#' @param name Descriptive geometry name.
#' @param shape Positive integer vector of point dimensions.
#' @param dimension Intrinsic dimension (a nonnegative integer).
#' @param metric Metric identifier.
#' @param operations Named list of functions. Required: `belongs(x, tol)`,
#'   `tangent(x, u)`, `inner(x, u, v)`, `egrad2rgrad(x, u)`, `retr(x, u)`.
#'   Optional: `transport(x, u, y, v)`, `project(x)`, `random(x)`,
#'   `residual(x)`, `exp(x, u)`, `log(x, y)`, `sqdist(x, y)`,
#'   `ehess2rhess(x, u, egrad, ehess)`. Here `u` in transport is the actual
#'   retraction step, and `x` in random is a zero tensor with requested placement.
#' @param capabilities Named list describing optional derivative capabilities.
#'   `hessian` declares exact ambient conversion, `curve_derivative` declares
#'   first derivatives of the retraction, and `isometric_transport` declares
#'   metric preservation. `snapshot_transport` declares that transport accepts
#'   an endpoint pair identified by a local logarithm, even when it is not a
#'   retraction inverse. Custom explicit transports default to FALSE; the
#'   default endpoint tangent projection is safe. Other defaults are conservative.
#' @param specification Named serializable list of geometry configuration.
#' @param batch_operations Optional named list of kernels that operate directly
#'   on leading batch axes. Missing kernels use the pointwise fallback. Built-in
#'   geometries provide native batch kernels for their common operations.
#' @param primitive_derivatives Named integer orders (0, 1 or 2) verified for each
#'   primitive. Omitted entries conservatively declare no derivative support.
#' @param required_operations Optional device probe operations; NULL uses all.
#' @param second_order_retraction Whether the retraction has verified order two.
#' @return An S3 `riem_manifold` object.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- riem.manifold("line", 1, 1, "euclidean", list(
#'     belongs = function(x, tol) TRUE,
#'     tangent = function(x, u) u,
#'     inner = function(x, u, v) torch::torch_sum(u * v),
#'     egrad2rgrad = function(x, u) u,
#'     retr = function(x, u) x + u))
#'   x <- torch::torch_tensor(2, dtype = torch::torch_float64())
#'   riem.retr(M, x, -x)
#' }
#' @export
riem.manifold <- function(name, shape, dimension, metric, operations,
                          capabilities = list(), specification = list(),
                          batch_operations = list(), primitive_derivatives = integer(),
                          required_operations = NULL, second_order_retraction = FALSE) {
  shape <- .positive_integer(shape, "shape")
  .number(dimension, "dimension", strict = FALSE)
  if (dimension != floor(dimension)) .stop("dimension must be an integer")
  required <- c("belongs", "tangent", "inner", "egrad2rgrad", "retr")
  if (!all(vapply(operations[required], is.function, logical(1))))
    .stop("Missing required geometry kernels: ", paste(required, collapse = ", "))
  if (!is.list(batch_operations) ||
      (length(batch_operations) && (is.null(names(batch_operations)) ||
       any(!vapply(batch_operations, is.function, logical(1))))))
    .stop("batch_operations must be a named list of functions")
  capabilities <- utils::modifyList(list(hessian = FALSE, curve_derivative = FALSE,
                                         isometric_transport = FALSE, snapshot_transport=is.null(operations$transport)), capabilities)
  if(length(primitive_derivatives) && (is.null(names(primitive_derivatives)) ||
     any(!primitive_derivatives %in% 0:2))) .stop("primitive_derivatives must be named orders from 0 to 2")
  capabilities$derivatives <- primitive_derivatives
  capabilities$required_operations <- required_operations
  capabilities$second_order_retraction <- isTRUE(second_order_retraction)
  structure(list(name = name, shape = shape, dimension = dimension, metric = metric,
                 operations = operations, capabilities = capabilities,
                 specification = specification,
                 batch_operations = batch_operations), class = "riem_manifold")
}
.check_manifold <- function(M) {
  if (!inherits(M, "riem_manifold")) .stop("Expected a riem_manifold object")
}
.product_check <- function(M, x) {
  if (!is.list(x) || !identical(names(x), names(M$factors)))
    .stop("Product points must have exactly the named factors, in constructor order")
}
.geom <- function(M, operation, ...) {
  .check_manifold(M)
  op <- M$operations[[operation]]
  if (is.null(op)) .stop(operation, " is unsupported for ", M$name, " / ", M$metric)
  args <- list(...); x <- args[[1L]]
  if(inherits(M,"riem_svd")) return(do.call(op,args))
  if (inherits(x, "torch_tensor") && length(x$shape) > length(M$shape)) {
    batch <- utils::modifyList(.builtin_batch_operations(M), M$batch_operations)
    if (is.function(batch[[operation]])) return(.batch_geom(M, batch[[operation]], args))
  }
  .pointwise(M, op, ...)
}
#' Inspect and Operate on Manifold Points
#'
#' Metric-aware tensor operations. No operation silently repairs input points.
#' `riem.project()` is the explicit repair boundary. Tangents use the same point
#' axes and representation as points. Batch axes must match exactly.
#' @param manifold A geometry specification.
#' @param x,y Points (tensors or named product lists).
#' @param u,v Tangent vectors; `u` is an ambient vector for projection or
#'   gradient conversion.
#' @param step Scalar multiplier of a retraction/transport step.
#' @param tol Nonnegative membership/tangency tolerance. `NULL` chooses a
#'   precision-aware default: 1e-7 for float64 and 1e-5 for float32.
#' @param egrad,ehess Ambient gradient and ambient Hessian-vector product.
#' @return `belongs` and `istangent` return logical values, one per batch point.
#'   `inner`, `norm`, `dist` and `sqdist` return tensors with point axes reduced.
#'   Other operations return tensors or named product lists of tensors.
#' @details Transport receives the path `(x, step * u, y)` and vector `v`.
#'   Exact exponential, logarithm, distance and Hessian conversion are optional;
#'   unsupported operations fail explicitly. Sphere logarithms reject antipodes.
#'   Distance itself is not differentiable on the diagonal; use squared distance
#'   when an objective needs a derivative there.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.sphere(3)
#'   x <- torch::torch_tensor(c(1, 0, 0), dtype = torch::torch_float64())
#'   u <- torch::torch_tensor(c(0, 0.2, 0), dtype = x$dtype)
#'   y <- riem.retr(M, x, u)
#'   riem.belongs(M, y)
#'   riem.inner(M, x, u, u)
#'   riem.transport(M, x, u, y, u)
#'   riem.sqdist(M, x, y)
#' }
#' @export
riem.belongs <- function(manifold, x, tol = NULL) {
  .check_manifold(manifold)
  if(is.null(tol)) tol<-.default_tolerance(x)
  .number(tol, "tol", strict = FALSE)
  if (inherits(manifold, "riem_product")) {
    .product_check(manifold, x)
    return(all(unlist(Map(riem.belongs, manifold$factors, x, MoreArgs = list(tol = tol)))))
  }
  if(inherits(manifold,"riem_svd")) return(.svd_shape(manifold,x)&&.allfinite(x)&&manifold$operations$belongs(x,tol))
  if (!.shape_ok(manifold, x) || !.finite(x)) return(FALSE)
  .pointwise(manifold, function(z) manifold$operations$belongs(z, tol), x)
}
#' @rdname riem.belongs
#' @export
riem.tangent <- function(manifold, x, u) {
  if (inherits(manifold, "riem_product")) {
    .product_check(manifold, x); .product_check(manifold, u)
    return(Map(riem.tangent, manifold$factors, x, u))
  }
  .geom(manifold, "tangent", x, u)
}
#' @rdname riem.belongs
#' @export
riem.istangent <- function(manifold, x, u, tol = NULL) {
  if(is.null(tol)) tol<-.default_tolerance(x)
  .number(tol,"tol",strict=FALSE)
  if (inherits(manifold, "riem_product")) {
    .product_check(manifold, x); .product_check(manifold, u)
    return(all(unlist(Map(riem.istangent, manifold$factors, x, u, MoreArgs = list(tol = tol)))))
  }
  if(inherits(manifold,"riem_svd")) {
    if(!.svd_shape(manifold,u)||!.allfinite(u)) return(FALSE)
    delta<-.add(manifold$operations$tangent(x,u),u,-1)
    return(.scalar(.tree_dot(delta,delta)$sqrt())<=tol*(1+.scalar(.tree_dot(u,u)$sqrt())))
  }
  if (!.shape_ok(manifold, u) || !.finite(u)) return(FALSE)
  .pointwise(manifold, function(a, b) {
    .scalar(.fnorm(manifold$operations$tangent(a, b) - b)) <= tol * (1 + .scalar(.fnorm(b)))
  }, x, u)
}
#' @rdname riem.belongs
#' @export
riem.inner <- function(manifold, x, u, v) {
  if (inherits(manifold, "riem_product")) {
    lapply(list(x, u, v), function(z) .product_check(manifold, z))
    terms <- Map(riem.inner, manifold$factors, x, u, v)
    return(Reduce(`+`, Map(`*`, terms, manifold$weights)))
  }
  .geom(manifold, "inner", x, u, v)
}
#' @rdname riem.belongs
#' @export
riem.norm <- function(manifold, x, u) riem.inner(manifold, x, u, u)$clamp_min(0)$sqrt()
#' @rdname riem.belongs
#' @export
riem.egrad2rgrad <- function(manifold, x, u) {
  if (inherits(manifold, "riem_product")) {
    .product_check(manifold, x); .product_check(manifold, u)
    return(Map(function(M, a, b, w) .scale(riem.egrad2rgrad(M, a, b), 1/w),
               manifold$factors, x, u, manifold$weights))
  }
  .geom(manifold, "egrad2rgrad", x, u)
}
#' @rdname riem.belongs
#' @export
riem.retr <- function(manifold, x, u, step = 1) {
  if (inherits(manifold, "riem_product")) {
    .product_check(manifold, x); .product_check(manifold, u)
    return(Map(riem.retr, manifold$factors, x, u, MoreArgs = list(step = step)))
  }
  .geom(manifold, "retr", x, .scale(u, step))
}
#' @rdname riem.belongs
#' @export
riem.transport <- function(manifold, x, u, y, v, step = 1) {
  if (inherits(manifold, "riem_product")) {
    lapply(list(x, u, y, v), function(z) .product_check(manifold, z))
    return(Map(riem.transport, manifold$factors, x, u, y, v, MoreArgs = list(step = step)))
  }
  if (is.null(manifold$operations$transport)) return(riem.tangent(manifold, y, v))
  .geom(manifold, "transport", x, .scale(u, step), y, v)
}
#' @rdname riem.belongs
#' @export
riem.project <- function(manifold, x) {
  if (inherits(manifold, "riem_product")) {
    .product_check(manifold, x)
    return(Map(riem.project, manifold$factors, x))
  }
  .geom(manifold, "project", x)
}
#' @rdname riem.belongs
#' @export
riem.exp <- function(manifold, x, u, step = 1) {
  if (inherits(manifold, "riem_product")) {
    .product_check(manifold, x); .product_check(manifold, u)
    return(Map(riem.exp, manifold$factors, x, u, MoreArgs = list(step = step)))
  }
  .geom(manifold, "exp", x, .scale(u, step))
}
#' @rdname riem.belongs
#' @export
riem.log <- function(manifold, x, y) {
  if (inherits(manifold, "riem_product")) {
    .product_check(manifold, x); .product_check(manifold, y)
    return(Map(riem.log, manifold$factors, x, y))
  }
  .geom(manifold, "log", x, y)
}
#' @rdname riem.belongs
#' @export
riem.sqdist <- function(manifold, x, y) {
  if (inherits(manifold, "riem_product")) {
    .product_check(manifold, x); .product_check(manifold, y)
    return(Reduce(`+`, Map(`*`, Map(riem.sqdist, manifold$factors, x, y), manifold$weights)))
  }
  .geom(manifold, "sqdist", x, y)
}
#' @rdname riem.belongs
#' @export
riem.dist <- function(manifold, x, y) riem.sqdist(manifold, x, y)$clamp_min(0)$sqrt()
#' @rdname riem.belongs
#' @export
riem.ehess2rhess <- function(manifold, x, u, egrad, ehess) {
  if (inherits(manifold, "riem_product")) {
    lapply(list(x, u, egrad, ehess), function(z) .product_check(manifold, z))
    return(Map(function(M, a, b, g, h, w) .scale(riem.ehess2rhess(M, a, b, g, h), 1/w),
               manifold$factors, x, u, egrad, ehess, manifold$weights))
  }
  if (!isTRUE(manifold$capabilities$hessian)) .stop("Exact Hessian conversion is unsupported for ", manifold$name)
  .geom(manifold, "ehess2rhess", x, u, egrad, ehess)
}
#' Generate Feasible Initial Points
#' @param manifold A geometry specification.
#' @param batch_shape Optional positive integer batch dimensions.
#' @param dtype Optional floating-point torch dtype or supported dtype name.
#'   The default is the dtype of device-bound manifold state, or float64.
#' @param device Device request. The default follows [riem.device()], which
#'   automatically selects a compatible visible accelerator before CPU.
#' @return A tensor or named product. Random generation uses torch's RNG.
#' @details Most geometries project a Gaussian draw. This is an initialization
#'   law, not a claim of uniformity. Sphere and orthogonal frames are isotropic.
#' @examples
#' if (torch::torch_is_installed()) {
#'   torch::torch_manual_seed(1)
#'   riem.random(manifold.sphere(3))
#' }
#' @export
riem.random <- function(manifold, batch_shape = integer(), dtype = NULL, device = NULL) {
  if (length(batch_shape)) batch_shape <- .positive_integer(batch_shape, "batch_shape")
  if(is.null(dtype)&&identical(manifold$specification$field,"complex")) dtype<-"complex128"
  execution <- .resolve_execution(device = device, dtype = dtype,
                                  reference = .placement_reference(manifold),
                                  operations=.required_operations(manifold))
  manifold <- .to_execution(manifold, execution)
  .random_execution(manifold, batch_shape, execution)
}
.random_execution <- function(manifold, batch_shape, execution) {
  if (inherits(manifold, "riem_product"))
    return(lapply(manifold$factors, .random_execution,
      batch_shape = batch_shape, execution = execution))
  if(inherits(manifold,"riem_svd")) {
    if(length(batch_shape)) .stop("Compact fixed-rank points do not support independent batches")
    return(manifold$operations$random(torch::torch_zeros(1,dtype=execution$dtype,device=execution$device)))
  }
  template <- torch::torch_zeros(c(batch_shape, manifold$shape),
    dtype = execution$dtype, device = execution$device)
  if (!is.null(manifold$operations$random)) return(.geom(manifold, "random", template))
  riem.project(manifold, torch::torch_randn_like(template))
}
#' @export
print.riem_manifold <- function(x, ...) {
  cat("<riem_manifold>", x$name, " | metric:", x$metric, " | dimension:", x$dimension, "\n")
  invisible(x)
}
