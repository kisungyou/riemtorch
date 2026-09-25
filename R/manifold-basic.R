#' Euclidean Tensor Geometry
#' @param shape Positive integer point dimensions.
#' @param field Real (default) or complex scalar field.
#' @return A `riem_manifold` with the Frobenius metric.
#' @examples
#' M <- manifold.euclidean(c(2, 3))
#' M
#' @export
manifold.euclidean <- function(shape,field=c("real","complex")) {
  if(match.arg(field)=="complex") return(.complex_euclidean(shape))
  shape <- .positive_integer(shape, "shape")
  riem.manifold("euclidean", shape, prod(shape), "euclidean", list(
    belongs = function(x, tol) TRUE, tangent = function(x, u) u,
    inner = function(x, u, v) .dot(u, v), egrad2rgrad = function(x, u) u,
    retr = function(x, u) x + u, exp = function(x, u) x + u,
    log = function(x, y) y - x, sqdist = function(x, y) .dot(y-x, y-x),
    project = function(x) x, transport = function(x, u, y, v) v,
    ehess2rhess = function(x, u, egrad, ehess) ehess,
    residual = function(x) 0),
    list(hessian = TRUE, curve_derivative = TRUE, snapshot_transport=TRUE,isometric_transport = TRUE),
    list(shape = shape), primitive_derivatives=setNames(rep(2L,9),c("tangent","inner","egrad2rgrad","retr","transport","exp","log","sqdist","project")), required_operations=c("basic"), second_order_retraction=TRUE)
}
.acos_sq <- function(c) {
  # Feasible unit vectors can produce inner products just outside [-1, 1]
  # through roundoff.  Clamp before both the local series and acos so that a
  # self-distance remains exactly nonnegative.
  c <- c$clamp(-1, 1)
  z <- -c + 1
  near <- z$abs() < 1e-4
  safe <- torch::torch_where(near, c * 0, c$clamp(-1 + 1e-15, 1 - 1e-15))
  torch::torch_where(near, z*2 + z^2/3 + z^3*4/45 + z^4/35, torch::torch_acos(safe)^2)
}
.sphere_exp <- function(x, u) {
  n2 <- .dot(u, u); small <- n2 < 1e-8
  safe <- torch::torch_where(small, torch::torch_ones_like(n2), n2)$sqrt()
  co <- torch::torch_where(small, -n2/2 + n2^2/24 + 1, safe$cos())
  si <- torch::torch_where(small, -n2/6 + n2^2/120 + 1, safe$sin()/safe)
  x * co + u * si
}
.sphere_log <- function(x, y) {
  c <- .dot(x, y)
  if (.scalar(c) < -1 + 1e-10) .stop("Sphere logarithm is undefined at antipodes")
  c <- c$clamp(-1, 1)
  z <- -c + 1; near <- z$abs() < 1e-5
  safe <- torch::torch_where(near, c * 0, c$clamp(-1+1e-15, 1-1e-15))
  ratio <- torch::torch_where(near, z/3 + z^2*2/15 + 1,
                              torch::torch_acos(safe)/(-safe^2+1)$sqrt())
  (y - x * c) * ratio
}
#' Round Sphere Geometry
#' @param field Real (default) or complex scalar field.
#' @param p Ambient dimension; `sphere(3)` represents the two-dimensional sphere.
#' @return A manifold with unit vector points, Euclidean tangents, and round metric.
#' @details Retraction is normalization, transport is tangent projection. Exact
#' exponential and local logarithm are available. The intrinsic Hessian includes
#' the sphere connection correction.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.sphere(3)
#'   x <- riem.random(M)
#'   riem.belongs(M, x)
#' }
#' @export
manifold.sphere <- function(p,field=c("real","complex")) {
  if(match.arg(field)=="complex") return(.complex_sphere(p))
  p <- .positive_integer(p, "p"); if (length(p) != 1 || p < 2) .stop("p must be at least 2")
  tangent <- function(x, u) u - x * .dot(x, u)
  riem.manifold("sphere", p, p-1, "round", list(
    belongs = function(x, tol) abs(.scalar(.dot(x,x))-1) <= tol,
    tangent = tangent, inner = function(x,u,v) .dot(u,v), egrad2rgrad = tangent,
    retr = function(x,u) .normalize(x+u), project = .normalize,
    exp = .sphere_exp, log = .sphere_log,
    sqdist = function(x,y) .acos_sq(.dot(x,y)),
    ehess2rhess = function(x,u,egrad,ehess) tangent(x,ehess) - .dot(x,egrad)*u,
    residual = function(x) abs(.scalar(.dot(x,x))-1)),
    list(hessian = TRUE, curve_derivative = TRUE), list(p=p), primitive_derivatives=setNames(rep(2L,9),c("tangent","inner","egrad2rgrad","retr","transport","exp","log","sqdist","project")), required_operations=c("basic"), second_order_retraction=TRUE)
}
#' Weighted Product Geometry
#' @param factors Nonempty, uniquely named list of manifold specifications.
#' @param weights Positive fixed metric weights; defaults to one per factor.
#' @return A `riem_product`, also inheriting from `riem_manifold`.
#' @details Points, tangents and gradients are identically named lists, including
#' nested products. Gradient conversion divides each factor by its metric weight.
#' All leaves of a solve must share device and dtype. Batches are never treated
#' as product factors.
#' @examples
#' M <- manifold.product(list(position = manifold.euclidean(2),
#'                            direction = manifold.sphere(3)), c(2, 5))
#' M
#' @export
manifold.product <- function(factors, weights = NULL) {
  if (!is.list(factors) || !length(factors) || is.null(names(factors)) ||
      any(!nzchar(names(factors))) || anyDuplicated(names(factors)))
    .stop("factors must be a nonempty, uniquely named list")
  lapply(factors, .check_manifold)
  if (is.null(weights)) weights <- rep(1, length(factors))
  if (!is.numeric(weights) || length(weights) != length(factors) || any(!is.finite(weights) | weights <= 0))
    .stop("weights must be finite, positive, and match factors")
  cap <- function(k) all(vapply(factors, function(M) isTRUE(M$capabilities[[k]]), logical(1)))
  structure(list(name = "product", metric = "weighted_product", factors = factors,
    weights = weights, shape = NULL, dimension = sum(vapply(factors, `[[`, numeric(1), "dimension")),
    capabilities = list(hessian=cap("hessian"), curve_derivative=cap("curve_derivative"),
                        snapshot_transport=cap("snapshot_transport"),isometric_transport=cap("isometric_transport"),second_order_retraction=cap("second_order_retraction"),
                        required_operations=unique(unlist(lapply(factors,function(M) .required_operations(M))))),
    specification = list(weights=weights,field=if(all(vapply(factors,function(M) identical(M$specification$field,"complex"),logical(1)))) "complex" else NULL)), class=c("riem_product", "riem_manifold"))
}
