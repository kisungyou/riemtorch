.diagnostic <- function(checks, curves=NULL, details=list()) {
  status <- if(any(checks$status=="fail")) "fail" else
    if(any(checks$status=="inconclusive")) "inconclusive" else "pass"
  structure(list(status=status,checks=checks,curves=curves,details=details),class="riem_check")
}
.check_row <- function(check,error,tolerance,note="") data.frame(
  check=check,error=error,tolerance=tolerance,
  status=if(!is.finite(error)) "inconclusive" else if(error<=tolerance) "pass" else "fail",
  note=note,stringsAsFactors=FALSE)
.check_direction <- function(M,x,u=NULL) {
  if(is.null(u)) u<-.search_noise(M,x)
  .check_tangent(M,x,u);n<-.nrm(M,x,u)
  if(n<=0) .stop("Diagnostic direction must be nonzero")
  .scale(u,1/n)
}
.diagnostic_steps <- function(x,steps) {
  if(is.null(steps)) steps<-10^seq(-1,if(.default_tolerance(x)>1e-7) -3 else -6,length.out=8)
  if(!is.numeric(steps)||length(steps)<3||any(!is.finite(steps)|steps<=0))
    .stop("steps must contain at least three positive values")
  sort(unique(steps),decreasing=TRUE)
}
.taylor_curve <- function(steps,errors) {
  slope<-c(NA_real_,diff(log(pmax(errors,.Machine$double.xmin)))/diff(log(steps)))
  data.frame(step=steps,error=errors,slope=slope)
}
#' Check a Manifold or User-Supplied Derivatives
#'
#' Numerical checks return evidence, not a proof of correctness. Directions
#' should be repeated at several interior points. Compact fixed-rank manifold
#' checks materialize dense matrices for the retraction derivative comparison. Unsupported Taylor tests are
#' reported as inconclusive, never silently replaced by another metric.
#' @param manifold A manifold specification.
#' @param x A feasible point. For the manifold check, NULL generates one.
#' @param u Optional nonzero tangent direction; otherwise generated randomly.
#' @param tolerance Relative error tolerance. NULL uses the point precision.
#' @param problem Scalar or least-squares problem.
#' @param steps Positive finite-difference step sizes; NULL uses a precision-aware grid.
#' @return A `riem_check` list with status, checks, curves and details.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.sphere(3)
#'   x <- riem.random(M, device = "cpu")
#'   riem.check.manifold(M, x)$checks
#'   P <- riem.problem(M, function(x) -x[1])
#'   riem.check.gradient(P, x)$status
#'   riem.check.hessian(P, x)$status
#'   L <- riem.problem.leastsquares(manifold.euclidean(2), function(x) x * 2)
#'   riem.check.adjoint(L, torch::torch_ones(2, dtype = torch::torch_float64()))$status
#' }
#' @export
riem.check.manifold <- function(manifold,x=NULL,u=NULL,tolerance=NULL) {
  M<-manifold;if(is.null(x)) x<-riem.random(M)
  .check_point(M,x)
  if(!all(riem.belongs(M,x))) .stop("x must be feasible")
  if(is.null(tolerance)) tolerance<-sqrt(.default_tolerance(x))
  .number(tolerance,"tolerance")
  u<-.check_direction(M,x,u);v<-.check_direction(M,x)
  a<-.tree_map(x,torch::torch_randn_like);g<-riem.egrad2rgrad(M,x,a)
  h<-if(.default_tolerance(x)>1e-7) 1e-3 else 1e-5
  y<-riem.retr(M,x,u,h)
  if(inherits(M,"riem_svd")) {
    chart<-list(U=.mt(.solve(.mt(x$S),.mt(u$U))),S=u$S,V=.mt(.solve(x$S,.mt(u$V))))
    ambient<-u$U$matmul(x$V$t())+x$U$matmul(u$S)$matmul(x$V$t())+x$U$matmul(u$V$t())
    dual<-abs(.scalar(.tree_dot(a,chart))-.ip(M,x,g,u))/(1+abs(.scalar(.tree_dot(a,chart))))
    derivative<-.scalar(.fnorm((riem.materialize(M,y)-riem.materialize(M,x))/h-ambient))
  } else {
    dual<-abs(.scalar(.tree_dot(a,u))-.ip(M,x,g,u))/(1+abs(.scalar(.tree_dot(a,u))))
    derivative<-.nrm(M,x,.add(.scale(.add(y,x,-1),1/h),u,-1))
  }
  checks<-rbind(.check_row("metric_duality",dual,tolerance),
    .check_row("metric_symmetry",abs(.ip(M,x,u,v)-.ip(M,x,v,u)),tolerance),
    .check_row("positive_metric",if(.ip(M,x,u,u)>0) 0 else 1,0),
    .check_row("tangent_membership",as.numeric(!all(riem.istangent(M,x,u))),0),
    .check_row("retraction_feasibility",as.numeric(!all(riem.belongs(M,y))),0),
    .check_row("retraction_derivative",derivative,max(tolerance,10*h)))
  if(isTRUE(M$capabilities$isometric_transport)) {
    w<-riem.transport(M,x,u,y,v,h)
    checks<-rbind(checks,.check_row("transport_isometry",abs(.ip(M,y,w,w)-.ip(M,x,v,v)),tolerance))
  }
  .diagnostic(checks,details=list(point=.clone(x),direction=u))
}
#' @rdname riem.check.manifold
#' @export
riem.check.gradient <- function(problem,x,u=NULL,steps=NULL,tolerance=NULL) {
  M<-problem$manifold;ev<-riem.evaluate(problem,x);u<-.check_direction(M,x,u)
  steps<-.diagnostic_steps(x,steps)
  if(is.null(tolerance)) tolerance<-sqrt(.default_tolerance(x))
  .number(tolerance,"tolerance")
  f<-.scalar(ev$value);slope<-.ip(M,x,ev$gradient,u)
  errors<-central<-rep(NA_real_,length(steps))
  for(i in seq_along(steps)) tryCatch({
    t<-steps[i];fp<-.scalar(riem.evaluate(problem,riem.retr(M,x,u,t),FALSE)$value)
    fm<-.scalar(riem.evaluate(problem,riem.retr(M,x,u,-t),FALSE)$value)
    errors[i]<-abs(fp-f-t*slope)
    central[i]<-abs((fp-fm)/(2*t)-slope)/(1+abs(slope))
  },error=function(e) NULL)
  err<-if(any(is.finite(central))) min(central,na.rm=TRUE) else NA_real_
  .diagnostic(.check_row("directional_gradient",err,tolerance),
    .taylor_curve(steps,errors),list(directional_errors=central,direction=u))
}
#' @rdname riem.check.manifold
#' @export
riem.check.hessian <- function(problem,x,u=NULL,steps=NULL,tolerance=NULL) {
  M<-problem$manifold;u<-.check_direction(M,x,u);v<-.check_direction(M,x)
  if(is.null(tolerance)) tolerance<-sqrt(.default_tolerance(x))*5
  .number(tolerance,"tolerance")
  H<-function(a) riem.hessian(problem,x,a)
  hu<-H(u);hv<-H(v);hs<-H(.add(u,v))
  linear<-.nrm(M,x,.add(.add(hs,hu,-1),hv,-1))/(1+.nrm(M,x,hs))
  sym<-abs(.ip(M,x,u,hv)-.ip(M,x,v,hu))/(1+abs(.ip(M,x,u,hv)))
  checks<-rbind(.check_row("hessian_linearity",linear,tolerance),.check_row("hessian_symmetry",sym,tolerance))
  has_exp<-all(riem.capabilities(M)$value[riem.capabilities(M)$operation=="exp"])
  if(!has_exp&&!isTRUE(M$capabilities$second_order_retraction))
    return(.diagnostic(rbind(checks,.check_row("hessian_taylor",NA_real_,tolerance,
      "Requires an exponential map or verified second-order retraction"))))
  move<-if(has_exp) riem.exp else riem.retr
  ev<-riem.evaluate(problem,x);f<-.scalar(ev$value);d<-.ip(M,x,ev$gradient,u);q<-.ip(M,x,u,hu)
  steps<-.diagnostic_steps(x,steps);errors<-second<-rep(NA_real_,length(steps))
  for(i in seq_along(steps)) tryCatch({
    t<-steps[i];fp<-.scalar(riem.evaluate(problem,move(M,x,u,t),FALSE)$value)
    fm<-.scalar(riem.evaluate(problem,move(M,x,u,-t),FALSE)$value)
    errors[i]<-abs(fp-f-t*d-t*t*q/2)
    second[i]<-abs((fp+fm-2*f)/(t*t)-q)/(1+abs(q))
  },error=function(e) NULL)
  err<-if(any(is.finite(second))) min(second,na.rm=TRUE) else NA_real_
  .diagnostic(rbind(checks,.check_row("hessian_taylor",err,tolerance)),
    .taylor_curve(steps,errors),list(second_difference_errors=second))
}
#' @rdname riem.check.manifold
#' @export
riem.check.adjoint <- function(problem,x,u=NULL,tolerance=NULL) {
  if(!inherits(problem,"riem_leastsquares")) .stop("Expected a least-squares problem")
  M<-problem$manifold;.check_point(M,x);u<-.check_direction(M,x,u)
  if(is.null(tolerance)) tolerance<-sqrt(.default_tolerance(x))
  .number(tolerance,"tolerance");counts<-.new_counts()
  r<-.residual(problem,x,counts);v<-torch::torch_randn_like(r)
  left<-.scalar(.dot(.residual_jvp(problem,x,u,counts),v))
  right<-.ip(M,x,u,.residual_adjoint(problem,x,v,counts))
  .diagnostic(.check_row("metric_adjoint",abs(left-right)/(1+abs(left)+abs(right)),tolerance))
}
#' Estimate Extreme Riemannian Hessian Eigenvalues
#' @param problem A problem supporting exact Hessian-vector products.
#' @param x A feasible point.
#' @param iterations Maximum Lanczos steps, capped by intrinsic dimension.
#' @param tolerance Breakdown tolerance.
#' @return Ritz eigenvalues, tangent eigenvectors, residual norms, iteration
#'   count and an explicit approximation qualification. A small residual does
#'   not certify that an unseen more extreme eigenvalue does not exist.
#' @examples
#' if (torch::torch_is_installed()) {
#'   P <- riem.problem(manifold.euclidean(2), function(x) (x*x)$sum())
#'   x <- torch::torch_ones(2, dtype = torch::torch_float64())
#'   riem.hessian.spectrum(P, x)$values
#' }
#' @export
riem.hessian.spectrum <- function(problem,x,iterations=20L,tolerance=1e-10) {
  M<-problem$manifold;.check_point(M,x);.number(tolerance,"tolerance")
  iterations<-.positive_integer(iterations,"iterations")
  if(length(iterations)!=1||M$dimension<1) .stop("Require scalar iterations and positive dimension")
  basis<-list(.check_direction(M,x));hs<-list();kmax<-min(iterations,M$dimension)
  for(k in seq_len(kmax)) {
    hs[[k]]<-riem.hessian(problem,x,basis[[k]]);w<-.clone(hs[[k]])
    for(pass in 1:2) for(b in basis) w<-.add(w,b,-.ip(M,x,b,w))
    n<-.nrm(M,x,w)
    if(n<tolerance||k==kmax) break
    basis[[k+1L]]<-.scale(w,1/n)
  }
  n<-length(hs);A<-matrix(0,n,n)
  for(i in seq_len(n)) for(j in seq_len(n)) A[i,j]<-.ip(M,x,basis[[i]],hs[[j]])
  e<-eigen((A+t(A))/2,symmetric=TRUE);ids<-unique(c(n,1L));vectors<-residuals<-list()
  for(j in seq_along(ids)) {
    index<-ids[j];v<-.scale(basis[[1]],e$vectors[1,index])
    if(n>1) for(i in 2:n) v<-.add(v,basis[[i]],e$vectors[i,index])
    vectors[[j]]<-v;residuals[[j]]<-.nrm(M,x,.add(riem.hessian(problem,x,v),v,-e$values[index]))
  }
  list(values=e$values[ids],vectors=vectors,residuals=unlist(residuals),iterations=n,
       qualification="Ritz estimates from a sampled Krylov subspace; no global extremality certificate")
}
