#' Least-Squares Objectives and Metric Adjoints
#' @param manifold Geometry specification.
#' @param residual Function `residual(x)` returning a one-dimensional tensor.
#' @param weights Optional nonnegative vector of diagonal residual weights or
#'   symmetric positive-semidefinite weight matrix, on the point's device/dtype.
#' @param jvp Optional residual differential callback `jvp(x, u)`.
#' @param adjoint Optional metric adjoint callback `adjoint(x, v)`. Both
#'   differential callbacks must be supplied together, or both omitted.
#' @param loss Robust residual loss: linear, huber, soft_l1 or cauchy.
#' @param loss_scale Positive residual scale.
#' @param preconditioner Optional tangent preconditioner; see [riem.problem()].
#' @param value_rgrad Optional fused full-objective callback; see [riem.problem()].
#' @param required_operations Device operations required by callbacks.
#' @param rhess Optional exact Hessian callback for the full objective.
#' @param data Optional registered data passed to callbacks as a named `data`
#'   argument.
#' @return A `riem_leastsquares` problem. Linear loss has objective 0.5 r' W r.
#'   Robust losses apply scale squared times rho((r/scale)^2) componentwise
#'   before diagonal weighting. The Gauss--Newton model uses nonnegative rho'
#'   weights and omits rho'' terms.
#' @details Without callbacks, reverse-mode differentiation computes the metric
#' adjoint, and reverse-over-reverse differentiation computes J u. This needs
#' differentiable backward rules for residual operations. GN and LM use J* W J,
#' which is explicitly an approximation to the objective's exact Hessian.
#' @examples
#' if (torch::torch_is_installed()) {
#'   P <- riem.problem.leastsquares(manifold.euclidean(2), function(x) x - 1)
#'   x <- torch::torch_zeros(2, dtype = torch::torch_float64())
#'   riem.optimize(P, x, "levenberg_marquardt")
#' }
#' @export
riem.problem.leastsquares <- function(manifold,residual,weights=NULL,jvp=NULL,
                                      adjoint=NULL,rhess=NULL,data=NULL,loss="linear",loss_scale=1,preconditioner=NULL,required_operations=NULL,value_rgrad=NULL) {
  if(!is.function(residual)) .stop("residual must be a function")
  if(xor(is.null(jvp),is.null(adjoint))) .stop("Supply both jvp and adjoint, or neither")
  if(!is.null(jvp)&&(!is.function(jvp)||!is.function(adjoint))) .stop("Differential callbacks must be functions")
  loss<-match.arg(loss,c("linear","huber","soft_l1","cauchy"));.number(loss_scale,"loss_scale")
  P<-riem.problem(manifold,function(x) NULL,rhess=rhess,data=data,preconditioner=preconditioner,required_operations=required_operations,value_rgrad=value_rgrad)
  P$loss<-loss;P$loss_scale<-loss_scale
  if(!is.null(weights)) {
    if(!inherits(weights,"torch_tensor")||!.finite(weights)||!(length(weights$shape)%in%c(1,2))) .stop("weights must be a finite vector or matrix tensor")
    if(length(weights$shape)==1) {if(.scalar(weights$min())<0) .stop("weights must be nonnegative")} else {
      if(weights$shape[1]!=weights$shape[2] || .scalar(.fnorm(weights-weights$t()))>1e-8 || .scalar(torch::linalg_eigvalsh(weights)$min())< -1e-10) .stop("weights must be symmetric positive semidefinite")
    }
    if(loss!="linear"&&length(weights$shape)==2) .stop("Robust losses support diagonal weights only")
    weights<-.clone(weights)
  }
  P$residual<-residual;P$weights<-weights;P$jvp<-jvp;P$adjoint<-adjoint
  class(P)<-c("riem_leastsquares",class(P));P
}
.residual<-function(P,x,counts) {
  .count(counts,"residual");r<-.problem_call(P,P$residual,x)
  if(!inherits(r,"torch_tensor")||length(r$shape)!=1||r$numel()<1) .stop("residual must return a nonempty vector tensor")
  ref<-.leaves(x)[[1]]
  if(!(r$dtype==ref$dtype)||!identical(r$device$type,ref$device$type)||
     !identical(r$device$index,ref$device$index))
    .stop("Residual placement must match points")
  r
}
.weight<-function(P,r) {
  W<-P$weights;if(is.null(W)) return(r)
  if(!(W$dtype==r$dtype)||!identical(W$device$type,r$device$type)||
     !identical(W$device$index,r$device$index)||W$shape[1]!=r$numel())
    .stop("Residual weights have incompatible shape, dtype or device")
  if(length(W$shape)==1) W*r else W$matmul(r)
}
.residual_jvp<-function(P,x,u,counts) {
  .count(counts,"jvp")
  if(!is.null(P$jvp)) return(.problem_call(P,P$jvp,x,u))
  previous_order <- .derivative_context$second_order
  .derivative_context$second_order <- TRUE
  on.exit({.derivative_context$second_order <- previous_order}, add = TRUE)
  torch::with_enable_grad({
    z<-.clone(x,TRUE);r<-.residual(P,z,counts);w<-torch::torch_zeros_like(r)$requires_grad_(TRUE)
    eg<-.autograd_tree(.dot(r,w),z,TRUE);pair<-.tree_dot(eg,u)
    if(!pair$requires_grad) return(torch::torch_zeros_like(r))
    torch::autograd_grad(pair,w,allow_unused=TRUE)[[1]]
  })
}
.residual_adjoint<-function(P,x,v,counts) {
  .count(counts,"adjoint")
  if(!is.null(P$adjoint)) out<-.problem_call(P,P$adjoint,x,v) else out<-torch::with_enable_grad({
    z<-.clone(x,TRUE);r<-.residual(P,z,counts)
    riem.egrad2rgrad(P$manifold,z,.autograd_tree(.dot(r,v$detach()),z))
  })
  .check_tangent(P$manifold,x,out,"Residual adjoint");.clone(out)
}
.gn<-function(P,x,u,counts) {
  .count(counts,"hessian")
  .residual_adjoint(P,x,.robust_weight(P,x,.residual_jvp(P,x,u,counts),counts),counts)
}
#' Finite-Sum Problems with Explicit Normalization
#' @param manifold Geometry specification.
#' @param fn Term callback `fn(x, i)` returning the scalar loss for index i.
#' @param n Number of terms.
#' @param normalization `"mean"` or `"sum"` for the full objective.
#' @param sampling `"without_replacement"` or `"with_replacement"` per minibatch.
#' @param batch_fn Optional vectorized callback `batch_fn(x, indices)` returning
#'   one loss per requested index as a vector tensor. When supplied, stochastic
#'   and chunked full evaluations use it instead of calling `fn` repeatedly.
#' @param data Optional registered data passed to callbacks as a named `data`
#'   argument.
#' @param preconditioner Optional tangent preconditioner; see [riem.problem()].
#' @param required_operations Device operations required by callbacks.
#' @param value_rgrad Optional fused `function(x, indices)` returning `value` and
#'   metric `gradient` for the requested minibatch, with the same mean/sum
#'   normalization as `fn`. Registered data is passed when present.
#' @param evaluation_batch_size Positive chunk size used for full objective and
#'   gradient evaluations. Chunk gradients are accumulated after detaching their
#'   computation graphs, bounding graph memory independently of `n`.
#' @return A `riem_finitesum` problem; deterministic solvers evaluate all terms.
#' @details A minibatch gradient is the average of its terms, multiplied by n
#' for sum normalization. Independent minibatches use torch's RNG. Initial,
#' requested periodic and final diagnostics evaluate the full objective and
#' gradient in chunks; intermediate minibatch norms are not stationarity
#' certificates. Indices passed to callbacks are one-based R integers.
#' @examples
#' if (torch::torch_is_installed()) {
#'   P <- riem.problem.finitesum(manifold.euclidean(1),
#'     function(x, i) torch::torch_sum((x - i)^2)/2, n = 3)
#'   x <- torch::torch_zeros(1, dtype = torch::torch_float64())
#'   riem.optimize(P, x, "stochastic_gradient",
#'                 control = list(max_iterations = 10, batch_size = 3))
#' }
#' @export
riem.problem.finitesum <- function(manifold,fn,n,normalization=c("mean","sum"),
                                  sampling=c("without_replacement","with_replacement"),
                                  batch_fn=NULL,data=NULL,
                                  evaluation_batch_size=1024L,value_rgrad=NULL,preconditioner=NULL,required_operations=NULL) {
  n<-.positive_integer(n,"n");if(length(n)!=1) .stop("n must be scalar")
  if(!is.function(fn)) .stop("fn must be a function")
  if(!is.null(batch_fn)&&!is.function(batch_fn)) .stop("batch_fn must be a function")
  evaluation_batch_size<-.positive_integer(evaluation_batch_size,"evaluation_batch_size")
  if(length(evaluation_batch_size)!=1) .stop("evaluation_batch_size must be scalar")
  P<-riem.problem(manifold,fn,data=data,value_rgrad=value_rgrad,preconditioner=preconditioner,required_operations=required_operations);P$term<-fn;P$batch_fn<-batch_fn;P$n<-n
  P$normalization<-match.arg(normalization);P$sampling<-match.arg(sampling)
  P$evaluation_batch_size<-evaluation_batch_size
  class(P)<-c("riem_finitesum",class(P));P
}

.robust_rho <- function(s,loss) switch(loss,
  linear=s,huber=torch::torch_where(s<=1,s,s$clamp_min(1)$sqrt()*2-1),
  soft_l1=((s+1)$sqrt()-1)*2,cauchy=(s+1)$log())
.robust_derivative <- function(s,loss) switch(loss,
  linear=torch::torch_ones_like(s),huber=torch::torch_where(s<=1,torch::torch_ones_like(s),s$clamp_min(1)$rsqrt()),
  soft_l1=(s+1)$rsqrt(),cauchy=(s+1)$reciprocal())
.leastsquares_value <- function(P,r) {
  if(is.null(P$loss)||P$loss=="linear") return(.dot(r,.weight(P,r))/2)
  z<-.robust_rho((r/P$loss_scale)^2,P$loss)*(P$loss_scale^2)
  if(!is.null(P$weights)) z<-z*P$weights
  z$sum()/2
}
.robust_weight <- function(P,x,v,counts) {
  if(is.null(P$loss)||P$loss=="linear") return(.weight(P,v))
  r<-.residual(P,x,counts)
  .weight(P,v)*.robust_derivative((r/P$loss_scale)^2,P$loss)
}
