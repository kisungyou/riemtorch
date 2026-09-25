#' Repeated and Scaled Manifold Geometries
#' @param manifold Base manifold with tensor points.
#' @param copies Positive number of repeated factors.
#' @param scale Positive distance multiplier; the metric is multiplied by scale squared.
#' @return A manifold specification. Power points have shape `c(copies, base_shape)`.
#'   Those factors belong to one optimization point; additional leading axes
#'   still denote independent batches. Scaling preserves the base representation.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.power(manifold.sphere(3), 4)
#'   x <- riem.random(M,device="cpu")
#'   riem.belongs(M,x)
#'   manifold.scaled(manifold.sphere(3),2)
#' }
#' @export
manifold.power <- function(manifold,copies) {
  M<-manifold;.check_manifold(M);copies<-.positive_integer(copies,"copies")
  if(length(copies)!=1||is.null(M$shape)) .stop("Power geometry requires tensor points and scalar copies")
  ops<-list(
    belongs=function(x,tol) all(riem.belongs(M,x,tol)),
    tangent=function(x,u) riem.tangent(M,x,u),
    inner=function(x,u,v) riem.inner(M,x,u,v)$sum(),
    egrad2rgrad=function(x,u) riem.egrad2rgrad(M,x,u),
    retr=function(x,u) riem.retr(M,x,u),
    transport=function(x,u,y,v) riem.transport(M,x,u,y,v),
    random=function(x) .random_execution(M,copies,list(device=x$device,dtype=x$dtype)))
  for(key in intersect(c("exp","log","project","ehess2rhess"),names(M$operations)))
    ops[[key]]<-local({k<-key;function(...) do.call(.geom,c(list(M,k),list(...)))})
  if(!is.null(M$operations$sqdist)) ops$sqdist<-function(x,y) riem.sqdist(M,x,y)$sum()
  out<-riem.manifold("power",c(copies,M$shape),copies*M$dimension,paste0("power_",M$metric),ops,
    capabilities=M$capabilities,specification=list(copies=copies,field=M$specification$field),
    primitive_derivatives=M$capabilities$derivatives,required_operations=.required_operations(M),
    second_order_retraction=M$capabilities$second_order_retraction)
  out$base<-M;out$copies<-copies;out
}
#' @rdname manifold.power
#' @export
manifold.scaled <- function(manifold,scale) {
  M<-manifold;.check_manifold(M);.number(scale,"scale")
  if(is.null(M$shape)) .stop("Scaled wrapper currently requires tensor points")
  out<-M;out$name<-"scaled";out$metric<-paste0("scaled_",M$metric)
  out$operations$inner<-function(x,u,v) riem.inner(M,x,u,v)*(scale^2)
  out$operations$egrad2rgrad<-function(x,u) riem.egrad2rgrad(M,x,u)/(scale^2)
  if(!is.null(M$operations$ehess2rhess)) out$operations$ehess2rhess<-
    function(x,u,egrad,ehess) riem.ehess2rhess(M,x,u,egrad,ehess)/(scale^2)
  if(!is.null(M$operations$sqdist)) out$operations$sqdist<-function(x,y) riem.sqdist(M,x,y)*(scale^2)
  out$batch_operations<-list();out$base<-M;out$scale<-scale
  out$specification<-list(scale=scale,base=M$specification,field=M$specification$field);out
}
#' Positive Tensors with a Log-Euclidean Metric
#' @param shape Positive point dimensions.
#' @return A positive-tensor manifold with metric sum(u*v/x^2), exact maps,
#'   isometric transport and exact ambient Hessian conversion.
#' @examples
#' if(torch::torch_is_installed()) {
#'   M <- manifold.positive(3)
#'   riem.random(M,device="cpu")
#' }
#' @export
manifold.positive <- function(shape) {
  shape<-.positive_integer(shape)
  riem.manifold("positive",shape,prod(shape),"log_euclidean",list(
    belongs=function(x,tol) .scalar(x$min())>0,
    tangent=function(x,u) u,inner=function(x,u,v) .dot(u/x,v/x),
    egrad2rgrad=function(x,u) u*x*x,
    retr=function(x,u) x*(u/x)$exp(),exp=function(x,u) x*(u/x)$exp(),
    log=function(x,y) x*(y/x)$log(),sqdist=function(x,y) .dot((y/x)$log(),(y/x)$log()),
    transport=function(x,u,y,v) v*(y/x),
    ehess2rhess=function(x,u,egrad,ehess) x*x*ehess+x*u*egrad,
    random=function(x) torch::torch_randn_like(x)$exp(),
    project=function(x) {if(.scalar(x$min())<=0) .stop("Positive projection requires positive input");x}),
    list(hessian=TRUE,curve_derivative=TRUE,snapshot_transport=TRUE,isometric_transport=TRUE),list(shape=shape),
    primitive_derivatives=setNames(rep(2L,9),c("tangent","inner","egrad2rgrad","retr","exp","log","sqdist","transport","ehess2rhess")),
    required_operations="basic",second_order_retraction=TRUE)
}
#' Affine Subspaces and Orthogonal Matrices
#' @param basis Matrix with orthonormal columns spanning the affine directions.
#' @param offset Vector locating the affine subspace; defaults to zero.
#' @param p Dimension of an orthogonal matrix.
#' @return A Frobenius geometry. The orthogonal group includes both determinant signs.
#' @examples
#' if(torch::torch_is_installed()) {
#'   B <- torch::torch_eye(3,dtype=torch::torch_float64())[,1:2]
#'   M <- manifold.affine(B)
#'   riem.belongs(M,riem.random(M,device="cpu"))
#'   manifold.orthogonal(3)
#' }
#' @export
manifold.affine <- function(basis,offset=NULL) {
  if(!inherits(basis,"torch_tensor")||length(basis$shape)!=2||!.finite(basis)) .stop("basis must be a finite matrix tensor")
  n<-basis$shape[1];d<-basis$shape[2]
  if(d>n||.scalar(.fnorm(basis$t()$matmul(basis)-.eye(basis,d)))>.default_tolerance(basis)*10)
    .stop("basis columns must be orthonormal")
  if(is.null(offset)) offset<-torch::torch_zeros(n,dtype=basis$dtype,device=basis$device)
  if(!inherits(offset,"torch_tensor")||!identical(as.integer(offset$shape),as.integer(n))||!.finite(offset)||
     offset$dtype!=basis$dtype||.device_string(offset$device)!=.device_string(basis$device)) .stop("offset must match basis placement and row dimension")
  basis<-.clone(basis);offset<-.clone(offset)
  project_tangent<-function(u) basis$matmul(basis$t()$matmul(u))
  project<-function(x) offset+project_tangent(x-offset)
  riem.manifold("affine",n,d,"euclidean",list(
    belongs=function(x,tol) .scalar(.fnorm(x-project(x)))<=tol*(1+.scalar(.fnorm(x))),
    tangent=function(x,u) project_tangent(u),inner=function(x,u,v) .dot(u,v),
    egrad2rgrad=function(x,u) project_tangent(u),retr=function(x,u) x+u,
    exp=function(x,u) x+u,log=function(x,y) y-x,sqdist=function(x,y) .dot(y-x,y-x),
    project=project,transport=function(x,u,y,v) v,
    ehess2rhess=function(x,u,egrad,ehess) project_tangent(ehess)),
    list(hessian=TRUE,curve_derivative=TRUE,snapshot_transport=TRUE,isometric_transport=TRUE),list(basis=basis,offset=offset),
    primitive_derivatives=setNames(rep(2L,9),c("tangent","inner","egrad2rgrad","retr","exp","log","sqdist","transport","project")),
    required_operations="basic",second_order_retraction=TRUE)
}
#' @rdname manifold.affine
#' @export
manifold.orthogonal <- function(p) {
  M<-.frame(p,p,"euclidean");M$name<-"orthogonal";M$metric<-"frobenius";M
}
.sinkhorn <- function(x,tolerance=1e-12,max_iterations=2000L) {
  if(!.finite(x)||.scalar(x$min())<=0) .stop("Balancing requires strictly positive entries")
  tol<-if(.default_tolerance(x)>1e-7) max(tolerance,1e-6) else tolerance
  for(i in seq_len(max_iterations)) {
    x<-x/x$sum(dim=2,keepdim=TRUE);x<-x/x$sum(dim=1,keepdim=TRUE)
    if(.scalar((x$sum(dim=2)-1)$abs()$max())<tol) return(x)
  }
  .stop("Positive matrix balancing did not converge")
}
#' Positive Doubly Stochastic Matrix Geometry
#' @param p Matrix dimension, at least two.
#' @return Strictly positive matrices with unit row and column sums, with the
#'   Fisher metric sum(u*v/x). No exact distance or Hessian conversion is claimed.
#' @details Retraction exponentiates a tangent ratio then balances its rows and
#'   columns. Tangent projection solves a gauge-fixed weighted normal system.
#'   Boundary points with zero entries are outside the manifold.
#' @examples
#' if(torch::torch_is_installed()) {
#'   M <- manifold.doublystochastic(3)
#'   x <- riem.random(M,device="cpu")
#'   x$sum(dim=1)
#' }
#' @export
manifold.doublystochastic <- function(p) {
  p<-.positive_integer(p);if(length(p)!=1||p<2) .stop("p must be at least two")
  tangent<-function(x,z) {
    C<-x$narrow(2,1,p-1L)
    A<-torch::torch_cat(list(torch::torch_cat(list(.eye(x,p),C),dim=2),
      torch::torch_cat(list(C$t(),.eye(x,p-1L)),dim=2)),dim=1)
    rhs<-torch::torch_cat(list(z$sum(dim=2),z$sum(dim=1)$narrow(1,1,p-1L)))
    ab<-.solve(A,rhs);a<-ab$narrow(1,1,p)
    b<-torch::torch_cat(list(ab$narrow(1,p+1,p-1L),torch::torch_zeros(1,dtype=x$dtype,device=x$device)))
    z-x*(a$unsqueeze(2)+b$unsqueeze(1))
  }
  riem.manifold("doublystochastic",c(p,p),(p-1)^2,"fisher",list(
    belongs=function(x,tol) .scalar(x$min())>0&&.scalar((x$sum(dim=1)-1)$abs()$max())<=tol&&.scalar((x$sum(dim=2)-1)$abs()$max())<=tol,
    tangent=tangent,inner=function(x,u,v) (u*v/x)$sum(),egrad2rgrad=function(x,u) tangent(x,x*u),
    retr=function(x,u) {logits<-x$log()+u/x;.sinkhorn((logits-logits$max())$exp())},
    project=.sinkhorn,random=function(x) .sinkhorn(torch::torch_randn_like(x)$exp())),
    specification=list(p=p),primitive_derivatives=c(tangent=2L,inner=2L,egrad2rgrad=2L),
    required_operations=c("basic","solve"))
}
