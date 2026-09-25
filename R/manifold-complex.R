.complex_euclidean <- function(shape) {
  M<-manifold.euclidean(shape);M$name<-"euclidean.complex";M$metric<-"real_hermitian"
  M$dimension<-M$dimension*2;M$specification$field<-"complex";M
}
.complex_sphere <- function(p) {
  M<-manifold.sphere(p);M$name<-"sphere.complex";M$dimension<-2*p-1
  M$specification$field<-"complex";M
}
.imaginary <- function(theta) torch::torch_complex(torch::torch_zeros_like(theta),theta)
#' Complex Phases and Unitary Matrices
#' @param p Ambient vector or matrix dimension.
#' @return A complex manifold with real Hermitian inner products and real scalar
#'   objectives. `complexcircle` represents p independent unit-modulus phases.
#'   `unitary` represents square matrices U with U* U = I.
#' @details Use complex64 or complex128 points with float32 or float64 losses,
#'   respectively. Automatic Hessians are qualified for complex Euclidean space,
#'   sphere and circle. Complex frames currently provide first-order geometry.
#'   Circle logarithms exclude antipodes; Grassmann logarithms exclude the cut locus.
#' @examples
#' if(torch::torch_is_installed()) {
#'   M <- manifold.complexcircle(3)
#'   x <- riem.random(M,device="cpu")
#'   P <- riem.problem(M,function(x) -x$real$sum())
#'   riem.optimize(P,x,device="cpu")$objective
#'   riem.random(manifold.unitary(2),device="cpu")
#' }
#' @export
manifold.complexcircle <- function(p) {
  p<-.positive_integer(p);if(length(p)!=1) .stop("p must be scalar")
  tangent<-function(x,u) u-x*(x$conj()*u)$real
  project<-function(x) {if(.scalar(x$abs()$min())<=0) .stop("Cannot normalize zero phase");x/x$abs()}
  angle<-function(x,y) torch::torch_angle(x$conj()*y)
  riem.manifold("complexcircle",p,p,"real_hermitian",list(
    belongs=function(x,tol) .scalar((x$abs()-1)$abs()$max())<=tol,
    tangent=tangent,inner=function(x,u,v) .dot(u,v),egrad2rgrad=tangent,
    retr=function(x,u) project(x+u),project=project,
    exp=function(x,u) x*.imaginary((x$conj()*u)$imag)$exp(),
    log=function(x,y) {a<-angle(x,y);if(.scalar(a$abs()$max())>pi-1e-8) .stop("Circle logarithm is undefined at antipodes");x*.imaginary(a)},
    sqdist=function(x,y) angle(x,y)$square()$sum(),
    transport=function(x,u,y,v) y*(x$conj()*v),
    ehess2rhess=function(x,u,egrad,ehess) tangent(x,ehess)-u*(x$conj()*egrad)$real,
    random=function(x) .imaginary(torch::torch_rand(x$shape,dtype=.real_dtype(x$dtype),device=x$device)*(2*pi))$exp()),
    list(hessian=TRUE,curve_derivative=TRUE,snapshot_transport=TRUE,isometric_transport=TRUE),list(p=p,field="complex"),
    primitive_derivatives=setNames(rep(2L,9),c("tangent","inner","egrad2rgrad","retr","exp","log","sqdist","transport","project")),
    required_operations="basic",second_order_retraction=TRUE)
}
.complex_polar <- function(x) {
  sv<-torch::linalg_svd(x,full_matrices=FALSE)
  if(.scalar(sv[[2]]$min())<=0) .stop("Complex frame projection needs full column rank")
  sv[[1]]$matmul(sv[[3]])
}
.complex_frame <- function(p,k,grassmann=FALSE) {
  shape<-.frame_dims(p,k);p<-shape[1];k<-shape[2]
  tangent<-if(grassmann) function(x,u) u-x$matmul(.adj(x)$matmul(u)) else
    function(x,u) u-x$matmul(.herm(.adj(x)$matmul(u)))
  ops<-list(belongs=function(x,tol) .scalar(.fnorm(.adj(x)$matmul(x)-.eye(x,k)))<=tol,
    tangent=tangent,inner=function(x,u,v) .dot(u,v),egrad2rgrad=tangent,
    retr=function(x,u) .complex_polar(x+u),project=.complex_polar,
    exp=function(x,u) {
      A<-.adj(x)$matmul(u);K<-u$matmul(.adj(x))-x$matmul(.adj(u))
      E<-torch::torch_matrix_exp(K)$matmul(x)
      if(grassmann) E else E$matmul(torch::torch_matrix_exp(-A))
    },random=function(x) .complex_polar(torch::torch_randn_like(x)))
  if(grassmann) {
    ops$log<-function(x,y) {
      C<-.adj(x)$matmul(y)
      if(.scalar(torch::linalg_svdvals(C)$min())<sqrt(.default_tolerance(x))) .stop("Complex Grassmann logarithm meets cut locus")
      Z<-.adj(.solve(.adj(C),.adj(y-x$matmul(C))))
      sv<-torch::linalg_svd(Z,full_matrices=FALSE)
      (sv[[1]]*sv[[2]]$atan()$unsqueeze(1))$matmul(sv[[3]])
    }
    ops$sqdist<-function(x,y) .acos_sq(torch::linalg_svdvals(.adj(x)$matmul(y))$clamp(0,1))$sum()
  }
  riem.manifold(if(grassmann) "grassmann.complex" else "stiefel.complex",shape,
    if(grassmann) 2*k*(p-k) else 2*p*k-k*k,"real_hermitian",ops,
    specification=list(p=p,k=k,field="complex",embedding="frame"),
    primitive_derivatives=c(tangent=2L,inner=2L,egrad2rgrad=2L,exp=2L),
    required_operations=c("basic","svd","solve","matrix_exp"),second_order_retraction=TRUE)
}
#' @rdname manifold.complexcircle
#' @export
manifold.unitary <- function(p) {
  M<-.complex_frame(p,p);M$name<-"unitary";M
}
