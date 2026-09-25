.frame_dims <- function(p,k) {
  p <- .positive_integer(p,"p"); k <- .positive_integer(k,"k")
  if (length(p)!=1 || length(k)!=1 || k>p) .stop("Require 1 <= k <= p")
  c(p,k)
}
.frame <- function(p,k,metric,grassmann=FALSE) {
  s <- .frame_dims(p,k); p<-s[1]; k<-s[2]
  tangent <- if (grassmann) function(x,u) u-x$matmul(x$t()$matmul(u)) else
    function(x,u) u-x$matmul(.sym(x$t()$matmul(u)))
  eg <- if (metric=="canonical" && !grassmann) function(x,u) u-x$matmul(u$t()$matmul(x)) else tangent
  inn <- if (metric=="canonical" && !grassmann) function(x,u,v)
    .dot(u,v)-.dot(x$t()$matmul(u),x$t()$matmul(v))/2 else function(x,u,v) .dot(u,v)
  ehess <- if (grassmann) function(x,u,egrad,ehess)
    tangent(x, ehess-u$matmul(x$t()$matmul(egrad))) else if (metric=="euclidean")
      function(x,u,egrad,ehess) tangent(x,ehess-u$matmul(.sym(x$t()$matmul(egrad)))) else
      function(x,u,egrad,ehess) {
        g<-eg(x,egrad)
        dg<-ehess-u$matmul(egrad$t()$matmul(x))-x$matmul(ehess$t()$matmul(x))-x$matmul(egrad$t()$matmul(u))
        correction<-u$matmul(x$t()$matmul(g))+g$matmul(x$t()$matmul(u))
        tangent(x,dg-(correction+x$matmul(x$t()$matmul(correction)))/2)
      }
  ops <- list(
    belongs=function(x,tol) .scalar(.fnorm(x$t()$matmul(x)-.eye(x,k)))<=tol,
    tangent=tangent, inner=inn, egrad2rgrad=eg,
    retr=function(x,u) .polar(x+u), project=.polar,
    residual=function(x) .scalar(.fnorm(x$t()$matmul(x)-.eye(x,k))))
  if (!is.null(ehess)) ops$ehess2rhess <- ehess
  ops$exp<-function(x,u) {
    A<-x$t()$matmul(u);K<-u$matmul(x$t())-x$matmul(u$t())
    if(grassmann) return(torch::torch_matrix_exp(K)$matmul(x))
    if(metric=="canonical") return(torch::torch_matrix_exp(K-x$matmul(A)$matmul(x$t()))$matmul(x))
    torch::torch_matrix_exp(K)$matmul(x)$matmul(torch::torch_matrix_exp(-A))
  }
  if(grassmann) {
    ops$log<-function(x,y) {
      cross<-x$t()$matmul(y)
      if(.scalar(torch::linalg_svdvals(cross)$min())<sqrt(.default_tolerance(x))) .stop("Grassmann logarithm meets the pi/2 cut locus")
      Z<-.solve(cross$t(),(y-x$matmul(cross))$t())$t()
      sv<-torch::linalg_svd(Z,full_matrices=FALSE)
      (sv[[1]]*sv[[2]]$atan()$unsqueeze(1))$matmul(sv[[3]])
    }
    ops$sqdist<-function(x,y) {sv<-torch::linalg_svdvals(x$t()$matmul(y));.acos_sq(sv$clamp(0,1))$sum()}
  }
  riem.manifold(if(grassmann) "grassmann" else "stiefel", s,
    if(grassmann) k*(p-k) else p*k-k*(k+1)/2, metric, ops,
    list(hessian=TRUE,curve_derivative=TRUE),
    list(p=p,k=k,embedding="frame"), primitive_derivatives=c(tangent=2L,inner=2L,egrad2rgrad=2L,transport=2L,retr=1L,project=1L), required_operations=c("basic","eigh","svd","matrix_exp"), second_order_retraction=grassmann || metric=="euclidean")
}
#' Stiefel and Grassmann Frame Geometries
#' @param p Ambient row dimension.
#' @param field Real (default) or complex scalar field.
#' @param k Number of orthonormal columns.
#' @param metric Required Stiefel metric, `"euclidean"` or `"canonical"`.
#' @param embedding Grassmann point representation: `"frame"` (p by k) or
#' `"projection"` (p by p orthogonal projectors, with half-Frobenius metric).
#' @return A geometry specification using polar retraction and projection
#' transport. Exact ambient Hessian conversion is available for Euclidean-metric
#' Stiefel and both Grassmann representations, and canonical-metric Stiefel.
#' @details Grassmann objectives must be invariant to orthogonal changes of
#' frame. Projection representation uses the metric equivalent to the frame
#' metric, including its factor of one half.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.stiefel(4, 2, metric = "euclidean")
#'   x <- riem.random(M)
#'   u <- riem.tangent(M, x, torch::torch_randn_like(x))
#'   riem.ehess2rhess(M, x, u, x, u)
#'
#'   manifold.grassmann(4, 2, embedding = "projection")
#' }
#' @export
manifold.stiefel <- function(p,k,metric,field=c("real","complex")) {
  metric<-match.arg(metric,c("euclidean","canonical"))
  if(match.arg(field)=="complex") {
    if(metric!="euclidean") .stop("Complex Stiefel currently supports the Euclidean metric")
    return(.complex_frame(p,k,FALSE))
  }
  .frame(p,k,metric)
}
#' @rdname manifold.stiefel
#' @export
manifold.grassmann <- function(p,k,embedding=c("frame","projection"),field=c("real","complex")) {
  if(match.arg(field)=="complex") {
    if(match.arg(embedding)!="frame") .stop("Complex Grassmann currently uses frame representation")
    return(.complex_frame(p,k,TRUE))
  }
  embedding <- match.arg(embedding)
  M <- .frame(p,k,"canonical",TRUE)
  if(embedding=="frame") return(M)
  s <- .frame_dims(p,k); p<-s[1]; k<-s[2]
  proj <- function(x) {
    e<-torch::linalg_eigh(.sym(x)); q<-e[[2]]$narrow(2,p-k+1,k)
    q$matmul(q$t())
  }
  tangent<-function(x,u) {u<-.sym(u); x$matmul(u)+u$matmul(x)-x$matmul(u)$matmul(x)*2}
  ehess<-function(x,u,egrad,ambient_hess) {
    g<-.sym(egrad);h<-.sym(ambient_hess)
    dg<-u$matmul(g)+g$matmul(u)-u$matmul(g)$matmul(x)*2-x$matmul(g)$matmul(u)*2+tangent(x,h)
    tangent(x,dg*2)
  }
  riem.manifold("grassmann",c(p,p),k*(p-k),"projection_half_frobenius",list(
    belongs=function(x,tol) .scalar(.fnorm(x-x$t()))<=tol &&
      .scalar(.fnorm(x$matmul(x)-x))<=tol && abs(.scalar(x$trace())-k)<=tol,
    tangent=tangent, inner=function(x,u,v) .dot(u,v)/2,
    exp=function(x,u) {K<-u$matmul(x)-x$matmul(u);E<-torch::torch_matrix_exp(K);.sym(E$matmul(x)$matmul(E$t()))},
    log=function(x,y) {
      X<-torch::linalg_eigh(x)[[2]]$narrow(2,p-k+1,k)
      Y<-torch::linalg_eigh(y)[[2]]$narrow(2,p-k+1,k)
      U<-riem.log(.frame(p,k,"canonical",TRUE),X,Y)
      .sym(U$matmul(X$t())+X$matmul(U$t()))
    },
    sqdist=function(x,y) {
      X<-torch::linalg_eigh(x)[[2]]$narrow(2,p-k+1,k)
      Y<-torch::linalg_eigh(y)[[2]]$narrow(2,p-k+1,k)
      riem.sqdist(.frame(p,k,"canonical",TRUE),X,Y)
    },
    egrad2rgrad=function(x,u) tangent(x,u)*2,
    retr=function(x,u) proj(x+u),project=proj,
    ehess2rhess=ehess,
    random=function(x) {q<-.polar(torch::torch_randn(c(p,k),dtype=x$dtype,device=x$device));q$matmul(q$t())},
    residual=function(x) .scalar(.fnorm(x$matmul(x)-x))),
    list(hessian=TRUE,curve_derivative=TRUE),
    specification=list(p=p,k=k,embedding=embedding), primitive_derivatives=c(tangent=2L,inner=2L,egrad2rgrad=2L,transport=2L,retr=1L), required_operations=c("basic","eigh"), second_order_retraction=TRUE)
}
#' Symmetric Positive-Definite Matrix Geometry
#' @param p Matrix size.
#' @param metric Required metric: `"airm"` (affine invariant), `"lerm"`
#'   (log Euclidean), or `"wasserstein"` (Bures--Wasserstein). Descriptive
#'   aliases `"affine_invariant"`, `"log_euclidean"`, and `"bures_wasserstein"`
#'   are accepted.
#' @return A manifold of p by p positive-definite matrices with symmetric tangents.
#' @details AIRM uses tr(X^-1 U X^-1 V), a second-order polynomial retraction,
#'   and congruence transport along the endpoint AIRM geodesic. LERM pulls back
#'   the Frobenius metric through log, with exact chart retraction and transport.
#'   Wasserstein uses 0.5 tr(L_X(U) V), where X L_X(U)+L_X(U) X=U;
#'   its local exponential requires I+L_X(U) positive definite.
#'   Custom matrix logarithm/root derivatives are first-order only.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.spd(2, metric = "airm")
#'   x <- torch::torch_eye(2, dtype = torch::torch_float64())
#'   riem.egrad2rgrad(M, x, x)
#'   riem.retr(M, x, 0.1 * x)
#' }
#' @export
manifold.spd <- function(p,metric) {
  p<-.positive_integer(p,"p"); if(length(p)!=1) .stop("p must be scalar")
  aliases<-c(affine_invariant="airm", log_euclidean="lerm", bures_wasserstein="wasserstein")
  if(length(metric)==1 && metric %in% names(aliases)) metric<-unname(aliases[metric])
  metric<-match.arg(metric,c("airm","lerm","wasserstein"))
  mf<-riem.matrix.function; df<-riem.matrix.frechet
  ops<-list(belongs=function(x,tol) .scalar(.fnorm(x-x$t()))<=tol && .scalar(torch::linalg_eigvalsh(.sym(x))$min())>0,
    tangent=function(x,u) .sym(u),
    project=function(x) {e<-torch::linalg_eigh(.sym(x)); (e[[2]]*e[[1]]$clamp_min(1e-8)$unsqueeze(1))$matmul(e[[2]]$t())},
    random=function(x) {a<-torch::torch_randn_like(x); a$matmul(a$t())/p+.eye(x)},
    residual=function(x) max(.scalar(.fnorm(x-x$t())), max(0,-.scalar(torch::linalg_eigvalsh(.sym(x))$min()))))
  if(metric=="airm") {
    ops$inner<-function(x,u,v) .dot(.solve(x,u),.solve(x,v)$t())
    ops$egrad2rgrad<-function(x,u) .sym(x$matmul(.sym(u))$matmul(x))
    ops$retr<-function(x,u) .sym(x+u+u$matmul(.solve(x,u))/2)
    ops$exp<-function(x,u) {s<-mf(x,"sqrt"); r<-mf(x,"invsqrt"); .sym(s$matmul(mf(.sym(r$matmul(u)$matmul(r)),"exp"))$matmul(s))}
    ops$log<-function(x,y) {s<-mf(x,"sqrt");r<-mf(x,"invsqrt");.sym(s$matmul(mf(.sym(r$matmul(y)$matmul(r)),"log"))$matmul(s))}
    ops$sqdist<-function(x,y) {r<-mf(x,"invsqrt"); z<-mf(.sym(r$matmul(y)$matmul(r)),"log");.dot(z,z)}
    ops$transport<-function(x,u,y,v) {s<-mf(x,"sqrt");r<-mf(x,"invsqrt");a<-s$matmul(mf(.sym(r$matmul(y)$matmul(r)),"sqrt"))$matmul(r);.sym(a$matmul(v)$matmul(a$t()))}
    ops$ehess2rhess<-function(x,u,egrad,ehess) .sym(x$matmul(.sym(ehess))$matmul(x)+
      (u$matmul(.sym(egrad))$matmul(x)+x$matmul(.sym(egrad))$matmul(u))/2)
  } else if(metric=="lerm") {
    ops$inner<-function(x,u,v) .dot(df(x,u,"log"),df(x,v,"log"))
    ops$egrad2rgrad<-function(x,u) {z<-mf(x,"log");df(z,df(z,.sym(u),"exp"),"exp")}
    ops$retr<-function(x,u) mf(mf(x,"log")+df(x,u,"log"),"exp")
    ops$exp<-ops$retr
    ops$log<-function(x,y) df(mf(x,"log"),mf(y,"log")-mf(x,"log"),"exp")
    ops$sqdist<-function(x,y) {z<-mf(x,"log")-mf(y,"log");.dot(z,z)}
    ops$transport<-function(x,u,y,v) df(mf(y,"log"),df(x,v,"log"),"exp")
  } else {
    ops$inner<-function(x,u,v) .dot(.sylvester(x,u),v)/2
    ops$egrad2rgrad<-function(x,u) {g<-.sym(u);2*(x$matmul(g)+g$matmul(x))}
    ops$retr<-function(x,u) {a<-.eye(x)+.sylvester(x,u);if(.scalar(torch::linalg_eigvalsh(a)$min())<=0) .stop("Wasserstein step leaves the local exponential domain");.sym(a$matmul(x)$matmul(a))}
    ops$exp<-ops$retr
    ops$log<-function(x,y) {s<-mf(x,"sqrt");r<-mf(x,"invsqrt");a<-r$matmul(mf(.sym(s$matmul(y)$matmul(s)),"sqrt"))$matmul(r)-.eye(x);.sym(a$matmul(x)+x$matmul(a))}
    ops$sqdist<-function(x,y) {s<-mf(x,"sqrt"); x$trace()+y$trace()-mf(.sym(s$matmul(y)$matmul(s)),"sqrt")$trace()*2}
  }
  riem.manifold("spd",c(p,p),p*(p+1)/2,metric,ops,
    list(hessian=metric=="airm",curve_derivative=metric=="airm",snapshot_transport=TRUE,isometric_transport=metric %in% c("airm","lerm")),list(p=p), primitive_derivatives=switch(metric,airm=c(tangent=2L,inner=2L,egrad2rgrad=2L,retr=2L,exp=1L,log=1L,sqdist=1L,transport=1L),lerm=c(sqdist=1L),wasserstein=c(tangent=2L,egrad2rgrad=2L,transport=2L,log=1L,sqdist=1L)), required_operations=c("basic","eigh","cholesky","matrix_exp"), second_order_retraction=metric %in% c("airm","lerm"))
}
