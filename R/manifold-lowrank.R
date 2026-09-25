.rank_support<-function(x,k) {
  e<-torch::linalg_eigh(.sym(x));q<-e[[2]]$narrow(2,x$shape[1]-k+1,k)
  q$matmul(q$t())
}
.rank_project<-function(x,k,repair=FALSE) {
  e<-torch::linalg_eigh(.sym(x));n<-x$shape[1];d<-e[[1]]$narrow(1,n-k+1,k);q<-e[[2]]$narrow(2,n-k+1,k)
  if(repair) d<-d$clamp_min(1e-8) else if(.scalar(d$min())<=0) .stop("Step leaves the positive rank stratum")
  .sym((q*d$unsqueeze(1))$matmul(q$t()))
}
.rank_tangent<-function(x,u,k) {z<-.sym(u);n<-.eye(x)-.rank_support(x,k);.sym(z-n$matmul(z)$matmul(n))}
.rank_belongs<-function(x,k,tol) {
  if(.scalar(.fnorm(x-x$t()))>tol) return(FALSE)
  e<-torch::linalg_eigvalsh(.sym(x));p<-x$shape[1]
  .scalar(e[p-k+1])>tol && (k==p || .scalar(e$narrow(1,1,p-k)$abs()$max())<=tol)
}
#' Fixed-Rank Rectangular Matrix Geometry
#' @param m,p Matrix dimensions.
#' @param k Fixed rank, at most min(m,p).
#' @param representation Dense `matrix` (default) or compact `svd`; see
#'   [riem.materialize()] for compact point and tangent contracts.
#' @return An embedded Frobenius geometry with dense or compact matrix points.
#' @details Retraction truncates an SVD to rank k. Steps are valid only while
#' the selected singular values are positive and separated from discarded ones.
#' SVD/retraction derivatives at repeated singular values are not advertised.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.fixedrank(4, 3, 2)
#'   riem.belongs(M, riem.random(M))
#' }
#' @export
manifold.fixedrank <- function(m,p,k,representation=c("matrix","svd")) {
  representation<-match.arg(representation)
  if(representation=="svd") return(.compact_fixedrank(m,p,k))
  dims<-.positive_integer(c(m,p,k));if(length(dims)!=3 || k>min(m,p)) .stop("Require k <= min(m,p)")
  proj<-function(x) {s<-torch::linalg_svd(x,full_matrices=FALSE);if(.scalar(s[[2]][k])<=1e-12) .stop("Rank-deficient candidate");(s[[1]]$narrow(2,1,k)*s[[2]]$narrow(1,1,k)$unsqueeze(1))$matmul(s[[3]]$narrow(1,1,k))}
  tan<-function(x,u) {s<-torch::linalg_svd(x,full_matrices=FALSE);a<-s[[1]]$narrow(2,1,k);b<-s[[3]]$narrow(1,1,k);P<-a$matmul(a$t());Q<-b$t()$matmul(b);P$matmul(u)+u$matmul(Q)-P$matmul(u)$matmul(Q)}
  riem.manifold("fixedrank",c(m,p),k*(m+p-k),"embedded",list(
    belongs=function(x,tol) {s<-torch::linalg_svdvals(x);.scalar(s[k])>tol && (k==min(m,p)||.scalar(s$narrow(1,k+1,min(m,p)-k)$max())<=tol)},
    tangent=tan,inner=function(x,u,v) .dot(u,v),egrad2rgrad=tan,
    retr=function(x,u) proj(x+u),project=proj,
    residual=function(x) {s<-torch::linalg_svdvals(x);if(k==min(m,p)) 0 else .scalar(s$narrow(1,k+1,min(m,p)-k)$max())}),
    specification=list(m=m,p=p,k=k), primitive_derivatives=c(inner=2L), required_operations=c("basic","eigh","svd"))
}
#' Fixed-Rank Positive-Semidefinite Geometry
#' @param p Matrix dimension.
#' @param k Rank.
#' @param metric Required metric, `"embedded"` or `"wasserstein"`.
#' @param representation `"matrix"` (p by p) or `"factor"` (p by k,
#'   available for Wasserstein).
#' @return A fixed-rank PSD manifold. Factor points Y represent Y Y'.
#' @details Embedded tangents eliminate the null-null block. Wasserstein uses
#' the quotient of full-column-rank factors by right orthogonal transformations;
#' matrix metric is 0.5 tr(L_X(U) V) with the rank-restricted Sylvester inverse.
#' Factor tangents are horizontal (Y' U symmetric) and have Frobenius metric.
#' The factor metric matches Riemann's spdk factor geometry. Matrix and factor
#' forms are isometric under U -> U Y' + Y U'. They are not the embedded metric.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.spdk(4, 2, "wasserstein", "factor")
#'   riem.belongs(M, riem.random(M))
#' }
#' @export
manifold.spdk <- function(p,k,metric,representation=c("matrix","factor")) {
  s<-.frame_dims(p,k);p<-s[1];k<-s[2];metric<-match.arg(metric,c("embedded","wasserstein"));representation<-match.arg(representation)
  dimension<-p*k-k*(k-1)/2
  if(representation=="factor") {
    if(metric!="wasserstein") .stop("factor representation is available only for wasserstein")
    tan<-function(x,u) u-x$matmul(.sylvester(x$t()$matmul(x),x$t()$matmul(u)-u$t()$matmul(x)))
    ops<-list(belongs=function(x,tol) .scalar(torch::linalg_svdvals(x)$min())>tol,
      tangent=tan,inner=function(x,u,v) .dot(u,v),egrad2rgrad=tan,
      retr=function(x,u) {y<-x+u;if(.scalar(torch::linalg_eigvalsh(.sym(x$t()$matmul(y)))$min())<=0) .stop("Factor step leaves local horizontal domain");y},
      project=function(x) {if(.scalar(torch::linalg_svdvals(x)$min())<=1e-10) .stop("Factor must have full column rank");x},
      residual=function(x) 0)
    return(riem.manifold("spdk",c(p,k),dimension,"wasserstein",ops,
      list(curve_derivative=TRUE),list(p=p,k=k,representation=representation), primitive_derivatives=c(inner=2L), required_operations=c("basic","eigh","svd")))
  }
  tan<-function(x,u) .rank_tangent(x,u,k)
  ops<-list(belongs=function(x,tol) .rank_belongs(x,k,tol),tangent=tan,
    inner=function(x,u,v) .dot(u,v),egrad2rgrad=tan,
    retr=function(x,u) .rank_project(x+u,k),project=function(x) .rank_project(x,k,TRUE),
    random=function(x) {a<-torch::torch_randn(c(p,k),dtype=x$dtype,device=x$device);a$matmul(a$t())},
    residual=function(x) {e<-torch::linalg_eigvalsh(.sym(x));max(.scalar(.fnorm(x-x$t())),if(k<p) .scalar(e$narrow(1,1,p-k)$abs()$max()) else 0)})
  if(metric=="wasserstein") {
    ops$inner<-function(x,u,v) .dot(.sylvester(x,u,k),v)/2
    ops$egrad2rgrad<-function(x,u) {g<-.sym(u);2*(x$matmul(g)+g$matmul(x))}
    ops$retr<-function(x,u) {a<-.eye(x)+.sylvester(x,u,k);if(.scalar(torch::linalg_eigvalsh(a)$min())<=0) .stop("Step leaves local Bures domain");.sym(a$matmul(x)$matmul(a))}
  }
  riem.manifold("spdk",c(p,p),dimension,metric,ops,specification=list(p=p,k=k,representation=representation), primitive_derivatives=c(inner=2L), required_operations=c("basic","eigh","svd"))
}
#' Fixed-Rank Elliptope and Spectrahedron
#' @param p Matrix dimension.
#' @param k Rank; elliptope requires k >= 2, spectrahedron k >= 1.
#' @return Embedded Frobenius geometry of PSD matrices with unit diagonal
#'   (elliptope) or unit trace (spectrahedron).
#' @details Tangents are orthogonal projections onto the intersection of the
#' rank-stratum tangent and the diagonal/trace constraint. Retraction is spectral
#' rank truncation followed by diagonal/trace normalization. Singular constraint
#' strata and loss of rank are outside the supported domain.
#' @examples
#' if (torch::torch_is_installed()) {
#'   riem.random(manifold.elliptope(4, 2))
#'   riem.random(manifold.spectrahedron(4, 2))
#' }
#' @export
manifold.elliptope <- function(p,k) .psd_constraint(p,k,TRUE)
#' @rdname manifold.elliptope
#' @export
manifold.spectrahedron <- function(p,k) .psd_constraint(p,k,FALSE)
.psd_constraint<-function(p,k,diagonal) {
  M<-manifold.spdk(p,k,"embedded");if(diagonal&&k<2) .stop("Elliptope requires k >= 2 on a smooth positive-dimensional stratum")
  old<-M$operations;norm<-if(diagonal) function(x) {d<-.diag(x)$sqrt();if(.scalar(d$min())<=0) .stop("Zero diagonal");x/(d$unsqueeze(1)*d$unsqueeze(2))} else function(x) x/x$trace()
  tan<-if(diagonal) function(x,u) {
    z<-old$tangent(x,u);I<-.eye(x)
    A<-torch::torch_stack(lapply(seq_len(p),function(i) .diag(old$tangent(x,.diag(I[i,])))),dim=2)
    a<-.solve(A,.diag(z)$unsqueeze(2))$squeeze(2)
    old$tangent(x,.sym(u)-.diag(a))
  } else function(x,u) {z<-old$tangent(x,u);z-z$trace()*.rank_support(x,k)/k}
  M$name<-if(diagonal) "elliptope" else "spectrahedron"
  M$dimension<-M$dimension-if(diagonal) p else 1
  M$operations$tangent<-tan;M$operations$egrad2rgrad<-tan
  M$operations$belongs<-function(x,tol) old$belongs(x,tol) && (if(diagonal) .scalar((.diag(x)-1)$abs()$max()) else abs(.scalar(x$trace())-1))<=tol
  M$operations$retr<-function(x,u) norm(old$retr(x,u))
  M$operations$project<-function(x) norm(old$project(x))
  M$operations$random<-function(x) norm(old$random(x))
  M$operations$residual<-function(x) max(old$residual(x),if(diagonal) .scalar((.diag(x)-1)$abs()$max()) else abs(.scalar(x$trace())-1))
  M
}
#' Kendall Landmark Shape Geometry
#' @param k Number of labeled landmarks.
#' @param p Spatial dimension, with k > p.
#' @param reflections Whether to identify reflections as well as rotations.
#'   FALSE uses SO(p) (Kendall); TRUE uses O(p), matching Riemann's shape quotient.
#' @return Centered k by p preshapes of unit Frobenius norm, modulo SO(p).
#' @details The supported regular stratum has full column rank. Horizontal
#' tangents are centered, orthogonal to the preshape, and satisfy X' U symmetric.
#' Retraction normalizes a centered tangent step; transport projects horizontally.
#' Objectives must be invariant to rotations of the representative.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.landmark(5, 2)
#'   riem.belongs(M, riem.random(M))
#' }
#' @export
manifold.landmark <- function(k,p,reflections=FALSE) {
  if(!is.logical(reflections)||length(reflections)!=1||is.na(reflections)) .stop("reflections must be TRUE or FALSE")
  dims<-.positive_integer(c(k,p));if(length(dims)!=2||k<=p) .stop("Require k > p")
  center<-function(x) x-x$mean(dim=1,keepdim=TRUE)
  proj<-function(x) .normalize(center(x))
  tan<-function(x,u) {z<-center(u);z<-z-x*.dot(x,z);z-x$matmul(.sylvester(x$t()$matmul(x),x$t()$matmul(z)-z$t()$matmul(x)))}
  riem.manifold("landmark",c(k,p),(k-1)*p-1-p*(p-1)/2,"kendall",list(
    belongs=function(x,tol) .scalar(.fnorm(x$mean(dim=1)))<=tol && abs(.scalar(.dot(x,x))-1)<=tol && .scalar(torch::linalg_svdvals(x)$min())>tol,
    tangent=tan,inner=function(x,u,v) .dot(u,v),egrad2rgrad=tan,
    retr=function(x,u) proj(x+u),project=proj,
    residual=function(x) max(.scalar(.fnorm(x$mean(dim=1))),abs(.scalar(.dot(x,x))-1))),
    specification=list(k=k,p=p,reflections=reflections), primitive_derivatives=c(inner=2L), required_operations=c("basic","eigh","svd"))
}
