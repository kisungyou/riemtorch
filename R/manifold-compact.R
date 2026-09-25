.svd_shape <- function(M,x) {
  if(!is.list(x)||!identical(names(x),c("U","S","V"))) return(FALSE)
  shapes<-list(c(M$specification$m,M$specification$k),rep(M$specification$k,2),
    c(M$specification$p,M$specification$k))
  all(vapply(seq_along(x),function(i) inherits(x[[i]],"torch_tensor")&&
    identical(as.integer(x[[i]]$shape),as.integer(shapes[[i]])),logical(1)))
}
.svd_ambient_apply <- function(x,u,v,transpose=FALSE) {
  if(!transpose) x$U$matmul(u$S$matmul(x$V$t()$matmul(v)))+
    u$U$matmul(x$V$t()$matmul(v))+x$U$matmul(u$V$t()$matmul(v)) else
      x$V$matmul(u$S$t()$matmul(x$U$t()$matmul(v)))+
      x$V$matmul(u$U$t()$matmul(v))+u$V$matmul(x$U$t()$matmul(v))
}
.compact_fixedrank <- function(m,p,k) {
  dims<-.positive_integer(c(m,p,k));if(length(dims)!=3||k>min(m,p)) .stop("Require k <= min(m,p)")
  tangent<-function(x,u) {
    if(inherits(u,"torch_tensor")) {
      zv<-u$matmul(x$V);ztu<-u$t()$matmul(x$U);C<-x$U$t()$matmul(zv)
      return(list(U=zv-x$U$matmul(C),S=C,V=ztu-x$V$matmul(C$t())))
    }
    list(U=u$U-x$U$matmul(x$U$t()$matmul(u$U)),S=u$S,
      V=u$V-x$V$matmul(x$V$t()$matmul(u$V)))
  }
  retract<-function(x,u) {
    a<-torch::linalg_qr(torch::torch_cat(list(x$U,u$U),dim=2),mode="reduced")
    b<-torch::linalg_qr(torch::torch_cat(list(x$V,u$V),dim=2),mode="reduced")
    I<-.eye(x$S,k);Z<-torch::torch_zeros_like(I)
    K<-torch::torch_cat(list(torch::torch_cat(list(x$S+u$S,I),dim=2),
      torch::torch_cat(list(I,Z),dim=2)),dim=1)
    sv<-torch::linalg_svd(a[[2]]$matmul(K)$matmul(b[[2]]$t()),full_matrices=FALSE)
    d<-sv[[2]]$narrow(1,1,k)
    if(.scalar(d$min())<=0) .stop("Compact retraction reached a lower-rank stratum")
    list(U=a[[1]]$matmul(sv[[1]]$narrow(2,1,k)),S=torch::torch_diag(d),
      V=b[[1]]$matmul(sv[[3]]$narrow(1,1,k)$t()))
  }
  ops<-list(
    belongs=function(x,tol) .scalar(.fnorm(x$U$t()$matmul(x$U)-.eye(x$S,k)))<=tol&&
      .scalar(.fnorm(x$V$t()$matmul(x$V)-.eye(x$S,k)))<=tol&&.scalar(torch::linalg_svdvals(x$S)$min())>tol,
    tangent=tangent,inner=function(x,u,v) .tree_dot(u,v),
    egrad2rgrad=function(x,u) list(
      U=.solve(x$S,(u$U-x$U$matmul(x$U$t()$matmul(u$U)))$t())$t(),
      S=u$S,V=.solve(x$S$t(),(u$V-x$V$matmul(x$V$t()$matmul(u$V)))$t())$t()),
    retr=retract,
    transport=function(x,u,y,v) {
      zv<-.svd_ambient_apply(x,v,y$V);ztu<-.svd_ambient_apply(x,v,y$U,TRUE)
      C<-y$U$t()$matmul(zv)
      list(U=zv-y$U$matmul(C),S=C,V=ztu-y$V$matmul(C$t()))
    },
    random=function(template) {
      ref<-template
      U<-torch::linalg_qr(torch::torch_randn(c(m,k),dtype=ref$dtype,device=ref$device))[[1]]
      V<-torch::linalg_qr(torch::torch_randn(c(p,k),dtype=ref$dtype,device=ref$device))[[1]]
      list(U=U,S=.eye(ref,k),V=V)
    })
  M<-riem.manifold("fixedrank",1,k*(m+p-k),"embedded",ops,
    specification=list(m=m,p=p,k=k,representation="svd"),
    primitive_derivatives=c(inner=2L),required_operations=c("basic","qr","svd","solve"),
    second_order_retraction=TRUE)
  M$shape<-NULL;class(M)<-c("riem_svd","riem_manifold");M
}
#' Materialize or Extract Entries from a Compact Matrix Point
#' @param manifold A fixed-rank geometry with representation `"svd"`.
#' @param x A named list U, S, V representing U S V'. S is a full rank square
#'   core; allowing a full core makes autodiff valid at repeated singular values.
#' @param rows,columns Equal-length one-based integer entry indices. NULL for
#'   both materializes the complete matrix.
#' @return A matrix tensor, or a vector of requested entries without allocating
#'   the full matrix. Tangents use U, S, V leaves for U_perp, core, V_perp, and
#'   represent U_perp V' + U core V' + U V_perp'. They are not point increments.
#' @details Objectives must depend only on the represented matrix, independently
#'   of the choice of orthonormal factors. Compact points currently support one
#'   solve at a time and first-order derivatives; no automatic Hessian is claimed.
#' @examples
#' if(torch::torch_is_installed()) {
#'   M <- manifold.fixedrank(10,8,2,representation="svd")
#'   x <- riem.random(M,device="cpu")
#'   riem.materialize(M,x,rows=c(1,4),columns=c(2,5))
#' }
#' @export
riem.materialize <- function(manifold,x,rows=NULL,columns=NULL) {
  if(!inherits(manifold,"riem_svd")||!.svd_shape(manifold,x)) .stop("Expected a compact fixed-rank point")
  if(is.null(rows)&&is.null(columns)) return(x$U$matmul(x$S)$matmul(x$V$t()))
  rows<-.positive_integer(rows,"rows");columns<-.positive_integer(columns,"columns")
  if(length(rows)!=length(columns)||any(rows>manifold$specification$m)||any(columns>manifold$specification$p))
    .stop("Entry indices must have equal lengths and lie within matrix dimensions")
  (x$U[rows,,drop=FALSE]$matmul(x$S)*x$V[columns,,drop=FALSE])$sum(dim=2)
}
