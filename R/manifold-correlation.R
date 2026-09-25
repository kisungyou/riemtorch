.corr_normalize<-function(x) {d<-.diag(x)$sqrt();.sym(x/(d$unsqueeze(1)*d$unsqueeze(2)))}
.corr_theta<-function(x) {L<-torch::linalg_cholesky(x);L/.diag(L)$unsqueeze(2)}
.nil_log<-function(L) {
  p<-L$shape[1];N<-L-.eye(L);power<-N;z<-torch::torch_zeros_like(L)
  if(p>1) for(i in seq_len(p-1)) {z<-z+(-1)^(i+1)*power/i;power<-power$matmul(N)}
  z$tril(-1)
}
.nil_exp<-function(z) {
  p<-z$shape[1];z<-z$tril(-1);out<-.eye(z);term<-out
  if(p>1) for(i in seq_len(p-1)) {term<-term$matmul(z)/i;out<-out+term}
  out
}
#' Full-Rank Correlation Geometries
#' @param p Matrix size, at least two.
#' @param metric Required metric: `"ecm"`, `"lec"`, or `"affine_quotient"`.
#' @return A manifold of positive-definite p by p matrices with unit diagonal.
#' @details Let L=chol(C) be lower triangular and Theta(C)=diag(L)^-1 L.
#' ECM pulls back the Frobenius metric on the strictly lower part of Theta.
#' LEC uses the strictly lower nilpotent log(Theta). Their inverse chart forms
#' L L' from the unit-lower matrix and normalizes its diagonal. Retractions and
#' transports are exact chart operations. The affine quotient uses the AIRM
#' quotient by positive diagonal congruences, with a horizontal lift and a
#' normalized SPD retraction. No approximate chart is substituted for this metric.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.correlation(3, "ecm")
#'   x <- riem.random(M)
#'   riem.belongs(M, x)
#'   riem.chart(M, x)
#' }
#' @export
manifold.correlation <- function(p,metric) {
  p<-.positive_integer(p);if(length(p)!=1||p<2) .stop("p must be at least two")
  metric<-match.arg(metric,c("ecm","lec","affine_quotient"))
  tangent<-function(x,u) {z<-.sym(u);z-.diag(.diag(z))}
  ops<-list(belongs=function(x,tol) .scalar(.fnorm(x-x$t()))<=tol &&
    .scalar((.diag(x)-1)$abs()$max())<=tol && .scalar(torch::linalg_eigvalsh(x)$min())>0,
    tangent=tangent,project=function(x) .corr_normalize(manifold.spd(p,"airm")$operations$project(x)),
    random=function(x) {a<-torch::torch_randn_like(x);.corr_normalize(a$matmul(a$t())+.eye(x))},
    residual=function(x) max(.scalar(.fnorm(x-x$t())),.scalar((.diag(x)-1)$abs()$max()),max(0,-.scalar(torch::linalg_eigvalsh(.sym(x))$min()))))
  if(metric!="affine_quotient") {
    chart<-if(metric=="ecm") function(x) .corr_theta(x)$tril(-1) else function(x) .nil_log(.corr_theta(x))
    inverse<-if(metric=="ecm") function(z) {a<-z$tril(-1)+.eye(z);.corr_normalize(a$matmul(a$t()))} else
      function(z) {a<-.nil_exp(z);.corr_normalize(a$matmul(a$t()))}
    ops$chart<-chart;ops$chart_inverse<-inverse
    ops$inner<-function(x,u,v) .dot(.jvp(chart,x,u),.jvp(chart,x,v))
    ops$egrad2rgrad<-function(x,u) torch::with_enable_grad({
      z<-chart(x)$detach()$requires_grad_(TRUE)
      g<-torch::autograd_grad(.dot(inverse(z),u),z)[[1]]
      .jvp(inverse,z,g)
    })
    ops$retr<-function(x,u) inverse(chart(x)+.jvp(chart,x,u))
    ops$exp<-ops$retr
    ops$log<-function(x,y) .jvp(inverse,chart(x),chart(y)-chart(x))
    ops$sqdist<-function(x,y) {z<-chart(x)-chart(y);.dot(z,z)}
    ops$transport<-function(x,u,y,v) .jvp(inverse,chart(y),.jvp(chart,x,v))
  } else {
    lift<-function(x,u) {inv<-.solve(x,.eye(x));a<-.solve(.eye(x)+inv*x$t(),-.diag(inv$matmul(u))$unsqueeze(2))$squeeze(2);D<-.diag(a);.sym(u+D$matmul(x)+x$matmul(D))}
    ops$inner<-function(x,u,v) .dot(.solve(x,lift(x,u)),.solve(x,lift(x,v))$t())
    ops$egrad2rgrad<-function(x,u) {g<-.sym(u);h<-g-.diag(.diag(g$matmul(x)));v<-x$matmul(h)$matmul(x);d<-.diag(v);tangent(x,v-(d$unsqueeze(1)+d$unsqueeze(2))*x/2)}
    ops$retr<-function(x,u) {v<-lift(x,u);.corr_normalize(.sym(x+v+v$matmul(.solve(x,v))/2))}
  }
  riem.manifold("correlation",c(p,p),p*(p-1)/2,metric,ops,
    list(snapshot_transport=TRUE,isometric_transport=metric!="affine_quotient"),list(p=p), primitive_derivatives=c(tangent=2L), required_operations=c("basic","eigh","cholesky","matrix_exp"))
}
#' Correlation Chart Coordinates
#' @param manifold An ECM or LEC correlation manifold.
#' @param x A correlation point or batch of points.
#' @param z Strictly lower triangular chart coordinates, with square point axes.
#' @return A tensor of chart coordinates or correlation matrices.
#' @details Both chart and inverse use native differentiable torch operations.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.correlation(2, "lec")
#'   x <- riem.random(M)
#'   riem.chart.inverse(M, riem.chart(M, x))
#' }
#' @export
riem.chart <- function(manifold,x) .geom(manifold,"chart",x)
#' @rdname riem.chart
#' @export
riem.chart.inverse <- function(manifold,z) .geom(manifold,"chart_inverse",z)
