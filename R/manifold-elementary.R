#' Oblique Matrices and the Fisher--Rao Simplex
#' @param p Ambient dimension (number of rows or probabilities).
#' @param k Number of oblique columns.
#' @param metric Simplex metric, currently only `"fisher_rao"`.
#' @return A geometry specification.
#' @details Oblique columns have unit norm and the product round metric.
#' The simplex has strictly positive entries summing to one and metric
#' sum(u*v/x), equivalent to a radius-two sphere under `2*sqrt(x)`.
#' The simplex retraction is an exponential-coordinate normalization.
#' Both geometries provide exact exponential and local logarithm maps. The
#' simplex exponential rejects geodesics that leave its positive orthant.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.oblique(3, 2)
#'   x <- riem.random(M)
#'   u <- riem.tangent(M, x, torch::torch_randn_like(x)) * 0.1
#'   riem.log(M, x, riem.exp(M, x, u))
#'
#'   S <- manifold.multinomial(4)
#'   riem.random(S)
#' }
#' @export
manifold.oblique <- function(p,k) {
  s<-c(.positive_integer(p),.positive_integer(k));if(length(s)!=2) .stop("p and k must be scalars")
  normcols<-function(x) {n<-(x*x)$sum(dim=1,keepdim=TRUE)$sqrt();if(.scalar(n$min())<=0) .stop("Zero column");x/n}
  tan<-function(x,u) u-x*(x*u)$sum(dim=1,keepdim=TRUE)
  expmap<-function(x,u) {
    n2<-(u*u)$sum(dim=1,keepdim=TRUE);small<-n2<1e-8
    safe<-torch::torch_where(small,torch::torch_ones_like(n2),n2)$sqrt()
    co<-torch::torch_where(small,-n2/2+n2^2/24+1,safe$cos())
    si<-torch::torch_where(small,-n2/6+n2^2/120+1,safe$sin()/safe)
    x*co+u*si
  }
  logmap<-function(x,y) {
    co<-(x*y)$sum(dim=1,keepdim=TRUE)
    if(.scalar(co$min()) < -1+1e-10) .stop("Oblique logarithm is undefined when a column is antipodal")
    co<-co$clamp(-1,1)
    z<-1-co;near<-z$abs()<1e-5
    safe<-torch::torch_where(near,torch::torch_zeros_like(co),co$clamp(-1+1e-15,1-1e-15))
    ratio<-torch::torch_where(near,z/3+z^2*(2/15)+1,
      torch::torch_acos(safe)/(1-safe^2)$sqrt())
    (y-x*co)*ratio
  }
  riem.manifold("oblique",s,(p-1)*k,"column_round",list(
    belongs=function(x,tol) .scalar(((x*x)$sum(dim=1)-1)$abs()$max())<=tol,
    tangent=tan,inner=function(x,u,v) .dot(u,v),egrad2rgrad=tan,
    retr=function(x,u) normcols(x+u),project=normcols,
    exp=expmap,log=logmap,
    sqdist=function(x,y) .acos_sq((x*y)$sum(dim=1))$sum(),
    ehess2rhess=function(x,u,egrad,ehess)
      tan(x,ehess-u*(x*egrad)$sum(dim=1,keepdim=TRUE)),
    residual=function(x) .scalar(((x*x)$sum(dim=1)-1)$abs()$max())),
    list(hessian=TRUE,curve_derivative=TRUE),list(p=p,k=k), primitive_derivatives=setNames(rep(2L,9),c("tangent","inner","egrad2rgrad","retr","transport","exp","log","sqdist","project")), required_operations=c("basic"), second_order_retraction=TRUE)
}
#' @rdname manifold.oblique
#' @export
manifold.multinomial <- function(p,metric="fisher_rao") {
  p<-.positive_integer(p);if(length(p)!=1 || p<2) .stop("p must be at least two")
  if(!identical(metric,"fisher_rao")) .stop("Only fisher_rao is available")
  expmap<-function(x,u) {
    q<-x$sqrt();v<-u/(q*2);r2<-.dot(v,v);r<-r2$clamp_min(0)$sqrt()
    if(.scalar(r)>0) {
      # In unit-sphere angle s, coordinate i first reaches zero at
      # atan2(q_i, -v_i / ||v||), which lies in (0, pi).  Checking this exit
      # time rejects paths that cross the boundary and later re-enter it.
      exit<-torch::torch_atan2(q,-v/r)$min()
      if(.scalar(r)>=.scalar(exit)-1e-12)
        .stop("Simplex exponential leaves the positive orthant")
    }
    z<-.sphere_exp(q,v)
    if(.scalar(z$min())<=0) .stop("Simplex exponential leaves the positive orthant")
    z^2
  }
  logmap<-function(x,y) x$sqrt()*.sphere_log(x$sqrt(),y$sqrt())*2
  riem.manifold("multinomial",p,p-1,metric,list(
    belongs=function(x,tol) .scalar(x$min())>0 && abs(.scalar(x$sum())-1)<=tol,
    tangent=function(x,u) u-x*u$sum(),inner=function(x,u,v) (u*v/x)$sum(),
    egrad2rgrad=function(x,u) x*(u-.dot(x,u)),
    retr=function(x,u) torch::nnf_softmax(x$log()+u/x,dim=1),
    project=function(x) {z<-x$clamp_min(1e-8);z/z$sum()},
    random=function(x) torch::nnf_softmax(torch::torch_randn_like(x),dim=1),
    exp=expmap,log=logmap,
    sqdist=function(x,y) .acos_sq(.dot(x$sqrt(),y$sqrt()))*4,
    residual=function(x) max(abs(.scalar(x$sum())-1),max(0,-.scalar(x$min())))),
    list(curve_derivative=TRUE),list(p=p), primitive_derivatives=setNames(rep(2L,9),c("tangent","inner","egrad2rgrad","retr","transport","exp","log","sqdist","project")), required_operations=c("basic"), second_order_retraction=FALSE)
}
#' Flat Torus in Angle Coordinates
#' @param d Number of circular factors.
#' @return A manifold with d real angles, each interpreted modulo 2*pi.
#' @examples
#' if (torch::torch_is_installed()) riem.random(manifold.torus(2))
#' @export
manifold.torus <- function(d) {
  d<-.positive_integer(d);if(length(d)!=1) .stop("d must be scalar")
  M<-manifold.euclidean(d);M$name<-"torus";M$metric<-"flat_angles"
  M$operations$log<-function(x,y) torch::torch_atan2((y-x)$sin(),(y-x)$cos())
  M$operations$sqdist<-function(x,y) {z<-torch::torch_atan2((y-x)$sin(),(y-x)$cos());.dot(z,z)}
  M$operations$random<-function(x) 2*pi*torch::torch_rand_like(x)-pi
  M$specification<-list(d=d);M
}
.lorentz<-function(x,y) .dot(x,y)-x[1]*y[1]*2
.acosh_sq<-function(c) {
  c<-c$clamp_min(1);z<-c-1;small<-z$abs()<1e-4
  safe<-torch::torch_where(small,torch::torch_ones_like(c)*2,c$clamp_min(1+1e-15))
  torch::torch_where(small,z*2-z^2/3+z^3*4/45-z^4/35,torch::torch_acosh(safe)^2)
}
.mobius_add<-function(x,y,c) {
  xy<-.dot(x,y);x2<-.dot(x,x);y2<-.dot(y,y)
  (x*(xy*(2*c)+y2*c+1)+y*(-x2*c+1))/(xy*(2*c)+x2*y2*c^2+1)
}
#' Hyperbolic Ball and Hyperboloid
#' @param d Intrinsic dimension.
#' @param model `"poincare"` or `"hyperboloid"`.
#' @param curvature Strictly negative sectional curvature.
#' @return A manifold with d ball coordinates or d+1 hyperboloid coordinates.
#' @details Hyperboloid uses signature (-,+,...,+) and the positive-time sheet.
#' Ball metric is 4/(1-c*||x||^2)^2 times Euclidean, where c=-curvature.
#' Retractions are local additive (ball) or timelike normalization (hyperboloid).
#' Exact exponential and logarithm maps are also available for both models.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.hyperbolic(2, "hyperboloid")
#'   x <- riem.random(M)
#'   u <- riem.tangent(M, x, torch::torch_randn_like(x)) * 0.05
#'   y <- riem.exp(M, x, u)
#'   riem.log(M, x, y)
#' }
#' @export
manifold.hyperbolic <- function(d,model=c("poincare","hyperboloid"),curvature=-1) {
  d<-.positive_integer(d);if(length(d)!=1) .stop("d must be scalar")
  model<-match.arg(model);c<--curvature;.number(c,"negative curvature")
  if(model=="poincare") {
    project<-function(x) x/(1+.fnorm(x)*sqrt(c))
    expmap<-function(x,u) {
      n2<-.dot(u,u);n<-n2$clamp_min(0)$sqrt();small<-n2<1e-16
      safe<-torch::torch_where(small,torch::torch_ones_like(n),n)
      lambda<-2/(-.dot(x,x)*c+1);a<-lambda*n*(sqrt(c)/2)
      coefficient<-torch::torch_where(small,lambda/2,torch::torch_tanh(a)/(safe*sqrt(c)))
      y<-.mobius_add(x,u*coefficient,c)
      if(.scalar(.dot(y,y)*c)>=1) .stop("Poincare exponential reached the numerical boundary")
      y
    }
    logmap<-function(x,y) {
      delta<-.mobius_add(-x,y,c);n2<-.dot(delta,delta);n<-n2$clamp_min(0)$sqrt();small<-n2<1e-16
      safe<-torch::torch_where(small,torch::torch_ones_like(n),n)
      z<-n*sqrt(c)
      if(.scalar(z)>=1) .stop("Poincare logarithm requires points inside the same ball")
      lambda<-2/(-.dot(x,x)*c+1)
      coefficient<-torch::torch_where(small,2/lambda,
        torch::torch_atanh(z)*2/(safe*lambda*sqrt(c)))
      delta*coefficient
    }
    ops<-list(belongs=function(x,tol) .scalar(.dot(x,x)*c)<1,
      tangent=function(x,u) u,inner=function(x,u,v) .dot(u,v)*4/(-.dot(x,x)*c+1)^2,
      egrad2rgrad=function(x,u) u*(-.dot(x,x)*c+1)^2/4,
      retr=function(x,u) {y<-x+u;if(.scalar(.dot(y,y)*c)>=1) .stop("Step leaves Poincare ball");y},
      project=project,random=function(x) project(torch::torch_randn_like(x)),
      exp=expmap,log=logmap,
      sqdist=function(x,y) .acosh_sq(.dot(x-y,x-y)*2*c/((-.dot(x,x)*c+1)*(-.dot(y,y)*c+1))+1)/c,
      residual=function(x) max(0,.scalar(.dot(x,x)*c)-1))
    shape<-d
  } else {
    project<-function(x) {v<-x$narrow(1,2,d);torch::torch_cat(list((.dot(v,v)+1/c)$sqrt()$reshape(1),v))}
    tan<-function(x,u) u+x*(.lorentz(x,u)*c)
    expmap<-function(x,u) {
      r2<-.lorentz(u,u)$clamp_min(0);r<-r2$sqrt();a<-r*sqrt(c);small<-r2<1e-16
      coefficient<-torch::torch_where(small,a^2/6+a^4/120+1,
        torch::torch_sinh(a)/torch::torch_where(small,torch::torch_ones_like(a),a))
      x*torch::torch_cosh(a)+u*coefficient
    }
    logmap<-function(x,y) {
      alpha<-.lorentz(x,y)*(-c);z<-alpha-1
      if(.scalar(z) < -1e-10) .stop("Hyperboloid logarithm requires points on the positive-time sheet")
      alpha<-alpha$clamp_min(1);z<-alpha-1
      near<-z$abs()<1e-5
      safe<-torch::torch_where(near,torch::torch_ones_like(alpha)*2,alpha$clamp_min(1+1e-15))
      coefficient<-torch::torch_where(near,-z/3+z^2*(2/15)+1,
        torch::torch_acosh(safe)/(safe^2-1)$sqrt())
      (y-x*alpha)*coefficient
    }
    ops<-list(belongs=function(x,tol) .scalar(x[1])>0 && abs(.scalar(.lorentz(x,x))+1/c)<=tol,
      tangent=tan,inner=function(x,u,v) .lorentz(u,v),
      egrad2rgrad=function(x,u) {sign<-torch::torch_cat(list(torch::torch_full(c(1),-1,dtype=x$dtype,device=x$device),torch::torch_ones(d,dtype=x$dtype,device=x$device)));tan(x,u*sign)},
      retr=function(x,u) {z<-x+u;n<-.lorentz(z,z)*(-c);if(.scalar(n)<=0||.scalar(z[1])<=0) .stop("Step leaves timelike sheet");z/n$sqrt()},
      project=project,random=function(x) project(torch::torch_randn_like(x)),
      exp=expmap,log=logmap,
      sqdist=function(x,y) .acosh_sq(.lorentz(x,y)*(-c))/c,
      residual=function(x) abs(.scalar(.lorentz(x,x))+1/c))
    shape<-d+1L
  }
  riem.manifold("hyperbolic",shape,d,paste0(model,"_",curvature),ops,
                list(curve_derivative=TRUE),list(d=d,model=model,curvature=curvature), primitive_derivatives=setNames(rep(2L,9),c("tangent","inner","egrad2rgrad","retr","transport","exp","log","sqdist","project")), required_operations=c("basic"), second_order_retraction=FALSE)
}
#' Generalized Orthogonality Geometries
#' @param p,k Ambient and frame dimensions.
#' @param B Fixed symmetric positive-definite torch tensor of size p by p.
#' @return A geometry with constraint X' B X=I and metric tr(U' B V).
#' @details Whitening uses the transpose of the Cholesky factor. B must already
#' share the point's device and dtype. Grassmann points identify frames differing
#' by right orthogonal transformations. B is copied at construction. The
#' whitening isometry also supplies exact ambient Hessian conversion.
#' @examples
#' if (torch::torch_is_installed()) {
#'   B <- torch::torch_eye(3, dtype = torch::torch_float64())
#'   M <- manifold.stiefel.generalized(3, 2, B)
#'   riem.random(M)
#' }
#' @export
manifold.stiefel.generalized <- function(p,k,B) .generalized(p,k,B,FALSE)
#' @rdname manifold.stiefel.generalized
#' @export
manifold.grassmann.generalized <- function(p,k,B) .generalized(p,k,B,TRUE)
.generalized<-function(p,k,B,grassmann) {
  s<-.frame_dims(p,k)
  if(!all(riem.belongs(manifold.spd(p,"airm"),B))) .stop("B must be a p by p SPD tensor")
  B<-.clone(B);W<-torch::linalg_cholesky(B)$t();base<-.frame(p,k,"euclidean",grassmann)
  chk<-function(x) {if(!(x$dtype == B$dtype)||!identical(x$device$type,B$device$type)||!identical(x$device$index,B$device$index)) .stop("B and points must share dtype/device")}
  tx<-function(x) {chk(x);W$matmul(x)}
  inv<-function(x) .solve(W,x)
  dual<-function(x) .solve(W$t(),x)
  riem.manifold(if(grassmann) "grassmann.generalized" else "stiefel.generalized",s,base$dimension,"B_euclidean",list(
    belongs=function(x,tol) base$operations$belongs(tx(x),tol),
    tangent=function(x,u) inv(base$operations$tangent(tx(x),tx(u))),
    inner=function(x,u,v) .dot(tx(u),tx(v)),
    egrad2rgrad=function(x,u) inv(base$operations$egrad2rgrad(tx(x),dual(u))),
    retr=function(x,u) inv(base$operations$retr(tx(x),tx(u))),
    project=function(x) inv(.polar(tx(x))),
    ehess2rhess=function(x,u,egrad,ehess)
      inv(base$operations$ehess2rhess(tx(x),tx(u),dual(egrad),dual(ehess))),
    residual=function(x) .scalar(.fnorm(x$t()$matmul(B)$matmul(x)-.eye(x,k)))),
    list(hessian=TRUE,curve_derivative=TRUE),list(p=p,k=k,B=B), primitive_derivatives=c(tangent=2L,inner=2L,egrad2rgrad=2L,transport=2L,retr=1L,project=1L), required_operations=c("basic","eigh","cholesky","svd"), second_order_retraction=TRUE)
}
#' Rotations and Rigid Motions
#' @param p Rotation matrix size.
#' @param d Spatial dimension of a rigid motion.
#' @return A rotation manifold SO(p), or a named product `rotation`, `translation`.
#' @details SO(p) uses the embedded Frobenius metric, polar retraction, exact
#' ambient Hessian conversion, and the matrix-exponential geodesic.
#' Rigid motions use the product of that metric and the Euclidean translation
#' metric; this is a Riemannian product, not a bi-invariant SE(d) metric.
#' @examples
#' if (torch::torch_is_installed()) {
#'   riem.random(manifold.rotation(3))
#'   riem.random(manifold.rigidmotion(2))
#' }
#' @export
manifold.rotation <- function(p) {
  p<-.positive_integer(p);if(length(p)!=1||p<2) .stop("p must be at least two")
  M<-.frame(p,p,"euclidean");M$name<-"rotation";M$metric<-"frobenius"
  M$operations$belongs<-function(x,tol) .scalar(.fnorm(x$t()$matmul(x)-.eye(x)))<=tol && .scalar(torch::linalg_det(x))>0
  M$operations$project<-function(x) {s<-torch::linalg_svd(x);u<-s[[1]];v<-s[[3]];d<-torch::torch_ones(p,dtype=x$dtype,device=x$device);d[p]<-torch::linalg_det(u$matmul(v));u$matmul(.diag(d))$matmul(v)}
  M$operations$exp<-function(x,u) x$matmul(torch::torch_matrix_exp(.skew(x$t()$matmul(u))))
  M
}
#' @rdname manifold.rotation
#' @export
manifold.rigidmotion <- function(d) {
  M<-manifold.product(list(rotation=manifold.rotation(d),translation=manifold.euclidean(d)))
  M$name<-"rigidmotion";M$specification$d<-d;M
}
