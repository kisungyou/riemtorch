.batch_dims <- function(M, x) {
  q <- length(M$shape)
  seq.int(length(x$shape) - q + 1L, length(x$shape))
}
.batch_shape <- function(M, x) head(as.integer(x$shape), -length(M$shape))
.batch_sum <- function(M, x, keepdim = FALSE) x$sum(dim = .batch_dims(M, x), keepdim = keepdim)
.batch_dot <- function(M, x, y) .batch_sum(M, x * y)
.batch_expand <- function(M, x, scalar) {
  scalar$reshape(c(.batch_shape(M, x), rep(1L, length(M$shape))))
}
.batch_norm <- function(M, x) .batch_dot(M, x, x)$clamp_min(0)$sqrt()
.batch_trace <- function(x) torch::torch_diagonal(x, dim1 = -2, dim2 = -1)$sum(dim = -1)
.batch_polar <- function(x) x$matmul(riem.matrix.function(.mt(x)$matmul(x), "invsqrt"))
.batch_geom <- function(M, op, args) {
  x <- args[[1L]]; s <- as.integer(x$shape)
  for (z in args) {
    .check_tensor(M, z)
    if (!identical(as.integer(z$shape), s))
      .stop("Batch and point shapes must match exactly; broadcasting is not implicit")
    if (!(z$dtype == x$dtype) || !identical(z$device$type, x$device$type) ||
        !identical(z$device$index, x$device$index))
      .stop("Tensor dtype and device must agree")
  }
  do.call(op, args)
}

# Native leading-batch kernels for the most frequently used built-in geometries.
# Membership and residual checks intentionally keep the conservative pointwise
# path because they return host-side logical values used by solver control flow.
.builtin_batch_operations <- function(M) {
  name <- M$name; metric <- M$metric
  if (name %in% c("euclidean", "torus")) {
    ops <- list(
      tangent = function(x, u) u,
      inner = function(x, u, v) .batch_dot(M, u, v),
      egrad2rgrad = function(x, u) u,
      retr = function(x, u) x + u,
      transport = function(x, u, y, v) v,
      exp = function(x, u) x + u,
      log = function(x, y) y - x,
      sqdist = function(x, y) .batch_dot(M, y - x, y - x),
      project = function(x) x,
      random = function(x) torch::torch_randn_like(x),
      ehess2rhess = function(x, u, egrad, ehess) ehess)
    if (name == "torus") {
      ops$log <- function(x, y) torch::torch_atan2((y-x)$sin(), (y-x)$cos())
      ops$sqdist <- function(x, y) {
        z <- torch::torch_atan2((y-x)$sin(), (y-x)$cos()); .batch_dot(M, z, z)
      }
      ops$random <- function(x) x$uniform_(-pi, pi)
    }
    return(ops)
  }
  if (name == "sphere") {
    tangent <- function(x, u) u - x * .batch_expand(M, x, .batch_dot(M, x, u))
    normalize <- function(x) {
      n <- .batch_norm(M, x)
      if (.scalar(n$min()) <= 0) .stop("Cannot normalize a zero point")
      x / .batch_expand(M, x, n)
    }
    exp <- function(x, u) {
      n2 <- .batch_dot(M, u, u); small <- n2 < 1e-8
      safe <- torch::torch_where(small, torch::torch_ones_like(n2), n2)$sqrt()
      co <- torch::torch_where(small, -n2/2 + n2^2/24 + 1, safe$cos())
      si <- torch::torch_where(small, -n2/6 + n2^2/120 + 1, safe$sin()/safe)
      x * .batch_expand(M, x, co) + u * .batch_expand(M, x, si)
    }
    log <- function(x, y) {
      cc <- .batch_dot(M, x, y)
      if (.scalar(cc$min()) < -1 + 1e-10) .stop("Sphere logarithm is undefined at antipodes")
      z <- -cc + 1; near <- z$abs() < 1e-5
      safe <- torch::torch_where(near, torch::torch_zeros_like(cc), cc$clamp(-1+1e-15, 1-1e-15))
      ratio <- torch::torch_where(near, z/3 + z^2*2/15 + 1,
        torch::torch_acos(safe)/(-safe^2+1)$sqrt())
      (y - x * .batch_expand(M, x, cc)) * .batch_expand(M, x, ratio)
    }
    return(list(tangent=tangent, inner=function(x,u,v) .batch_dot(M,u,v),
      egrad2rgrad=tangent, retr=function(x,u) normalize(x+u), project=normalize,
      exp=exp, log=log, sqdist=function(x,y) .acos_sq(.batch_dot(M,x,y)),
      transport=function(x,u,y,v) tangent(y,v),
      ehess2rhess=function(x,u,egrad,ehess)
        tangent(x,ehess)-u*.batch_expand(M,x,.batch_dot(M,x,egrad)),
      random=function(x) normalize(torch::torch_randn_like(x))))
  }
  if (name == "oblique") {
    rowdim <- -2L
    colsum <- function(z) z$sum(dim=rowdim, keepdim=TRUE)
    normcols <- function(x) {
      n <- colsum(x*x)$sqrt()
      if (.scalar(n$min()) <= 0) .stop("Zero column")
      x/n
    }
    tangent <- function(x,u) u-x*colsum(x*u)
    exp <- function(x,u) {
      n2 <- colsum(u*u); small <- n2 < 1e-8
      safe <- torch::torch_where(small,torch::torch_ones_like(n2),n2)$sqrt()
      co <- torch::torch_where(small,-n2/2+n2^2/24+1,safe$cos())
      si <- torch::torch_where(small,-n2/6+n2^2/120+1,safe$sin()/safe)
      x*co+u*si
    }
    log <- function(x,y) {
      cc <- colsum(x*y)
      if (.scalar(cc$min()) < -1+1e-10) .stop("Oblique logarithm is undefined at a column antipode")
      z <- -cc+1; near <- z$abs()<1e-5
      safe <- torch::torch_where(near,torch::torch_zeros_like(cc),cc$clamp(-1+1e-15,1-1e-15))
      ratio <- torch::torch_where(near,z/3+z^2*2/15+1,
        torch::torch_acos(safe)/(-safe^2+1)$sqrt())
      (y-x*cc)*ratio
    }
    return(list(tangent=tangent,inner=function(x,u,v) .batch_dot(M,u,v),
      egrad2rgrad=tangent,retr=function(x,u) normcols(x+u),project=normcols,
      exp=exp,log=log,sqdist=function(x,y) .acos_sq(colsum(x*y)$squeeze(-2))$sum(dim=-1),
      transport=function(x,u,y,v) tangent(y,v),
      ehess2rhess=function(x,u,egrad,ehess) tangent(x,ehess)-u*colsum(x*egrad),
      random=function(x) normcols(torch::torch_randn_like(x))))
  }
  if (name == "multinomial") {
    expand <- function(x,z) z$unsqueeze(-1)
    tangent <- function(x,u) u-x*expand(x,u$sum(dim=-1))
    inner <- function(x,u,v) (u*v/x)$sum(dim=-1)
    return(list(tangent=tangent,inner=inner,
      egrad2rgrad=function(x,u) x*(u-expand(x,(x*u)$sum(dim=-1))),
      retr=function(x,u) torch::nnf_softmax(x$log()+u/x,dim=-1),
      project=function(x) {z<-x$clamp_min(1e-8);z/expand(z,z$sum(dim=-1))},
      random=function(x) torch::nnf_softmax(torch::torch_randn_like(x),dim=-1),
      sqdist=function(x,y) .acos_sq((x$sqrt()*y$sqrt())$sum(dim=-1))*4))
  }
  if (name %in% c("stiefel", "grassmann", "rotation") &&
      metric != "projection_half_frobenius") {
    grass <- name == "grassmann"
    tangent <- if (grass) function(x,u) u-x$matmul(.mt(x)$matmul(u)) else
      function(x,u) u-x$matmul(.sym(.mt(x)$matmul(u)))
    eg <- if(metric=="canonical"&&!grass) function(x,u) u-x$matmul(.mt(u)$matmul(x)) else tangent
    inner <- if(metric=="canonical"&&!grass) function(x,u,v)
      .batch_dot(M,u,v)-(.mt(x)$matmul(u)*.mt(x)$matmul(v))$sum(dim=c(-2,-1))/2 else
      function(x,u,v) .batch_dot(M,u,v)
    ops <- list(tangent=tangent,inner=inner,egrad2rgrad=eg,
      retr=function(x,u) .batch_polar(x+u),project=.batch_polar,
      transport=function(x,u,y,v) tangent(y,v),random=function(x) .batch_polar(torch::torch_randn_like(x)))
    if(grass) ops$ehess2rhess <- function(x,u,egrad,ehess)
      tangent(x,ehess-u$matmul(.mt(x)$matmul(egrad)))
    if(!grass&&metric=="euclidean") ops$ehess2rhess <- function(x,u,egrad,ehess)
      tangent(x,ehess-u$matmul(.sym(.mt(x)$matmul(egrad))))
    return(ops)
  }
  if (name == "grassmann" && metric == "projection_half_frobenius") {
    p <- M$specification$p; k <- M$specification$k
    tangent <- function(x,u) {u<-.sym(u);x$matmul(u)+u$matmul(x)-2*x$matmul(u)$matmul(x)}
    project <- function(x) {
      e <- torch::linalg_eigh(.sym(x));q <- e[[2]]$narrow(-1,p-k+1,k);q$matmul(.mt(q))
    }
    ehess <- function(x,u,egrad,ambient_hess) {
      g<-.sym(egrad);h<-.sym(ambient_hess)
      dg<-u$matmul(g)+g$matmul(u)-2*u$matmul(g)$matmul(x)-
        2*x$matmul(g)$matmul(u)+tangent(x,h)
      tangent(x,2*dg)
    }
    random <- function(x) {
      b <- head(as.integer(x$shape),-2L)
      q <- .batch_polar(torch::torch_randn(c(b,p,k),dtype=x$dtype,device=x$device))
      q$matmul(.mt(q))
    }
    return(list(tangent=tangent,inner=function(x,u,v) .batch_dot(M,u,v)/2,
      egrad2rgrad=function(x,u) 2*tangent(x,u),retr=function(x,u) project(x+u),
      project=project,transport=function(x,u,y,v) tangent(y,v),
      ehess2rhess=ehess,random=random))
  }
  if (name == "spd") {
    mf<-riem.matrix.function;df<-riem.matrix.frechet
    project<-function(x) {e<-torch::linalg_eigh(.sym(x));
      (e[[2]]*e[[1]]$clamp_min(1e-8)$unsqueeze(-2))$matmul(.mt(e[[2]]))}
    random<-function(x) {a<-torch::torch_randn_like(x);a$matmul(.mt(a))/M$specification$p+.eye(x)}
    ops<-list(tangent=function(x,u).sym(u),project=project,random=random)
    if(metric=="airm") {
      ops$inner<-function(x,u,v) .batch_dot(M,.solve(x,u),.mt(.solve(x,v)))
      ops$egrad2rgrad<-function(x,u) .sym(x$matmul(.sym(u))$matmul(x))
      ops$retr<-function(x,u) .sym(x+u+u$matmul(.solve(x,u))/2)
      ops$exp<-function(x,u){s<-mf(x,"sqrt");r<-mf(x,"invsqrt");.sym(s$matmul(mf(.sym(r$matmul(u)$matmul(r)),"exp"))$matmul(s))}
      ops$log<-function(x,y){s<-mf(x,"sqrt");r<-mf(x,"invsqrt");.sym(s$matmul(mf(.sym(r$matmul(y)$matmul(r)),"log"))$matmul(s))}
      ops$sqdist<-function(x,y){r<-mf(x,"invsqrt");z<-mf(.sym(r$matmul(y)$matmul(r)),"log");.batch_dot(M,z,z)}
      ops$transport<-function(x,u,y,v){s<-mf(x,"sqrt");r<-mf(x,"invsqrt");a<-s$matmul(mf(.sym(r$matmul(y)$matmul(r)),"sqrt"))$matmul(r);.sym(a$matmul(v)$matmul(.mt(a)))}
      ops$ehess2rhess<-function(x,u,egrad,ehess) .sym(x$matmul(.sym(ehess))$matmul(x)+
        (u$matmul(.sym(egrad))$matmul(x)+x$matmul(.sym(egrad))$matmul(u))/2)
    } else if(metric=="lerm") {
      ops$inner<-function(x,u,v) .batch_dot(M,df(x,u,"log"),df(x,v,"log"))
      ops$egrad2rgrad<-function(x,u){z<-mf(x,"log");df(z,df(z,.sym(u),"exp"),"exp")}
      ops$retr<-function(x,u) mf(mf(x,"log")+df(x,u,"log"),"exp")
      ops$exp<-ops$retr
      ops$log<-function(x,y) df(mf(x,"log"),mf(y,"log")-mf(x,"log"),"exp")
      ops$sqdist<-function(x,y){z<-mf(x,"log")-mf(y,"log");.batch_dot(M,z,z)}
      ops$transport<-function(x,u,y,v) df(mf(y,"log"),df(x,v,"log"),"exp")
    } else {
      ops$inner<-function(x,u,v) .batch_dot(M,.sylvester(x,u),v)/2
      ops$egrad2rgrad<-function(x,u){g<-.sym(u);2*(x$matmul(g)+g$matmul(x))}
      ops$retr<-function(x,u){a<-.eye(x)+.sylvester(x,u);if(.scalar(torch::linalg_eigvalsh(a)$min())<=0).stop("Wasserstein step leaves the local exponential domain");.sym(a$matmul(x)$matmul(a))}
      ops$exp<-ops$retr
      ops$log<-function(x,y){s<-mf(x,"sqrt");r<-mf(x,"invsqrt");a<-r$matmul(mf(.sym(s$matmul(y)$matmul(s)),"sqrt"))$matmul(r)-.eye(x);.sym(a$matmul(x)+x$matmul(a))}
      ops$sqdist<-function(x,y){s<-mf(x,"sqrt");.batch_trace(x)+.batch_trace(y)-2*.batch_trace(mf(.sym(s$matmul(y)$matmul(s)),"sqrt"))}
      ops$transport<-function(x,u,y,v) .sym(v)
    }
    return(ops)
  }
  list()
}
