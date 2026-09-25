test_that("matrix function derivatives are stable at repeated spectra", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(91)
  U<-tensor(matrix(c(1,.2,.2,-.3),2));G<-tensor(matrix(c(.3,.4,.4,1.2),2))
  for(values in list(c(1,1),c(2,2),c(1,1+1e-10),c(.001,3))) for(type in c("log","sqrt","invsqrt","exp")) {
    X<-torch::torch_diag(tensor(values))$requires_grad_(TRUE)
    out<-riem.matrix.function(X,type)
    grad<-torch::autograd_grad((out*G)$sum(),X)[[1]]
    expect_true(all(is.finite(as.numeric(grad))))
    h<-if(min(values)<.01) 1e-7 else 1e-5
    fd<-((riem.matrix.function(X$detach()+h*U,type)-riem.matrix.function(X$detach()-h*U,type))*G)$sum()/(2*h)
    expect_equal(scalar(fd),scalar((grad*U)$sum()),tolerance=1e-5)
  }
  I<-torch::torch_eye(2,dtype=torch::torch_float64());expect_lt(normf(riem.matrix.frechet(I,U,"log")-U),1e-12)
  X<-I$requires_grad_(TRUE)
  P<-riem.problem(manifold.spd(2,"airm"),function(x)riem.matrix.function(x,"log")$sum())
  expect_error(riem.hessian(P,X,U),"first derivatives only")
})
test_that("squared distances have finite coincident derivatives", {
  skip_if_not(torch::torch_is_installed())
  for(M in list(manifold.sphere(3),manifold.spd(2,"airm"),manifold.spd(2,"lerm"),manifold.spd(2,"wasserstein"),manifold.hyperbolic(2),manifold.hyperbolic(2,"hyperboloid"),manifold.multinomial(3))) {
    target<-riem.random(M);P<-riem.problem(M,function(x) riem.sqdist(M,x,target))
    ev<-riem.evaluate(P,target)
    expect_true(all(vapply(riemtorch:::.leaves(ev$gradient),function(z)all(is.finite(as.numeric(z))),logical(1))))
    expect_lt(scalar(riem.norm(M,target,ev$gradient)),1e-6)
  }
})
test_that("exact Hessians include connection and satisfy self adjointness", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(32)
  for(M in list(manifold.euclidean(3),manifold.sphere(3),manifold.spd(2,"airm"))) {
    x<-riem.random(M);u<-riem.tangent(M,x,torch::torch_randn_like(x));v<-riem.tangent(M,x,torch::torch_randn_like(x))
    P<-riem.problem(M,function(x)(x^2)$sum()/2)
    Hu<-riem.hessian(P,x,u);Hv<-riem.hessian(P,x,v)
    expect_equal(scalar(riem.inner(M,x,Hu,v)),scalar(riem.inner(M,x,u,Hv)),tolerance=1e-7)
    if(M$name=="sphere") expect_lt(normf(Hu),1e-8)
    if(M$name=="euclidean") expect_lt(normf(Hu-u),1e-10)
    if(M$name=="spd") {
      expected<-x$matmul(u)$matmul(x)+(u$matmul(x)$matmul(x)+x$matmul(x)$matmul(u))/2
      expect_lt(normf(Hu-expected),1e-7)
    }
    u<-u/(1+normf(u));h<-1e-4
    fp<-riem.evaluate(P,riem.retr(M,x,u,h),FALSE)$value
    fm<-riem.evaluate(P,riem.retr(M,x,u,-h),FALSE)$value
    f0<-riem.evaluate(P,x,FALSE)$value
    expect_equal(scalar((fp+fm-f0*2)/h^2),scalar(riem.inner(M,x,u,riem.hessian(P,x,u))),tolerance=2e-5)
  }
})
test_that("least squares adjoints use the selected weighted product metric", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.product(list(a=manifold.euclidean(2),b=manifold.euclidean(1)),c(2,5))
  x<-list(a=tensor(c(1,2)),b=tensor(3));u<-list(a=tensor(c(.3,-.2)),b=tensor(.7))
  P<-riem.problem.leastsquares(M,function(z) z$a+z$b,weights=tensor(c(1,2)))
  counts<-riemtorch:::.new_counts();v<-tensor(c(.4,.8))
  Ju<-riemtorch:::.residual_jvp(P,x,u,counts);Jv<-riemtorch:::.residual_adjoint(P,x,v,counts)
  expect_equal(scalar((Ju*v)$sum()),scalar(riem.inner(M,x,u,Jv)),tolerance=1e-9)
  expect_equal(as.numeric(Ju),c(1,.5))
})
test_that("analytic callbacks, unused leaves, and exact counts are handled", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.euclidean(2);x<-tensor(c(1,2));P<-riem.problem(M,function(z)(z^2)$sum()/2,egrad=function(z) z)
  ev<-riem.evaluate(P,x);expect_equal(ev$evaluations$fn,1L);expect_equal(ev$evaluations$gradient,1L)
  expect_equal(as.numeric(ev$gradient),c(1,2))
  expect_error(riem.problem(M,function(x)x$sum(),egrad=identity,rgrad=identity),"not both")
  expect_error(riem.evaluate(riem.problem(M,function(x)3),x),"scalar torch")
  product<-manifold.product(list(a=M,b=M));P<-riem.problem(product,function(z)(z$a^2)$sum())
  expect_equal(as.numeric(riem.evaluate(P,list(a=x,b=x))$gradient$b),c(0,0))
})
