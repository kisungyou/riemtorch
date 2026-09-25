test_that("public diagnostics distinguish correct and incorrect derivatives", {
  M<-manifold.sphere(3);x<-tensor(c(0.6,0.8,0));u<-tensor(c(-0.8,0.6,0))
  P<-riem.problem(M,function(x) -x[1])
  expect_identical(riem.check.manifold(M,x,u)$status,"pass")
  expect_identical(riem.check.gradient(P,x,u)$status,"pass")
  expect_identical(riem.check.hessian(P,x,u)$status,"pass")
  bad<-riem.problem(M,function(x) -x[1],rgrad=function(x) x*0)
  expect_identical(riem.check.gradient(bad,x,u)$status,"fail")
  Q<-riem.problem.leastsquares(manifold.euclidean(2),function(x) x*2)
  expect_identical(riem.check.adjoint(Q,tensor(c(1,2)))$status,"pass")
  Q$adjoint<-function(x,v) v*3
  expect_identical(riem.check.adjoint(Q,tensor(c(1,2)))$status,"fail")
  C<-riem.manifold("custom",2,2,"euclidean",list(
    belongs=function(x,tol) TRUE,tangent=function(x,u) u,inner=function(x,u,v) (u*v)$sum(),
    egrad2rgrad=function(x,u) u,retr=function(x,u) x+u),
    primitive_derivatives=c(retr=1L))
  P<-riem.problem(C,function(x) (x*x)$sum(),rhess=function(x,u) u*2)
  expect_identical(riem.check.hessian(P,tensor(c(1,2)))$status,"inconclusive")
  expect_true(riem.capabilities(C)$first_derivative[riem.capabilities(C)$operation=="retr"])
})

test_that("Lanczos returns accurate Ritz pairs for a small quadratic", {
  A<-torch::torch_diag(tensor(c(1,3,8)))
  P<-riem.problem(manifold.euclidean(3),function(x,data) (x*data$matmul(x))$sum()/2,data=A)
  z<-riem.hessian.spectrum(P,tensor(c(1,1,1)),3)
  expect_equal(z$values,c(1,8),tolerance=1e-8)
  expect_lt(max(z$residuals),1e-8)
})

test_that("streamed finite-sum values gradients and HVPs match an analytic oracle", {
  for(normalization in c("sum","mean")) {
    P<-riem.problem.finitesum(manifold.euclidean(2),function(x,i) ((x-i)^2)$sum()/2,
      23,normalization=normalization,evaluation_batch_size=4)
    x<-tensor(c(1,2));u<-tensor(c(0.2,-0.3));scale<-if(normalization=="sum") 23 else 1
    z<-riem.evaluate(P,x)
    expect_equal(as.numeric(z$gradient),as.numeric((x-12)*scale),tolerance=1e-10)
    expect_equal(as.numeric(riem.hessian(P,x,u)),as.numeric(u*scale),tolerance=1e-10)
    expect_false(riem.evaluate(P,x,FALSE)$value$requires_grad)
    expect_equal(z$evaluations$terms,23)
  }
})

test_that("device probes report requested operations and reject unknown names", {
  z<-riem.devices(operations="basic")
  expect_true(all(z$operations=="basic"))
  expect_true(z$compatible[z$type=="cpu"])
  expect_error(riem.devices(operations="not_an_operation"),"Unknown")
  P<-riem.problem(manifold.sphere(3),function(x) -x[1],required_operations="basic")
  fit<-riem.optimize(P,tensor(c(0,1,0)),device="cpu")
  expect_equal(fit$execution$operations,"basic")
  expect_match(fit$execution$model,"CPU")
})
