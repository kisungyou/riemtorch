test_that("public extension contracts solve a geometry unknown to the driver", {
  skip_if_not(torch::torch_is_installed())
  M<-riem.manifold("positive_line",1,1,"log_pullback",list(
    belongs=function(x,tol)scalar(x)>0,tangent=function(x,u)u,
    inner=function(x,u,v)(u*v/x^2)$sum(),egrad2rgrad=function(x,u)x^2*u,
    retr=function(x,u)x*(u/x)$exp(),transport=function(x,u,y,v)v*y/x,
    project=function(x)x$abs()+.01),list(isometric_transport=TRUE))
  P<-riem.problem(M,function(x)((x$log()-log(3))^2)$sum()/2)
  fit<-riem.optimize(P,tensor(1),"conjugate_gradient")
  expect_equal(scalar(fit$point),3,tolerance=1e-9);expect_true(fit$converged)
})
test_that("schedules validate and have the documented endpoints", {
  expect_equal(riem.schedule(.1)(5),.1)
  expect_equal(riem.schedule(.1,"cosine",total_iterations=10,minimum=.01)(10),.01)
  expect_lt(riem.schedule(.1,"polynomial")(100),.1)
  expect_error(riem.schedule(-1),"initial")
})
test_that("a zero schedule endpoint stops without misreporting convergence", {
  skip_if_not(torch::torch_is_installed())
  P<-riem.problem.finitesum(manifold.euclidean(1),function(x,i)((x-3)^2)$sum()/2,1)
  fit<-riem.optimize(P,tensor(0),"stochastic_gradient",list(
    max_iterations=5,schedule=riem.schedule(.1,"cosine",total_iterations=1)))
  expect_equal(fit$termination,"stopped_small_step");expect_false(fit$converged)
})
