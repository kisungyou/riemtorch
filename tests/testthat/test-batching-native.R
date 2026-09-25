test_that("built-in batch kernels agree with pointwise geometry", {
  torch::torch_manual_seed(71)
  cases<-list(manifold.euclidean(3),manifold.sphere(3),manifold.oblique(3,2),
    manifold.stiefel(4,2,"euclidean"),manifold.grassmann(4,2),
    manifold.grassmann(4,2,"projection"),manifold.spd(2,"airm"),
    manifold.spd(2,"lerm"),manifold.spd(2,"wasserstein"))
  for(M in cases) {
    x<-riem.random(M,batch_shape=3);a<-torch::torch_randn_like(x)
    u<-riem.tangent(M,x,a)
    batched<-riem.inner(M,x,u,u)
    pointwise<-torch::torch_stack(lapply(1:3,function(i)
      riem.inner(M,x$select(1,i),u$select(1,i),u$select(1,i))))
    expect_equal(as_array(batched),as_array(pointwise),tolerance=1e-9,
      info=paste(M$name,M$metric))
    y<-riem.retr(M,x,u,step=1e-3)
    expect_true(all(riem.belongs(M,y,tol=1e-5)),info=paste(M$name,M$metric))
  }
})

test_that("batched spectral matrix functions preserve shape and gradients", {
  x<-torch::torch_eye(3,dtype=torch::torch_float64())$unsqueeze(1)$expand(c(4,3,3))$clone()
  x<-x$requires_grad_(TRUE)
  y<-riem.matrix.function(x,"log")
  expect_identical(as.integer(y$shape),c(4L,3L,3L))
  y$square()$sum()$backward()
  expect_true(all(as_array(torch::torch_isfinite(x$grad))))
  d<-riem.matrix.frechet(x$detach(),torch::torch_ones_like(x),"sqrt")
  expect_identical(as.integer(d$shape),c(4L,3L,3L))
})

test_that("custom manifolds can provide a native batch kernel", {
  calls<-0L
  M<-riem.manifold("batch_line",1,1,"euclidean",list(
    belongs=function(x,tol) TRUE,tangent=function(x,u)u,
    inner=function(x,u,v)(u*v)$sum(),egrad2rgrad=function(x,u)u,
    retr=function(x,u)x+u),batch_operations=list(
      inner=function(x,u,v){calls<<-calls+1L;(u*v)$sum(dim=-1)}))
  x<-torch::torch_zeros(c(5,1),dtype=torch::torch_float64())
  u<-torch::torch_ones_like(x)
  expect_equal(as_array(riem.inner(M,x,u,u)),rep(1,5))
  expect_equal(calls,1L)
  expect_true(riem.capabilities(M)$native_batch[
    riem.capabilities(M)$operation=="inner"])
})
