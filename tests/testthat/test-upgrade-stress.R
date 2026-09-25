test_that("expanded geometry contracts remain usable at float32 and on noncontiguous points", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(932)
  cases<-list(manifold.positive(c(2,3)),manifold.doublystochastic(3),manifold.orthogonal(3),
    manifold.scaled(manifold.sphere(4),2),manifold.power(manifold.sphere(4),3),
    manifold.stiefel(4,2,"canonical"),manifold.grassmann(4,2),manifold.fixedrank(5,4,2,"svd"),
    manifold.euclidean(3,"complex"),manifold.sphere(3,"complex"),manifold.complexcircle(3),manifold.unitary(3))
  for(M in cases) {
    dtype<-if(identical(M$specification$field,"complex")) "complex64" else "float32"
    x<-riem.random(M,device="cpu",dtype=dtype)
    expect_true(riem.belongs(M,x))
    u<-riem.tangent(M,x,riemtorch:::.tree_map(x,torch::torch_randn_like))
    expect_true(riem.istangent(M,x,u,tol=1e-4))
    y<-riem.retr(M,x,u,step=.001)
    expect_true(riem.belongs(M,y,tol=1e-4))
    expect_true(is.finite(scalar(riem.inner(M,x,u,u))))
    if(inherits(x,"torch_tensor")&&length(x$shape)==2&&x$shape[1]==x$shape[2]) {
      z<-x$t();expect_false(z$is_contiguous());expect_true(riem.belongs(M,z,tol=1e-4))
    }
  }
})
test_that("compact diagnostics and repeated-core invariance are qualified", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(81)
  M<-manifold.fixedrank(6,5,2,"svd");x<-riem.random(M,device="cpu")
  expect_equal(riem.check.manifold(M,x)$status,"pass")
  # At repeated singular values, a full core and horizontal factors avoid a
  # differentiated diagonal-SVD chart.
  P<-riem.problem(M,function(z) riem.materialize(M,z)$square()$sum()/2)
  expect_false(riem.check.gradient(P,x)$status=="fail")
  x$S<-torch::torch_diag(tensor(c(1,1e-8)))
  expect_false(riem.belongs(M,x));expect_error(riem.evaluate(P,x),"outside")
  x$S<-torch::torch_diag(tensor(c(1,1e-5)))
  expect_true(riem.belongs(M,x));expect_true(is.finite(scalar(riem.evaluate(P,x)$value)))
})
test_that("float32 solver variants reach meaningful accuracy", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.euclidean(3);x<-tensor(c(1,2,3))$to(dtype=torch::torch_float32())
  P<-riem.problem(M,function(z) z$square()$sum()/2)
  for(method in c("steepest_descent","conjugate_gradient","barzilai_borwein","lbfgs","trust_regions","newton_cg")) {
    fit<-riem.optimize(P,x,method,device="cpu",control=list(gradient_tolerance=1e-5,max_iterations=30))
    expect_lt(fit$objective,1e-7);expect_true(fit$point$dtype==torch::torch_float32())
    expect_false(fit$termination=="numerical_failure")
  }
})
