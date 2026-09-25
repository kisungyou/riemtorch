test_that("solver variants callbacks budgets and fused evaluations work", {
  P<-riem.problem(manifold.euclidean(2),function(x) ((x-1)^2)$sum()/2)
  x<-tensor(c(4,-2))
  for(beta in c("PR+","FR","HS+","DY"))
    expect_lt(riem.optimize(P,x,"conjugate_gradient",list(cg_beta=beta),device="cpu")$objective,1e-10)
  for(bb in c("BB1","BB2","alternating"))
    expect_lt(riem.optimize(P,x,"barzilai_borwein",list(bb_variant=bb),device="cpu")$objective,1e-10)
  f<-riem.optimize(P,x,control=list(callback=function(state) TRUE),device="cpu")
  expect_identical(f$termination,"user_stopped")
  f<-riem.optimize(P,x,control=list(max_evaluations=2),device="cpu")
  expect_identical(f$termination,"evaluation_budget")
  calls<-0L
  P<-riem.problem(manifold.euclidean(2),function(x) ((x-1)^2)$sum()/2,
    value_rgrad=function(x) {calls<<-calls+1L;list(value=((x-1)^2)$sum()/2,gradient=x-1)})
  f<-riem.optimize(P,x,device="cpu")
  expect_lt(f$objective,1e-10);expect_equal(calls,2)
  z<-riem.multistart(P,initials=list(x,tensor(c(1,1))),device="cpu")
  expect_length(z$fits,2);expect_lt(z$best$objective,1e-10)
})

test_that("preconditioned CG solves an ill-conditioned diagonal system", {
  d<-tensor(c(1,100,10000));M<-manifold.euclidean(3)
  P<-riem.problem(M,function(x,data) (data*x*x)$sum()/2,data=d,
    preconditioner=function(x,u,data) u/data)
  f<-riem.optimize(P,tensor(c(1,1,1)),"newton_cg",device="cpu")
  expect_lt(f$objective,1e-12)
  expect_equal(f$diagnostics$subproblems[[1]]$iterations,1)
  P$preconditioner<-function(x,u,data) -u
  f<-riem.optimize(P,tensor(c(1,1,1)),"newton_cg",device="cpu")
  expect_match(f$message,"positive definite")
})

test_that("variance reduction preserves normalization and converges", {
  P<-riem.problem.finitesum(manifold.euclidean(1),function(x,i) ((x-i)^2)$sum()/2,5)
  for(method in c("svrg","srg")) {
    torch::torch_manual_seed(10)
    f<-riem.optimize(P,tensor(0),method,list(step_size=0.2,max_iterations=120),device="cpu")
    expect_equal(as.numeric(f$point),3,tolerance=1e-6)
    expect_lt(f$gradient_norm,1e-6)
    expect_lt(f$evaluations$terms,120*5)
  }
})

test_that("robust least squares has matching gradients and PSD models", {
  for(loss in c("linear","huber","soft_l1","cauchy")) {
    P<-riem.problem.leastsquares(manifold.euclidean(1),function(x) torch::torch_cat(list(x-1,x-2,x-100)),loss=loss)
    expect_identical(riem.check.gradient(P,tensor(4),u=tensor(1))$status,"pass")
    expect_gt(scalar(riemtorch:::.gn(P,tensor(4),tensor(1),riemtorch:::.new_counts())),0)
    f<-riem.optimize(P,tensor(0),"levenberg_marquardt",device="cpu")
    expect_true(is.finite(f$objective))
    if(loss=="huber") expect_equal(as.numeric(f$point),2,tolerance=1e-4)
  }
  expect_error(riem.problem.leastsquares(manifold.euclidean(2),function(x)x,
    weights=torch::torch_eye(2),loss="huber"),"diagonal")
})

test_that("new real geometries satisfy metric and retraction contracts", {
  B<-torch::torch_eye(4,dtype=torch::torch_float64())[,1:2]
  cases<-list(manifold.power(manifold.sphere(3),4),manifold.scaled(manifold.sphere(3),2),
    manifold.positive(3),manifold.affine(B),manifold.orthogonal(3),manifold.doublystochastic(3))
  for(M in cases) {
    x<-riem.random(M,device="cpu")
    expect_true(all(riem.belongs(M,x)))
    expect_identical(riem.check.manifold(M,x)$status,"pass",info=M$name)
    P<-riem.problem(M,function(x) (x*x)$sum()/2)
    f<-riem.optimize(P,x,control=list(max_iterations=3),device="cpu")
    expect_true(all(riem.belongs(M,f$point)))
  }
  M<-manifold.power(manifold.sphere(3),4)
  expect_equal(riem.random(M,batch_shape=2,device="cpu")$shape,c(2,4,3))
})

test_that("frame exponentials and canonical Hessians agree with independent differences", {
  for(M in list(manifold.stiefel(5,2,"canonical"),manifold.stiefel(5,2,"euclidean"),
    manifold.grassmann(5,2),manifold.grassmann(5,2,"projection"))) {
    x<-riem.random(M,device="cpu");u<-riemtorch:::.search_noise(M,x,0.1)
    y<-riem.exp(M,x,u)
    expect_true(all(riem.belongs(M,y)))
    expect_lt(normf((riem.exp(M,x,u,1e-5)-x)/1e-5-u),1e-4)
    if(M$name=="grassmann") {
      v<-riem.log(M,x,y);z<-riem.exp(M,x,v)
      if(M$metric=="canonical") expect_lt(normf(z$matmul(z$t())-y$matmul(y$t())),1e-7)
      else expect_lt(normf(z-y),1e-7)
    }
    target<-torch::torch_randn_like(x)
    P<-riem.problem(M,function(x,data) ((x-data)^2)$sum()/2,data=target)
    if(M$name=="stiefel") expect_identical(riem.check.hessian(P,x)$status,"pass",info=M$metric)
  }
})

test_that("compact fixed rank matches dense geometry without singular-value gaps", {
  M<-manifold.fixedrank(8,6,2,"svd");D<-manifold.fixedrank(8,6,2)
  x<-riem.random(M,device="cpu");A<-riem.materialize(M,x)
  expect_true(riem.belongs(M,x));expect_true(riem.belongs(D,A))
  expect_equal(as.numeric(riem.materialize(M,x,c(1,4),c(2,5))),c(scalar(A[1,2]),scalar(A[4,5])))
  target<-torch::torch_randn_like(A)
  P<-riem.problem(M,function(x,data) ((riem.materialize(M,x)-data)^2)$sum()/2,data=target)
  expect_identical(riem.check.gradient(P,x)$status,"pass")
  g<-riem.evaluate(P,x)$gradient
  ambient<-riemtorch:::.svd_ambient_apply(x,g,torch::torch_eye(6,dtype=torch::torch_float64()))
  expect_lt(normf(ambient-riem.tangent(D,A,A-target)),1e-7)
  u<-riemtorch:::.search_noise(M,x,0.01);y<-riem.retr(M,x,u)
  expect_true(riem.belongs(M,y))
  moved<-riem.transport(M,x,u,y,g)
  expect_true(riem.istangent(M,y,moved))
  fit<-riem.optimize(P,x,"lbfgs",list(max_iterations=30),device="cpu")
  expect_lt(fit$objective,scalar(P$fn(x,target)))
  Q<-torch::linalg_qr(torch::torch_randn(c(2,2),dtype=torch::torch_float64()))[[1]]
  rotated<-list(U=x$U$matmul(Q),S=Q$t()$matmul(x$S)$matmul(Q),V=x$V$matmul(Q))
  expect_lt(normf(riem.materialize(M,rotated)-A),1e-10)
})
test_that("structured constructors support fused gradients and preconditioners", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.euclidean(1);x<-tensor(0)
  P<-riem.problem.finitesum(M,function(x,i) ((x-i)^2)$sum()/2,n=5,evaluation_batch_size=2,
    value_rgrad=function(x,indices) {
      delta<-x-tensor(indices)
      list(value=delta$square()$mean()/2,gradient=(x-mean(indices)))
    },preconditioner=function(x,u) u)
  ev<-riem.evaluate(P,x);expect_equal(scalar(ev$gradient),-3,tolerance=1e-12)
  expect_equal(scalar(ev$value),5.5,tolerance=1e-12);expect_equal(ev$evaluations$terms,5L)
  L<-riem.problem.leastsquares(M,function(x) x-2,
    value_rgrad=function(x) list(value=(x-2)$square()$sum()/2,gradient=x-2))
  expect_equal(as.numeric(riem.optimize(L,x,"levenberg_marquardt",device="cpu")$point),2,tolerance=1e-5)
  custom<-M;custom$capabilities$snapshot_transport<-FALSE
  V<-riem.problem.finitesum(custom,function(x,i) x$square()$sum(),n=2)
  expect_error(riem.optimize(V,tensor(1),"svrg",device="cpu"),"snapshot transport")
})
