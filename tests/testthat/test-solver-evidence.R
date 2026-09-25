test_that("every geometry decreases a custom scalar objective", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(208)
  cases<-geom_cases()
  for(label in names(cases)) {
    M<-cases[[label]];target<-riem.random(M);x<-riem.random(M)
    P<-riem.problem(M,geom_loss(M,target));f0<-scalar(riem.evaluate(P,x,FALSE)$value)
    fit<-riem.optimize(P,x,control=list(max_iterations=50))
    expect_true(all(riem.belongs(M,fit$point)),info=label)
    expect_lt(fit$objective,f0-1e-7,label=paste(label,fit$message))
    expect_false(fit$termination %in% c("domain_error","numerical_failure"),info=paste(label,fit$message))
  }
})
test_that("first-order methods solve a sphere linear objective", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.sphere(3);a<-tensor(c(1,2,3));P<-riem.problem(M,function(x)-(a*x)$sum());x<-tensor(c(1,0,0))
  for(method in c("steepest_descent","conjugate_gradient","barzilai_borwein","lbfgs")) {
    fit<-riem.optimize(P,x,method,list(max_iterations=300))
    expect_equal(fit$objective,-sqrt(14),tolerance=1e-7,info=method)
    expect_lt(normf(fit$point-a/sqrt(14)),1e-5,label=method)
  }
})
test_that("Procrustes and principal subspace have known solutions", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(183)
  A<-torch::torch_randn(c(4,2),dtype=torch::torch_float64());Q<-riem.project(manifold.stiefel(4,2,"euclidean"),A)
  for(metric in c("euclidean","canonical")) {
    M<-manifold.stiefel(4,2,metric);P<-riem.problem(M,function(x)-(A*x)$sum())
    fit<-riem.optimize(P,riem.random(M),"conjugate_gradient",list(max_iterations=300))
    expect_lt(normf(fit$point-Q),1e-5)
  }
  M<-manifold.grassmann(4,2);D<-torch::torch_diag(tensor(c(5,3,1,0)))
  P<-riem.problem(M,function(x) -(x*D$matmul(x))$sum())
  fit<-riem.optimize(P,riem.random(M),"conjugate_gradient",list(max_iterations=250))
  expect_lt(normf(fit$point$matmul(fit$point$t())-torch::torch_diag(tensor(c(1,1,0,0)))),1e-5)
})
test_that("SPD log-coordinate objectives have known solutions at repeated initial spectra", {
  skip_if_not(torch::torch_is_installed())
  target<-torch::torch_diag(tensor(c(.2,-.3,.6)))
  for(metric in c("airm","lerm")) {
    M<-manifold.spd(3,metric)
    P<-riem.problem(M,function(x) ((riem.matrix.function(x,"log")-target)^2)$sum()/2)
    fit<-riem.optimize(P,torch::torch_eye(3,dtype=torch::torch_float64()),"steepest_descent",list(max_iterations=100))
    expect_lt(fit$objective,1e-10,label=metric)
    expect_lt(normf(fit$point-torch::torch_matrix_exp(target)),1e-5)
  }
})
test_that("second-order methods solve anisotropic quadratics with model diagnostics", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.euclidean(3);w<-tensor(c(1,3,7));P<-riem.problem(M,function(x)(w*(x-1)^2)$sum()/2);x<-tensor(c(3,-2,4))
  for(method in c("trust_regions","newton_cg","adaptive_regularization_cubics")) {
    fit<-riem.optimize(P,x,method,list(max_iterations=100,inner_tolerance=.01))
    expect_lt(fit$objective,1e-10,label=paste(method,fit$message))
    expect_gt(fit$evaluations$hessian,0)
    expect_true(length(fit$diagnostics$subproblems)>0)
  }
  for(method in c("gauss_newton","levenberg_marquardt")) {
    L<-riem.problem.leastsquares(M,function(x) w$sqrt()*(x-1))
    fit<-riem.optimize(L,x,method,list(max_iterations=50,inner_tolerance=.01))
    expect_lt(fit$objective,1e-10,label=method);expect_identical(fit$diagnostics$model,"gauss_newton")
  }
})
test_that("strong Wolfe differentiates the actual retraction curve", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.sphere(3);a<-tensor(c(1,2,3));P<-riem.problem(M,function(x)-(x*a)$sum());x<-tensor(c(1,0,0))
  fit<-riem.optimize(P,x,"conjugate_gradient",list(line_search="strong_wolfe",max_iterations=100))
  expect_equal(fit$objective,-sqrt(14),tolerance=1e-7)
  ctl<-riemtorch:::.controls(list(line_search="strong_wolfe"));c<-riemtorch:::.new_counts();ev<-riem.evaluate(P,x)
  d<--ev$gradient;ans<-riemtorch:::.linesearch(P,x,scalar(ev$value),ev$gradient,d,ctl,c)
  slope<-scalar(riem.inner(M,x,ev$gradient,d))
  expect_true(ans$ok);expect_lte(ans$f,scalar(ev$value)+ctl$armijo*ans$step*slope)
  expect_lte(abs(riemtorch:::.curve_slope(P,x,d,ans$step,c)),-ctl$wolfe*slope)
})
test_that("stochastic normalization and product blocks are explicit", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(8)
  M<-manifold.euclidean(1);x<-tensor(0)
  meanP<-riem.problem.finitesum(M,function(x,i)((x-i)^2)$sum()/2,3)
  sumP<-riem.problem.finitesum(M,function(x,i)((x-i)^2)$sum()/2,3,"sum")
  expect_equal(scalar(riem.evaluate(sumP,x)$value),3*scalar(riem.evaluate(meanP,x)$value))
  expect_equal(as.numeric(riem.evaluate(sumP,x)$gradient),3*as.numeric(riem.evaluate(meanP,x)$gradient))
  fit<-riem.optimize(meanP,x,"stochastic_gradient",list(batch_size=3,step_size=.2,max_iterations=100))
  expect_equal(as.numeric(fit$point),2,tolerance=1e-6);expect_equal(fit$diagnostics$sampling$stationarity,"full_gradient")
  P<-riem.problem(manifold.product(list(a=M,b=M),c(2,3)),function(z)((z$a-1)^2+(z$b+2)^2)$sum()/2)
  fit<-riem.optimize(P,list(a=x,b=x),"alternating_gradient",list(max_iterations=200))
  expect_lt(fit$objective,1e-10)
})
test_that("heuristics improve fixtures without a stationarity certificate", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(1001)
  M<-manifold.euclidean(2);P<-riem.problem(M,function(x)(x^2)$sum())
  for(method in c("particle_swarm","nelder_mead")) {
    fit<-riem.optimize(P,tensor(c(1,2)),method,list(max_iterations=60,population=12))
    expect_lt(fit$objective,.01,label=paste(method,fit$message));expect_true(is.na(fit$gradient_norm));expect_false(fit$converged)
  }
  A<-torch::torch_diag(tensor(c(5,2,1,0)))
  for(method in c("stiefel_annealing","grassmann_macg")) {
    M<-if(method=="grassmann_macg") manifold.grassmann(4,2) else manifold.stiefel(4,2,"euclidean")
    P<-riem.problem(M,function(x)-(x*A$matmul(x))$sum());x<-riem.random(M);f0<-scalar(riem.evaluate(P,x,FALSE)$value)
    fit<-riem.optimize(P,x,method,list(max_iterations=30,population=12,temperature=.1))
    expect_lt(fit$objective,f0,label=paste(method,fit$message));expect_true(all(riem.belongs(M,fit$point)))
  }
})
test_that("failure diagnostics and evaluation counts are honest", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.euclidean(1);x<-tensor(1)
  P<-riem.problem(M,function(x)(x^2)$sum()/2)
  fit<-riem.optimize(P,x)
  expect_equal(fit$evaluations$fn,3L)
  expect_equal(fit$evaluations$gradient,2L)
  expect_equal(fit$accepted_steps,1L)
  bad<-riem.problem(M,function(x) x$sum()*NaN)
  expect_equal(riem.optimize(bad,x)$termination,"nonfinite_objective")
  domain<-riem.optimize(riem.problem(manifold.sphere(2),function(x)x$sum()),tensor(c(2,0)))
  expect_equal(domain$termination,"domain_error")
  small<-riem.optimize(P,x,control=list(step_size=1e-9,step_tolerance=1e-8))
  expect_equal(small$termination,"stopped_small_step");expect_false(small$converged)
  failed<-riem.optimize(P,x,control=list(step_size=100,max_linesearch=1))
  expect_equal(failed$termination,"line_search_failed");expect_equal(failed$rejected_steps,1L)
  expect_error(riem.optimize(P,x,control=list(unknown=1)),"Unknown control")
})
test_that("trust-region CG detects negative curvature on the sphere", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.sphere(3);a<-tensor(c(1,2,3));x<-tensor(c(-1,0,0))
  P<-riem.problem(M,function(x)-(x*a)$sum())
  fit<-riem.optimize(P,x,"trust_regions",list(max_iterations=50))
  expect_equal(fit$objective,-sqrt(14),tolerance=1e-8)
  statuses<-vapply(fit$diagnostics$subproblems,`[[`,character(1),"status")
  expect_true("negative_curvature" %in% statuses)
  ratios<-vapply(fit$diagnostics$subproblems,function(s)s$ratio,numeric(1))
  expect_true(all(is.finite(ratios)))
})
test_that("explicit Hessian and least-squares callbacks work on metric problems", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.euclidean(2);x<-tensor(c(3,-1));w<-tensor(c(2,4))
  P<-riem.problem(M,function(x)(w*x^2)$sum()/2,egrad=function(x)w*x,rhess=function(x,u)w*u)
  fit<-riem.optimize(P,x,"newton_cg",list(inner_tolerance=.01))
  expect_lt(fit$objective,1e-12)
  L<-riem.problem.leastsquares(M,function(x)w*(x-1),
    jvp=function(x,u) w*u,adjoint=function(x,v)w*v)
  fit<-riem.optimize(L,x,"gauss_newton",list(inner_tolerance=.01))
  expect_lt(fit$objective,1e-12)
})
