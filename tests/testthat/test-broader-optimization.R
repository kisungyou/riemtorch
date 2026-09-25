test_that("augmented Lagrangian satisfies analytic equality and inequality KKT systems", {
  skip_if_not(torch::torch_is_installed())
  for(explicit in c(FALSE,TRUE)) {
    P<-riem.problem.constrained(manifold.euclidean(2),function(x) ((x-2)^2)$sum()/2,
      equality=function(x) x$sum()$reshape(1)-1,
      equality_adjoint=if(explicit) function(x,w) torch::torch_ones_like(x)*w else NULL)
    fit<-riem.optimize(P,tensor(c(0,0)),device="cpu",control=list(max_iterations=25,gradient_tolerance=1e-6))
    expect_true(fit$converged,info=fit$message);expect_equal(as.numeric(fit$point),c(.5,.5),tolerance=2e-6)
    expect_lt(fit$kkt$feasibility,1e-6);expect_lt(fit$kkt$stationarity,1e-6)
    expect_equal(as.numeric(fit$multipliers$equality),1.5,tolerance=2e-6)
  }
  P<-riem.problem.constrained(manifold.euclidean(2),function(x) ((x-tensor(c(2,-1)))^2)$sum()/2,
    inequality=function(x) x)
  fit<-riem.optimize(P,tensor(c(-.5,-.5)),device="cpu",control=list(max_iterations=30,gradient_tolerance=1e-6))
  expect_true(fit$converged,info=fit$message);expect_equal(as.numeric(fit$point),c(0,-1),tolerance=2e-6)
  expect_lt(fit$kkt$complementarity,1e-6);expect_equal(as.numeric(fit$multipliers$inequality),c(2,0),tolerance=2e-6)
})
test_that("constraint data and termination budgets are explicit", {
  skip_if_not(torch::torch_is_installed())
  P<-riem.problem.constrained(manifold.sphere(3),function(x,data) -x[2],
    equality=function(x,data) x[1]$reshape(1)-data,data=tensor(.5))
  fit<-riem.optimize(P,tensor(c(1,0,0)),device="cpu",control=list(max_iterations=30,gradient_tolerance=1e-5))
  expect_lt(abs(scalar(fit$point[1])-.5),1e-5);expect_lt(abs(scalar(fit$point[2])-sqrt(.75)),1e-4)
  limited<-riem.optimize(P,tensor(c(1,0,0)),device="cpu",control=list(max_evaluations=3))
  expect_equal(limited$termination,"evaluation_budget");expect_false(limited$converged)
  stopped<-riem.optimize(P,tensor(c(1,0,0)),device="cpu",control=list(callback=function(s) TRUE))
  expect_equal(stopped$termination,"user_stopped")
  bad<-riem.problem.constrained(manifold.euclidean(1),function(x) x$square()$sum(),equality=function(x) 1)
  expect_equal(riem.optimize(bad,tensor(1),device="cpu")$termination,"numerical_failure")
})
test_that("intrinsic proximal gradient matches soft threshold and log coordinates", {
  skip_if_not(torch::torch_is_installed())
  for(positive in c(FALSE,TRUE)) {
    M<-if(positive) manifold.positive(2) else manifold.euclidean(2)
    coordinate<-if(positive) function(x) x$log() else function(x) x
    P<-riem.problem.composite(M,function(x) ((coordinate(x)-tensor(c(2,-.5)))^2)$sum()/2,
      nonsmooth=function(x) coordinate(x)$abs()$sum(),
      prox=function(q,step) {
        z<-coordinate(q);v<-z$sign()*(z$abs()-step)$clamp(min=0)
        if(positive) v$exp() else v
      })
    x<-if(positive) tensor(c(1,1)) else tensor(c(0,0))
    fit<-riem.optimize(P,x,device="cpu")
    expect_true(fit$converged);expect_equal(as.numeric(coordinate(fit$point)),c(1,0),tolerance=1e-6)
    expect_lt(fit$proximal_residual,1e-7);expect_true(riem.belongs(M,fit$point))
  }
})
test_that("intrinsic distance proximal maps work on SPD and hyperbolic geometries", {
  skip_if_not(torch::torch_is_installed())
  for(M in list(manifold.spd(2,"lerm"),manifold.hyperbolic(2))) {
    target<-if(M$name=="spd") torch::torch_eye(2,dtype=torch::torch_float64()) else tensor(c(0,0))
    P<-riem.problem.composite(M,nonsmooth=function(x) riem.dist(M,x,target),
      prox=function(q,step) {
        distance<-scalar(riem.dist(M,q,target))
        if(distance<=step) target$clone() else riem.exp(M,q,riem.log(M,q,target)*(step/distance))
      })
    start<-if(M$name=="spd") target*2 else tensor(c(.2,.1))
    fit<-riem.optimize(P,start,device="cpu")
    expect_true(fit$converged,info=fit$message);expect_lt(scalar(riem.dist(M,fit$point,target)),1e-6)
  }
})
test_that("cyclic proximal point reports a cycle residual without false stationarity", {
  skip_if_not(torch::torch_is_installed())
  P<-riem.problem.composite(manifold.euclidean(1),
    nonsmooth=list(function(x) x$square()$sum()/2,function(x) ((x-2)^2)$sum()/2),
    prox=list(function(q,step) q/(1+step),function(q,step) (q+2*step)/(1+step)))
  fit<-riem.optimize(P,tensor(3),device="cpu",control=list(max_iterations=300))
  expect_equal(scalar(fit$point),1,tolerance=.02);expect_false(fit$converged)
  expect_equal(fit$diagnostics$residual_kind,"cycle_displacement_over_step")
  expect_error(riem.problem.composite(manifold.sphere(3),nonsmooth=function(x) x[1],prox=function(q,step) q),"experimental")
  bad<-riem.problem.composite(manifold.positive(1),nonsmooth=function(x) x$sum(),prox=function(q,step) -q)
  failed<-riem.optimize(bad,tensor(1),device="cpu")
  expect_equal(failed$termination,"numerical_failure");expect_equal(scalar(failed$point),1)
})
test_that("multistart excludes lower-cost constraint-violating results", {
  skip_if_not(torch::torch_is_installed())
  P<-riem.problem.constrained(manifold.euclidean(1),function(x) -x$sum(),inequality=function(x) x-1)
  fit<-riem.multistart(P,initials=list(tensor(2),tensor(0)),control=list(max_iterations=0),device="cpu")
  expect_equal(fit$feasible,c(FALSE,TRUE));expect_equal(fit$best_index,2L)
  expect_equal(fit$best$method,"augmented_lagrangian")
})
