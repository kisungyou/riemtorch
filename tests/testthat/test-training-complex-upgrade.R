test_that("complex derivatives agree with real and imaginary perturbations", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(512)
  for(M in list(manifold.euclidean(3,"complex"),manifold.sphere(3,"complex"),manifold.complexcircle(3))) {
    x<-riem.random(M,device="cpu");a<-torch::torch_complex(tensor(c(.2,-.4,.3)),tensor(c(.7,.1,-.2)))
    P<-riem.problem(M,function(z) (z-a)$abs()$square()$sum()/2)
    expect_true(riem.belongs(M,x));expect_true(x$is_complex())
    ev<-riem.evaluate(P,x)
    for(u in list(torch::torch_complex(tensor(c(1,0,0)),tensor(c(0,0,0))),
                  torch::torch_complex(tensor(c(0,0,0)),tensor(c(1,0,0))))) {
      u<-riem.tangent(M,x,u);h<-1e-5
      fd<-(P$fn(riem.exp(M,x,u*h))-P$fn(riem.exp(M,x,-u*h)))/(2*h)
      expect_equal(scalar(fd),scalar(riem.inner(M,x,ev$gradient,u)),tolerance=1e-7)
    }
    expect_false(riem.check.gradient(P,x)$status=="fail")
    expect_false(riem.check.hessian(P,x)$status=="fail")
    fit<-riem.optimize(P,x,device="cpu",control=list(max_iterations=30))
    expect_lt(fit$objective,scalar(ev$value))
  }
})
test_that("complex frames and phases preserve membership and local maps", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(51)
  for(M in list(manifold.stiefel(4,2,"euclidean",field="complex"),manifold.grassmann(4,2,field="complex"),manifold.unitary(3))) {
    x<-riem.random(M,device="cpu");u<-riem.tangent(M,x,torch::torch_randn_like(x))*.05
    expect_true(riem.belongs(M,x));expect_true(riem.istangent(M,x,u))
    expect_true(riem.belongs(M,riem.retr(M,x,u)))
    y<-riem.exp(M,x,u);expect_true(riem.belongs(M,y))
    if(M$name=="grassmann.complex") expect_lt(scalar((riem.log(M,x,y)-u)$abs()$max()),1e-8)
    expect_false(M$capabilities$hessian)
  }
  M<-manifold.power(manifold.complexcircle(2),3);x<-riem.random(M,dtype="complex64",device="cpu")
  expect_true(riem.belongs(M,x));expect_true(riem.inner(M,x,x,x)$dtype==torch::torch_float32())
})
test_that("adaptive power states are independent and checkpoints remain readable", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.power(manifold.euclidean(2),2)
  for(make in list(function(...) optim_radam(...,amsgrad=TRUE),optim_radagrad)) {
    x<-torch::nn_parameter(tensor(matrix(c(1,2,3,4),2)));o<-make(list(x),manifold=M)
    o$zero_grad();x$square()$sum()$backward();o$step()
    expect_equal(o$state$get(x)$variance$numel(),2)
    path<-tempfile();riem.save(o,path);y<-torch::nn_parameter(torch::torch_zeros_like(x));r<-make(list(y),manifold=M)
    riem.load(r,path);expect_equal(as.numeric(x),as.numeric(y));unlink(path)
  }
  x<-torch::nn_parameter(tensor(c(1,2)));o<-optim_radam(list(x));x$square()$sum()$backward();o$step()
  old<-o$state_dict();old$schema_version<-1L
  old$groups[[1]]$options[c("amsgrad","backtrack","armijo","max_backtracks")]<-NULL
  expect_silent(o$load_state_dict(old))
})
test_that("sparse gradients coalesce and leave untouched rows and state unchanged", {
  skip_if_not(torch::torch_is_installed())
  for(base in list(manifold.euclidean(3),manifold.sphere(3),manifold.hyperbolic(3))) {
    M<-manifold.power(base,6);embedding<-torch::nn_embedding(6,3,sparse=TRUE)$to(dtype=torch::torch_float64())
    torch::with_no_grad(embedding$weight$copy_(riem.random(M,device="cpu")))
    x<-embedding$weight;before<-x$detach()$clone();opt<-optim_radam(list(x),manifold=M,lr=.005,amsgrad=TRUE)
    opt$zero_grad();embedding(torch::torch_tensor(c(2L,2L,5L),dtype=torch::torch_int64()))$square()$sum()$backward();opt$step()
    expect_true(riem.belongs(M,x));expect_equal(as.numeric(x[c(1,3,4,6),]),as.numeric(before[c(1,3,4,6),]),tolerance=0)
    s<-opt$state$get(x);expect_equal(as.integer(s$row_steps),c(0L,1L,0L,0L,1L,0L))
    expect_equal(as.numeric(s$variance[c(1,3,4,6)]),rep(0,4))
    oldrow<-x[5,]$detach()$clone();oldv<-s$variance[5]$clone()
    opt$zero_grad();embedding(torch::torch_tensor(c(2L),dtype=torch::torch_int64()))$sum()$backward();opt$step()
    expect_equal(as.numeric(x[5,]),as.numeric(oldrow),tolerance=0)
    expect_equal(scalar(opt$state$get(x)$variance[5]),scalar(oldv),tolerance=0)
    path<-tempfile();riem.save(opt,path);expect_silent(riem.load(opt,path));unlink(path)
  }
})
test_that("training line search decreases and rolls back failed closures", {
  skip_if_not(torch::torch_is_installed())
  x<-torch::nn_parameter(tensor(c(1,2)));o<-optim_rlinesearch(list(x),lr=10)
  closure<-function() {o$zero_grad();v<-x$square()$sum()/2;v$backward();v}
  initial<-scalar(closure());o$step(closure);expect_lt(scalar(closure()),initial)
  before<-x$detach()$clone();state<-o$state$get(x)$step;n<-0
  expect_error(o$step(function() {n<<-n+1;if(n>1) stop("closure failed");closure()}),"closure failed")
  expect_equal(as.numeric(x),as.numeric(before),tolerance=0);expect_equal(o$state$get(x)$step,state)
})
test_that("managed training averages accumulation, clips metric norms and resumes epoch hooks", {
  skip_if_not(torch::torch_is_installed())
  net<-torch::nn_module("accum_net",initialize=function() self$w<-torch::nn_parameter(tensor(0)),forward=function(x) self$w*x)
  loader<-torch::dataloader(torch::tensor_dataset(tensor(c(1,2,3,4))),batch_size=1,shuffle=FALSE)
  loss<-function(m,b) (m(b[[1]])-b[[1]])$square()$mean()/2
  factory<-function(m) optim_rsgd(m$parameters,lr=.1)
  state<-0L;schedules<-0L
  hooks<-list(get=function() state,set=function(z) state<<-z)
  scheduler<-function(o,e,v) {state<<-e;schedules<<-schedules+1L}
  fit<-riem.train(net(),loader,loss,factory,epochs=1,device="cpu",accumulate=4,
    validation=function(m,e) c(weight=scalar(m$w)),scheduler=scheduler,data_state=hooks)
  expect_equal(scalar(fit$model$w),.75,tolerance=1e-12);expect_equal(fit$steps,1)
  expect_equal(fit$training_state$data_order,1L);expect_equal(schedules,1L)
  clipped<-riem.train(net(),loader,loss,factory,epochs=1,device="cpu",accumulate=4,clip_norm=1)
  expect_equal(scalar(clipped$model$w),.1,tolerance=1e-12)
  path<-tempfile();riem.save(fit$optimizer,path,fit$model,fit$training_state)
  state<-99L
  resumed<-riem.train(net(),loader,loss,factory,epochs=3,device="cpu",checkpoint=path,
    restore_rng=TRUE,accumulate=4,data_state=hooks,callback=function(s) TRUE)
  expect_equal(state,1L);expect_equal(resumed$history$epoch,2L)
  expect_equal(resumed$termination,"user_stopped");expect_equal(resumed$steps,2L);unlink(path)
})
test_that("legacy scalar power variance checkpoints migrate explicitly", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.power(manifold.euclidean(2),2);x<-torch::nn_parameter(tensor(matrix(1,2,2)))
  o<-optim_radam(list(x),manifold=M);x$square()$sum()$backward();o$step()
  old<-o$state_dict();old$schema_version<-1L;old$state[[1]]$variance<-old$state[[1]]$variance$sum()
  old$groups[[1]]$geometry[[1]]<-list(name=M$name,shape=M$shape,dimension=M$dimension,metric=M$metric,specification=list(copies=2L))
  old$groups[[1]]$options[c("amsgrad","backtrack","armijo","max_backtracks")]<-NULL
  expect_silent(o$load_state_dict(old));expect_equal(o$state$get(x)$variance$numel(),2)
})
test_that("complex frame solvers decrease appropriate invariant objectives", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(630)
  for(M in list(manifold.stiefel(4,2,"euclidean",field="complex"),manifold.grassmann(4,2,field="complex"),manifold.unitary(3))) {
    x<-riem.random(M,device="cpu");target<-riem.random(M,device="cpu")
    loss<-if(M$name=="grassmann.complex") function(z) (z$matmul(z$t()$conj())-target$matmul(target$t()$conj()))$abs()$square()$sum()/2 else
      function(z) (z-target)$abs()$square()$sum()/2
    P<-riem.problem(M,loss);fit<-riem.optimize(P,x,"lbfgs",device="cpu",control=list(max_iterations=25))
    expect_lt(fit$objective,scalar(loss(x)));expect_true(riem.belongs(M,fit$point))
  }
})
