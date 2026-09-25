test_that("registered data reaches scalar and structured callbacks", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.euclidean(1);x<-tensor(0);target<-tensor(2)
  P<-riem.problem(M,function(x,data)((x-data)^2)$sum()/2,
    egrad=function(x,data)x-data,data=target)
  ev<-riem.evaluate(P,x)
  expect_equal(scalar(ev$value),2);expect_equal(as.numeric(ev$gradient),-2)
  L<-riem.problem.leastsquares(M,function(x,data)x-data,data=target)
  expect_equal(scalar(riem.evaluate(L,x)$value),2)
})

test_that("vectorized finite sums are chunked and preserve normalization", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.euclidean(1);x<-tensor(0);data<-tensor(1:5)
  calls<-new.env(parent=emptyenv());calls$n<-0L;calls$sizes<-integer()
  batch<-function(x,indices,data) {
    calls$n<-calls$n+1L;calls$sizes<-c(calls$sizes,length(indices))
    (x-data[indices])$square()$reshape(c(length(indices))) / 2
  }
  P<-riem.problem.finitesum(M,function(...) stop("term callback should not run"),
    n=5,batch_fn=batch,data=data,evaluation_batch_size=2)
  ev<-riem.evaluate(P,x)
  expect_equal(calls$n,3L);expect_equal(calls$sizes,c(2L,2L,1L))
  expect_equal(scalar(ev$value),mean((1:5)^2)/2,tolerance=1e-12)
  expect_equal(as.numeric(ev$gradient),-3,tolerance=1e-12)
  Ps<-riem.problem.finitesum(M,function(...) stop("unused"),n=5,
    normalization="sum",batch_fn=batch,data=data,evaluation_batch_size=2)
  expect_equal(scalar(riem.evaluate(Ps,x,FALSE)$value),sum((1:5)^2)/2,
    tolerance=1e-12)
})

test_that("stochastic optimization avoids unintended full evaluations", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(18)
  M<-manifold.euclidean(1);x<-tensor(0);data<-tensor(1:5)
  P<-riem.problem.finitesum(M,function(x,i,data)((x-data[i])^2)$sum()/2,
    n=5,batch_fn=function(x,indices,data)
      (x-data[indices])$square()$reshape(c(length(indices)))/2,
    data=data,evaluation_batch_size=2)
  fit<-riem.optimize(P,x,"stochastic_gradient",list(max_iterations=4,
    batch_size=2,step_size=.01,gradient_tolerance=0,
    full_evaluation_every=0))
  expect_equal(fit$evaluations$fn,10L)
  expect_equal(fit$evaluations$terms,18L)
  expect_equal(fit$diagnostics$sampling$full_evaluation_every,0)
  expect_true(all(vapply(fit$history[-1],function(z)
    identical(z$evaluation,"minibatch"),logical(1))))
  expect_true(is.finite(fit$objective));expect_true(is.finite(fit$gradient_norm))
})

test_that("managed training places batches and returns execution metadata", {
  skip_if_not(torch::torch_is_installed())
  # Own the storage before handing R-created matrices to the torch 0.17 loader.
  x<-tensor(matrix(c(0,1,1,2),ncol=1))$clone();y<-(x*2)$clone()
  loader<-torch::dataloader(torch::tensor_dataset(x,y),batch_size=2,shuffle=FALSE)
  type<-torch::nn_module("managed_test_module",
    initialize=function() self$weight<-torch::nn_parameter(
      torch::torch_zeros(1,1,dtype=torch::torch_float64())),
    forward=function(z) z$matmul(self$weight))
  fit<-riem.train(type(),loader,
    loss=function(model,batch)(model(batch[[1]])-batch[[2]])$square()$mean(),
    optimizer_factory=function(model) optim_rsgd(model$parameters,lr=.1),
    epochs=3,device="cpu")
  expect_s3_class(fit,"riem_training_fit")
  expect_equal(nrow(fit$history),3);expect_equal(fit$steps,6L)
  expect_equal(fit$execution$device,"cpu")
  expect_lt(fit$history$loss[3],fit$history$loss[1])
})

test_that("managed training moves tensor loss arguments", {
  type<-torch::nn_module("managed_dot_module",
    initialize=function() self$weight<-torch::nn_parameter(
      torch::torch_zeros(1,1,dtype=torch::torch_float64())),
    forward=function(z) z$matmul(self$weight))
  x<-tensor(matrix(c(0,1),ncol=1))$clone()
  loader<-torch::dataloader(torch::tensor_dataset(x),batch_size=2,shuffle=FALSE)
  target<-tensor(matrix(c(0,2),ncol=1))
  fit<-riem.train(type(),loader,
    loss=function(model,batch,target)(model(batch[[1]])-target)$square()$mean(),
    optimizer_factory=function(model) optim_rsgd(model$parameters,lr=.1),
    epochs=1,device="cpu",dtype="float32",target=target)
  expect_equal(fit$execution$dtype,"float32")
  expect_true(fit$model$weight$dtype==torch::torch_float32())
})

test_that("managed training preserves existing module precision by default", {
  skip_if_not(torch::torch_is_installed())
  type<-torch::nn_module("managed_mixed_precision_module",
    initialize=function() {
      self$weight<-torch::nn_parameter(
        torch::torch_zeros(1,1,dtype=torch::torch_float64()))
      self$bias<-torch::nn_parameter(
        torch::torch_zeros(1,dtype=torch::torch_float32()))
    },forward=function(x) x$matmul(self$weight)+self$bias)
  x<-torch::torch_ones(c(2,1),dtype=torch::torch_float64())
  loader<-torch::dataloader(torch::tensor_dataset(x,x),batch_size=2)
  fit<-riem.train(type(),loader,
    loss=function(model,batch)(model(batch[[1]])-batch[[2]])$square()$mean(),
    optimizer_factory=function(model) optim_rsgd(model$parameters,lr=.1),
    device="cpu")
  expect_true(fit$model$weight$dtype==torch::torch_float64())
  expect_true(fit$model$bias$dtype==torch::torch_float32())
})

test_that("training optimizers move built-in geometry state to parameters", {
  skip_if_not(torch::torch_is_installed())
  source_geometry<-manifold.stiefel.generalized(3,2,
    torch::torch_eye(3,dtype=torch::torch_float32()))
  target_geometry<-manifold.stiefel.generalized(3,2,
    torch::torch_eye(3,dtype=torch::torch_float64()))
  x<-torch::nn_parameter(riem.random(target_geometry,device="cpu"))
  opt<-optim_rsgd(list(x),lr=.01,manifold=source_geometry)
  moved<-opt$param_groups[[1]]$manifold$specification$B
  expect_true(moved$dtype==x$dtype)
  expect_equal(moved$device$type,x$device$type)
})

test_that("version two checkpoints restore module and progress state", {
  skip_if_not(torch::torch_is_installed())
  type<-torch::nn_module("checkpoint_test_module",
    initialize=function() {
      self$weight<-torch::nn_parameter(tensor(1))
      self$offset<-torch::nn_buffer(tensor(3))
    },forward=function(x)x*self$weight+self$offset)
  model<-type();opt<-optim_rsgd(model$parameters,lr=.1)
  path<-tempfile(fileext=".pt");set.seed(71);torch::torch_manual_seed(72)
  riem.save(opt,path,model=model,
    training_state=list(epoch=4L,step=12L,data_order="fixed"),include_rng=TRUE)
  expected_r<-runif(1);expected_t<-torch::torch_randn(1,dtype=torch::torch_float64())
  torch::with_no_grad({model$weight$fill_(9);model$offset$fill_(8)})
  riem.load(opt,path,model=model,restore_rng=TRUE)
  expect_equal(scalar(model$weight),1);expect_equal(scalar(model$offset),3)
  expect_equal(opt$checkpoint$training_state$epoch,4L)
  expect_equal(runif(1),expected_r)
  expect_equal(as.numeric(torch::torch_randn(1,dtype=torch::torch_float64())),
    as.numeric(expected_t),tolerance=0)
  unlink(path)
})

test_that("version one parameter checkpoints remain loadable", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.sphere(2);x<-torch::nn_parameter(tensor(c(1,0)))
  opt<-optim_rsgd(list(list(params=list(x),manifold=M,keys="direction")),lr=.1)
  (-x[2])$backward();opt$step()
  expected<-x$detach()$clone();dict<-opt$state_dict()
  points<-list(direction=expected$clone())
  path<-tempfile(fileext=".pt")
  torch::torch_save(list(schema_version=1L,parameters=points,optimizer=dict),path)
  torch::with_no_grad(x$copy_(tensor(c(1,0))))
  riem.load(opt,path,device="cpu")
  expect_equal(as.numeric(x),as.numeric(expected),tolerance=0)
  expect_equal(opt$checkpoint$schema_version,1L)
  expect_null(opt$checkpoint$training_state)
  unlink(path)
})
