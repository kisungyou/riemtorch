test_that("RSGD reduces to Euclidean momentum and preserves identity", {
  skip_if_not(torch::torch_is_installed())
  x<-torch::nn_parameter(tensor(c(1,2)));y<-torch::nn_parameter(tensor(c(1,2)))
  ours<-optim_rsgd(list(x),lr=.1,momentum=.8);reference<-torch::optim_sgd(list(y),lr=.1,momentum=.8)
  object<-x
  for(i in 1:6) {
    ours$zero_grad();reference$zero_grad();(x^2)$sum()$backward();(y^2)$sum()$backward();ours$step();reference$step()
    expect_equal(as.numeric(x),as.numeric(y),tolerance=1e-12)
  }
  expect_identical(object,x)
})
test_that("mixed matrix and Euclidean training maintains feasible tangent state", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(731)
  for(make in list(optim_rsgd,optim_radam)) {
    M<-manifold.stiefel(4,2,"canonical");S<-manifold.spd(2,"airm")
    x<-torch::nn_parameter(riem.random(M));s<-torch::nn_parameter(torch::torch_eye(2,dtype=torch::torch_float64()));b<-torch::nn_parameter(tensor(2))
    frozen<-torch::nn_parameter(tensor(3));frozen$requires_grad_(FALSE)
    unused<-torch::nn_parameter(tensor(5));ref<-list(x=x,s=s,b=b)
    opt<-make(list(list(params=list(x),manifold=M,keys="frame"),list(params=list(s),manifold=S,keys="spd"),list(params=list(b,frozen,unused),keys=c("bias","frozen","unused"))),lr=.01)
    A<-riem.random(M)
    for(i in 1:10) {
      opt$zero_grad()
      loss<-((x-A)^2)$sum()+((s-torch::torch_eye(2,dtype=s$dtype)*2)^2)$sum()+b^2
      loss$backward();opt$step()
      expect_true(riem.belongs(M,x));expect_true(riem.belongs(S,s))
      expect_true(riem.istangent(M,x,opt$state$get(x)$momentum))
      expect_true(riem.istangent(S,s,opt$state$get(s)$momentum))
    }
    expect_identical(x,ref$x);expect_identical(s,ref$s);expect_identical(b,ref$b)
    expect_equal(scalar(frozen),3);expect_equal(scalar(unused),5)
    expect_null(opt$state$get(unused))
  }
})
test_that("training failures are atomic across groups", {
  skip_if_not(torch::torch_is_installed())
  a<-torch::nn_parameter(tensor(1));x<-torch::nn_parameter(tensor(c(0,0)))
  opt<-optim_rsgd(list(list(params=list(a)),list(params=list(x),manifold=manifold.hyperbolic(2))),lr=100)
  (a^2+x$sum())$backward()
  expect_error(opt$step(),"leaves")
  expect_equal(scalar(a),1);expect_equal(as.numeric(x),c(0,0));expect_null(opt$state$get(a))
  expect_error(optim_rsgd(list(list(params=list(a),weight_decay=.1))),"regularizer")
})
test_that("checkpoint continuation agrees with uninterrupted runs", {
  skip_if_not(torch::torch_is_installed())
  for(make in list(optim_rsgd,optim_radam)) {
    M<-manifold.sphere(3);x<-torch::nn_parameter(tensor(c(1,0,0)))
    opt<-make(list(list(params=list(x),manifold=M,keys="direction")),lr=.05)
    step<-function(o,p) {o$zero_grad();(-p[2]+.2*p[3]^2)$backward();o$step()}
    for(i in 1:4) step(opt,x)
    path<-tempfile(fileext=".pt");riem.save(opt,path)
    for(i in 1:5) step(opt,x)
    expected<-x$detach()$clone()
    y<-torch::nn_parameter(tensor(c(1,0,0)));resumed<-make(list(list(params=list(y),manifold=M,keys="direction")),lr=.5)
    object<-y;riem.load(resumed,path)
    for(i in 1:5) step(resumed,y)
    expect_lt(normf(y-expected),1e-12);expect_identical(y,object)
    expect_equal(resumed$state$get(y)$step,opt$state$get(x)$step)
    bad<-torch::nn_parameter(tensor(c(1,0,0)));wrong<-make(list(list(params=list(bad),manifold=M,keys="other")))
    expect_error(riem.load(wrong,path),"mapping")
    expect_equal(as.numeric(bad),c(1,0,0))
    unlink(path)
  }
})
test_that("state dictionaries reject geometry and corrupt momentum", {
  skip_if_not(torch::torch_is_installed())
  M<-manifold.sphere(3);x<-torch::nn_parameter(tensor(c(1,0,0)));opt<-optim_rsgd(list(x),manifold=M)
  (-x[2])$backward();opt$step();state<-opt$state_dict()
  state$state[[1]]$momentum<-x$detach()$clone()
  before<-x$detach()$clone();expect_error(opt$load_state_dict(state),"not tangent")
  expect_lt(normf(x-before),1e-14)
})
test_that("factorwise adaptive state is scalar and basis invariant", {
  skip_if_not(torch::torch_is_installed());torch::torch_manual_seed(119)
  M<-manifold.grassmann(4,2);Q<-riem.random(manifold.rotation(2));start<-riem.random(M)
  x<-torch::nn_parameter(start$clone());y<-torch::nn_parameter(start$matmul(Q))
  a<-optim_radam(list(x),manifold=M,lr=.01);b<-optim_radam(list(y),manifold=M,lr=.01)
  D<-torch::torch_diag(tensor(c(4,3,1,0)))
  for(i in 1:5) {
    a$zero_grad();b$zero_grad();(-(x*D$matmul(x))$sum())$backward();(-(y*D$matmul(y))$sum())$backward();a$step();b$step()
  }
  expect_lt(normf(y-x$matmul(Q)),1e-9);expect_equal(a$state$get(x)$variance$numel(),1)
})
test_that("duplicate parameter ownership is rejected", {
  skip_if_not(torch::torch_is_installed())
  x<-torch::nn_parameter(tensor(c(1,2)))
  expect_error(optim_rsgd(list(list(params=list(x)),list(params=list(x)))),"multiple groups")
})
