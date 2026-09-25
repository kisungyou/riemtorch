#' Define a Scalar Manifold Optimization Problem
#' @param manifold Geometry specification.
#' @param fn Function of one point returning one finite scalar torch tensor.
#' @param egrad Optional ambient gradient callback `egrad(x)`.
#' @param rgrad Optional metric gradient callback `rgrad(x)`, mutually exclusive
#'   with `egrad`.
#' @param rhess Optional exact Riemannian Hessian callback `rhess(x, u)`.
#' @param label Optional problem label.
#' @param value_rgrad Optional callback returning `list(value, gradient)` with
#'   a scalar value and a Riemannian gradient, evaluated together.
#' @param preconditioner Optional callback `preconditioner(x, u)` applying a
#'   positive definite self-adjoint tangent-space preconditioner.
#' @param required_operations Device operations required by callbacks; NULL
#'   conservatively probes all. See [riem.devices()].
#' @param data Optional registered data. When supplied, callbacks receive it as
#'   an additional named `data` argument. Registered tensor trees can be moved
#'   with the problem by the execution driver.
#' @return A `riem_problem` holding callbacks and geometry; construction does not
#'   evaluate callbacks or initialize devices.
#' @details Deterministic objectives must remain fixed during a solve. Freeze
#' batches, dropout and other random state before line searches. The ordinary
#' solver result is detached; differentiation through an entire solve is not
#' supported. Analytic derivatives must match the objective and selected metric.
#' @examples
#' if (torch::torch_is_installed()) {
#'   M <- manifold.sphere(3)
#'   a <- torch::torch_tensor(c(1, 2, 3), dtype = torch::torch_float64())
#'   problem <- riem.problem(M,
#'     function(x, data) -torch::torch_sum(x * data), data = a)
#'   fit <- riem.optimize(problem, riem.random(M))
#'   fit$objective
#' }
#' @export
riem.problem <- function(manifold,fn,egrad=NULL,rgrad=NULL,rhess=NULL,label=NULL,
                         data=NULL,required_operations=NULL,value_rgrad=NULL,preconditioner=NULL) {
  .check_manifold(manifold)
  if(!is.function(fn)) .stop("fn must be a function")
  if(!is.null(egrad)&&!is.null(rgrad)) .stop("Supply egrad or rgrad, not both")
  if(!is.null(value_rgrad)&&(!is.null(egrad)||!is.null(rgrad))) .stop("value_rgrad cannot be combined with separate gradients")
  for(cb in list(egrad,rgrad,rhess,value_rgrad,preconditioner)) if(!is.null(cb)&&!is.function(cb)) .stop("Derivative callbacks must be functions")
  structure(list(manifold=manifold,fn=fn,egrad=egrad,rgrad=rgrad,rhess=rhess,
    label=label,data=data,value_rgrad=value_rgrad,preconditioner=preconditioner,required_operations=.probe_operations(required_operations)),class="riem_problem")
}
.problem_call<-function(P,callback,...) {
  args<-list(...)
  if(is.null(P$data)) do.call(callback,args) else
    do.call(callback,c(args,list(data=P$data)))
}
.check_point<-function(M,x) {
  if(inherits(M,"riem_product")) {
    .product_check(M,x);Map(.check_point,M$factors,x)
  } else .check_tensor(M,x,batch=FALSE)
  leaves<-.leaves(x);ref<-leaves[[1]]
  if(!all(vapply(leaves,function(z) z$dtype==ref$dtype && identical(z$device$type,ref$device$type) && identical(z$device$index,ref$device$index),logical(1)))) .stop("All product leaves must share dtype and device")
  invisible(x)
}
.check_tangent<-function(M,x,u,label="derivative") {
  .check_point(M,u)
  if(!.allfinite(u)) .stop(label," contains nonfinite values")
  if(!all(riem.istangent(M,x,u))) .stop(label," is not tangent in the declared representation")
  invisible(u)
}
.loss_check<-function(value,x) {
  if(!inherits(value,"torch_tensor")||value$numel()!=1) .stop("Objective must return a scalar torch tensor")
  ref<-.leaves(x)[[1]]
  if(!(value$dtype==.real_dtype(ref$dtype))||!identical(value$device$type,ref$device$type)||!identical(value$device$index,ref$device$index)) .stop("Objective must preserve point dtype and device")
  value$reshape(integer())
}
.new_counts<-function(control=NULL) {
  e<-new.env(parent=emptyenv())
  for(k in c("fn","gradient","hessian","residual","terms","jvp","adjoint")) e[[k]]<-0L
  if(!is.null(control)) attr(e,"runtime")<-list(start=proc.time()[[3]],control=control)
  e
}
.count<-function(counts,key,n=1L) {
  .budget_check(counts)
  counts[[key]]<-counts[[key]]+n
}
.losses_check<-function(value,x,n) {
  if(!inherits(value,"torch_tensor")||length(value$shape)!=1||value$numel()!=n)
    .stop("batch_fn must return one loss per requested index in a vector tensor")
  ref<-.leaves(x)[[1]]
  if(!(value$dtype==.real_dtype(ref$dtype))||!identical(value$device$type,ref$device$type)||
     !identical(value$device$index,ref$device$index))
    .stop("Batch losses must preserve point dtype and device")
  value
}
.finitesum_value<-function(P,x,counts,indices) {
  .count(counts,"terms",length(indices))
  if(!is.null(P$batch_fn)) {
    losses<-.losses_check(.problem_call(P,P$batch_fn,x,as.integer(indices)),x,
                          length(indices))
    z<-losses$mean()
  } else {
    z<-Reduce(`+`,lapply(indices,function(i)
      .loss_check(.problem_call(P,P$term,x,i),x)))/length(indices)
  }
  if(P$normalization=="sum") z<-z*P$n
  z
}
.value<-function(P,x,counts,indices=NULL) {
  .count(counts,"fn")
  if(inherits(P,"riem_finitesum")) {
    if(is.null(indices)&&P$n>P$evaluation_batch_size) {
      # Differentiable values retain their graph by definition. Diagnostic
      # callers use no_grad; derivative callers stream gradients/HVPs below.
      z <- NULL
      start<-1L
      while(start<=P$n) {
        ids <- seq.int(start, min(P$n, start+P$evaluation_batch_size-1L))
        part <- .finitesum_value(P,x,counts,ids)*(length(ids)/P$n)
        z <- if(is.null(z)) part else z+part
        start<-tail(ids,1L)+1L
      }
    } else {
      if(is.null(indices)) indices<-seq_len(P$n)
      z<-.finitesum_value(P,x,counts,indices)
    }
  } else if(inherits(P,"riem_leastsquares")) {
    r<-.residual(P,x,counts);z<-.leastsquares_value(P,r)
  } else if(!is.null(P$value_rgrad)) z<-.problem_call(P,P$value_rgrad,x)$value else z<-.problem_call(P,P$fn,x)
  .loss_check(z,x)
}
.autograd_tree<-function(value,x,create_graph=FALSE) {
  leaves<-.leaves(x)
  if(!value$requires_grad) return(.zeros(x))
  grads<-torch::autograd_grad(value,leaves,create_graph=create_graph,allow_unused=TRUE)
  for(i in seq_along(grads)) if(is.null(grads[[i]])||torch::is_undefined_tensor(grads[[i]])) grads[[i]]<-torch::torch_zeros_like(leaves[[i]])
  .unflatten(x,grads)
}
.evaluate<-function(P,x,counts,gradient=TRUE,indices=NULL,chunk_size=NULL) {
  if(inherits(P,"riem_finitesum")&&is.null(indices)&&!is.null(chunk_size)&&
     P$n>chunk_size) {
    value <- g <- NULL; valid <- TRUE
    start<-1L
    while(start<=P$n) {
      ids <- seq.int(start,min(P$n,start+chunk_size-1L));start<-tail(ids,1L)+1L
      part <- .evaluate(P,x,counts,gradient=gradient,indices=ids)
      weight <- length(ids)/P$n
      v <- part$value$detach()*weight
      value <- if(is.null(value)) v else value+v
      if(gradient) {
        if(is.null(part$gradient)) valid <- FALSE else if(valid)
          g <- if(is.null(g)) .scale(part$gradient,weight) else .add(g,part$gradient,weight)
      }
    }
    if(!gradient) return(list(value=value))
    return(list(value=value,gradient=if(valid) g else NULL))
  }
  if(!gradient) return(list(value=torch::with_no_grad(.value(P,x,counts,indices))))
  if(!is.null(P$value_rgrad)) {
    .count(counts,"gradient");.count(counts,"fn")
    ans<-if(inherits(P,"riem_finitesum")) {
      if(is.null(indices)) indices<-seq_len(P$n)
      .count(counts,"terms",length(indices))
      .problem_call(P,P$value_rgrad,x,indices)
    } else .problem_call(P,P$value_rgrad,x)
    if(!is.list(ans)||is.null(ans$value)||is.null(ans$gradient)) .stop("value_rgrad must return value and gradient")
    ans$value<-.loss_check(ans$value,x);.check_tangent(P$manifold,x,ans$gradient)
    return(list(value=ans$value$detach(),gradient=.clone(ans$gradient)))
  }
  .count(counts,"gradient")
  M<-P$manifold
  torch::with_enable_grad({
    z<-.clone(x,grad=is.null(P$egrad)&&is.null(P$rgrad))
    value<-.value(P,z,counts,indices)
    if(!.finite(value)) return(list(value=value,gradient=NULL))
    if(!is.null(P$rgrad)) g<-.problem_call(P,P$rgrad,z) else {
      eg<-if(!is.null(P$egrad)) .problem_call(P,P$egrad,z) else .autograd_tree(value,z)
      .check_point(M,eg)
      g<-riem.egrad2rgrad(M,z,eg)
    }
    .check_tangent(M,z,g)
    list(value=value$detach(),gradient=.clone(g))
  })
}
#' Evaluate Values and Metric Derivatives
#' @param problem A scalar, finite-sum or least-squares problem.
#' @param x A feasible point without batch axes.
#' @param gradient Whether to compute the gradient.
#' @param u Tangent direction for a Hessian-vector product.
#' @return `riem.evaluate()` returns value, optional metric gradient and counts.
#'   `riem.hessian()` returns an exact Riemannian Hessian-vector product.
#' @details Automatic Hessian conversion is available for Euclidean, sphere,
#' torus, oblique, Euclidean-metric Stiefel, both Grassmann representations,
#' generalized Stiefel/Grassmann, rotations, AIRM SPD, and weighted products of
#' supported factors, when the objective supports double backward. Objectives
#' using custom first-order matrix logarithm/root kernels require an analytic
#' `rhess` for exact second-order methods.
#' @examples
#' if (torch::torch_is_installed()) {
#'   P <- riem.problem(manifold.euclidean(2), function(x) torch::torch_sum(x^2)/2)
#'   x <- torch::torch_tensor(c(1, 2), dtype = torch::torch_float64())
#'   riem.evaluate(P, x)
#'   riem.hessian(P, x, x)
#' }
#' @export
riem.evaluate <- function(problem,x,gradient=TRUE) {
  .check_point(problem$manifold,x)
  if(!all(riem.belongs(problem$manifold,x))) .stop("x is outside the manifold")
  counts<-.new_counts();chunk<-if(inherits(problem,"riem_finitesum"))
    problem$evaluation_batch_size else NULL
  ans<-.evaluate(problem,x,counts,gradient,chunk_size=chunk)
  ans$evaluations<-as.list(counts);ans
}
.hessian<-function(P,x,u,counts,indices=NULL) {
  if(inherits(P,"riem_finitesum") && is.null(indices) && P$n>P$evaluation_batch_size && is.null(P$rhess)) {
    out <- NULL
    start<-1L
    while(start<=P$n) {
      ids <- seq.int(start,min(P$n,start+P$evaluation_batch_size-1L));start<-tail(ids,1L)+1L
      part <- .hessian(P,x,u,counts,indices=ids); weight <- length(ids)/P$n
      out <- if(is.null(out)) .scale(part,weight) else .add(out,part,weight)
    }
    return(out)
  }
  .count(counts,"hessian");M<-P$manifold;.check_tangent(M,x,u,"Hessian input")
  if(!is.null(P$rhess)) h<-.problem_call(P,P$rhess,x,u) else {
    if(!isTRUE(M$capabilities$hessian)) .stop("Exact Hessian conversion unavailable; supply rhess(x, u)")
    if(!is.null(P$rgrad)||!is.null(P$egrad)||!is.null(P$value_rgrad)) .stop("Analytic gradient problems require an explicit rhess for second-order solvers")
    previous_order <- .derivative_context$second_order
    .derivative_context$second_order <- TRUE
    on.exit({.derivative_context$second_order <- previous_order}, add = TRUE)
    h<-torch::with_enable_grad({
      z<-.clone(x,TRUE);v<-.value(P,z,counts,indices)
      .count(counts,"gradient")
      eg<-.autograd_tree(v,z,TRUE)
      eh<-.autograd_tree(.tree_dot(eg,u),z)
      riem.ehess2rhess(M,z,u,eg,eh)
    })
  }
  .check_tangent(M,x,h,"Hessian output");.clone(h)
}
#' @rdname riem.evaluate
#' @export
riem.hessian <- function(problem,x,u) .hessian(problem,x,u,.new_counts())
