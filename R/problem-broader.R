#' Smooth Equality and Inequality Constrained Problems
#' @inheritParams riem.problem
#' @param equality,inequality Optional functions returning real vector tensors.
#'   Equalities are h(x)=0; inequalities are g(x)<=0. At least one is required.
#' @param equality_adjoint,inequality_adjoint Optional metric adjoints
#'   `function(x, weights)` returning the Riemannian gradient of the weighted
#'   constraint sum. Omitted adjoints use automatic differentiation.
#' @return A constrained `riem_problem` for `method="augmented_lagrangian"`.
#' @details Registered data is passed to every callback. The solver uses the
#'   Powell--Hestenes--Rockafellar inequality penalty and smooth first-order
#'   inner solvers. Feasibility, Lagrangian stationarity, complementarity,
#'   nonnegative inequality multipliers and inner-solver status are reported.
#'   A KKT stopping test does not certify a global optimum.
#' @examples
#' if(torch::torch_is_installed()) {
#'   P <- riem.problem.constrained(manifold.euclidean(2),
#'     function(x) ((x-2)^2)$sum()/2, equality=function(x) x$sum()$reshape(1)-1)
#'   fit <- riem.optimize(P,torch::torch_zeros(2,dtype=torch::torch_float64()),
#'                        "augmented_lagrangian",device="cpu")
#'   fit$kkt
#' }
#' @export
riem.problem.constrained <- function(manifold,fn,equality=NULL,inequality=NULL,
  egrad=NULL,rgrad=NULL,equality_adjoint=NULL,inequality_adjoint=NULL,label=NULL,
  data=NULL,required_operations=NULL,preconditioner=NULL) {
  if(is.null(equality)&&is.null(inequality)) .stop("At least one constraint callback is required")
  for(f in list(equality,inequality,equality_adjoint,inequality_adjoint))
    if(!is.null(f)&&!is.function(f)) .stop("Constraint callbacks must be functions")
  if(is.null(equality)&&!is.null(equality_adjoint)||is.null(inequality)&&!is.null(inequality_adjoint))
    .stop("An adjoint requires its constraint callback")
  P<-riem.problem(manifold,fn,egrad=egrad,rgrad=rgrad,label=label,data=data,
    required_operations=required_operations,preconditioner=preconditioner)
  P$equality<-equality;P$inequality<-inequality
  P$equality_adjoint<-equality_adjoint;P$inequality_adjoint<-inequality_adjoint
  class(P)<-c("riem_constrained",class(P));P
}
#' Smooth Plus Nonsmooth Manifold Problems
#' @inheritParams riem.problem
#' @param nonsmooth Function returning h(x), a real scalar tensor (positive
#'   infinity can encode infeasibility), or a list of such functions for cyclic
#'   proximal point. In that case `fn` must be NULL.
#' @param prox Function `prox(q, step)` solving exactly
#'   argmin_y h(y)+d(y,q)^2/(2*step), or a matching list for cyclic proximal point.
#' @param qualified Whether to require an initially qualified geometry. TRUE
#'   supports Euclidean, affine, log-Euclidean positive/SPD, hyperbolic and their
#'   products, powers and fixed metric scalings. FALSE permits experimental use
#'   on other geometries with exact exponential, logarithm and distance maps.
#' @return A composite `riem_problem` for `"proximal_gradient"` or
#'   `"cyclic_proximal_point"`.
#' @details A proximal callback is a mathematical contract; ambient shrinkage
#'   followed by projection generally does not satisfy it. Callbacks must use
#'   the declared metric (including scale and product weights). The proximal
#'   gradient solver uses an exponential gradient step and intrinsic prox,
#'   with sufficient-decrease backtracking. Cyclic proximal point uses a
#'   diminishing step sequence and reports a cycle residual, which alone is
#'   not a stationarity certificate for a sum of nonsmooth functions.
#' @examples
#' if(torch::torch_is_installed()) {
#'   P <- riem.problem.composite(manifold.euclidean(2),
#'     function(x) ((x-2)^2)$sum()/2, nonsmooth=function(x) x$abs()$sum(),
#'     prox=function(q,step) q$sign()*(q$abs()-step)$clamp(min=0))
#'   riem.optimize(P,torch::torch_zeros(2,dtype=torch::torch_float64()),
#'                 "proximal_gradient",device="cpu")$point
#' }
#' @export
riem.problem.composite <- function(manifold,fn=NULL,nonsmooth,prox,egrad=NULL,rgrad=NULL,
  label=NULL,data=NULL,required_operations=NULL,qualified=TRUE) {
  .check_manifold(manifold)
  if(!is.logical(qualified)||length(qualified)!=1||is.na(qualified)) .stop("qualified must be TRUE or FALSE")
  cyclic<-is.list(nonsmooth)
  if(cyclic) {
    if(!is.null(fn)||!is.list(prox)||!length(nonsmooth)||length(prox)!=length(nonsmooth)||
       any(!vapply(c(nonsmooth,prox),is.function,logical(1))))
      .stop("Cyclic problems require matching nonempty lists of values and proximal maps, with fn=NULL")
  } else if(!is.function(nonsmooth)||!is.function(prox)) .stop("nonsmooth and prox must be functions")
  if(qualified&&!.prox_qualified(manifold)) .stop("This geometry is experimental for intrinsic proximal methods; set qualified=FALSE explicitly")
  if(is.null(fn)) fn<-function(x,...) .tree_dot(x,x)*0
  P<-riem.problem(manifold,fn,egrad=egrad,rgrad=rgrad,label=label,data=data,required_operations=required_operations)
  P$nonsmooth<-nonsmooth;P$prox<-prox;P$cyclic<-cyclic;P$qualified<-qualified
  class(P)<-c("riem_composite",class(P));P
}
.prox_qualified <- function(M) {
  if(inherits(M,"riem_product")) return(all(vapply(M$factors,.prox_qualified,logical(1))))
  if(!is.null(M$base)) return(.prox_qualified(M$base))
  M$name %in% c("euclidean","euclidean.complex","positive","affine","hyperbolic")||M$name=="spd"&&M$metric=="lerm"
}
.constraint_values <- function(P,x,kind,counts) {
  f<-P[[kind]]
  if(is.null(f)) return(NULL)
  .count(counts,"residual")
  v<-.problem_call(P,f,x);ref<-.leaves(x)[[1]]
  if(!inherits(v,"torch_tensor")||length(v$shape)!=1||v$numel()<1||v$dtype!=.real_dtype(ref$dtype)||
     .device_string(v$device)!=.device_string(ref$device)||!.finite(v))
    .stop(kind," must return a finite real vector on the point device and precision")
  v
}
.constraint_adjoint <- function(P,x,kind,w,counts) {
  if(is.null(w)) return(.zeros(x))
  .count(counts,"adjoint");cb<-P[[paste0(kind,"_adjoint")]]
  if(!is.null(cb)) out<-.problem_call(P,cb,x,w) else {
    z<-.clone(x,TRUE);v<-.constraint_values(P,z,kind,counts)
    out<-riem.egrad2rgrad(P$manifold,z,.autograd_tree((v*w)$sum(),z))
  }
  .check_tangent(P$manifold,x,out,"Constraint adjoint");.clone(out)
}
.outer_controls <- function(control,type) {
  extra<-if(type=="augmented_lagrangian") list(penalty=10,penalty_increase=5,
    feasibility_tolerance=1e-6,complementarity_tolerance=1e-6,penalty_contraction=.5,
    inner_method="lbfgs",inner_control=list(max_iterations=100L),inner_accuracy=.1) else
    list(proximal_tolerance=1e-7,cycle_power=.75)
  selected<-intersect(names(control),names(extra));extra<-utils::modifyList(extra,control[selected])
  ctl<-.controls(control[setdiff(names(control),names(extra))])
  if(type=="augmented_lagrangian") {
    for(k in c("penalty","penalty_increase","feasibility_tolerance","complementarity_tolerance","inner_accuracy")) .number(extra[[k]],k)
    if(extra$penalty_increase<=1||extra$penalty_contraction<=0||extra$penalty_contraction>=1) .stop("Invalid penalty controls")
    extra$inner_method<-match.arg(extra$inner_method,c("lbfgs","steepest_descent","conjugate_gradient","barzilai_borwein"))
    .controls(extra$inner_control)
  } else {
    .number(extra$proximal_tolerance,"proximal_tolerance",strict=FALSE)
    if(extra$cycle_power<=.5||extra$cycle_power>1) .stop("cycle_power must be in (0.5,1]")
  }
  c(ctl,extra)
}
