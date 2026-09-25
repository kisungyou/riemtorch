.stop_run <- function(reason) stop(structure(list(message=reason,call=NULL,termination=reason),
  class=c("riem_stop","error","condition")))
.error_termination <- function(e) if(inherits(e,"riem_stop")) e$termination else "numerical_failure"
.budget_check <- function(counts) {
  runtime<-attr(counts,"runtime");if(is.null(runtime)) return(invisible(NULL))
  ctl<-runtime$control
  if(proc.time()[[3]]-runtime$start>=ctl$max_time) .stop_run("time_budget")
  used<-sum(unlist(as.list(counts))[setdiff(names(as.list(counts)),"terms")])
  if(used>=ctl$max_evaluations) .stop_run("evaluation_budget")
  invisible(NULL)
}
.iteration_stop <- function(P,x,f,g,iteration,ctl,counts) {
  check<-tryCatch({.budget_check(counts);NULL},riem_stop=function(e) e$termination)
  if(!is.null(check)) return(check)
  if(!is.null(ctl$callback)) {
    runtime<-attr(counts,"runtime")
    state<-list(point=.clone(x),objective=f,gradient=if(is.null(g)) NULL else .clone(g),
      iteration=iteration,evaluations=as.list(counts),event="iteration",
      elapsed=if(is.null(runtime)) 0 else proc.time()[[3]]-runtime$start)
    result<-ctl$callback(state)
    if(!is.null(result)&&(!is.logical(result)||length(result)!=1||is.na(result)))
      .stop("callback must return TRUE, FALSE, or NULL")
    if(isTRUE(result)) return("user_stopped")
  }
  NULL
}
.precondition <- function(P,x,u) {
  if(is.null(P$preconditioner)) return(u)
  z<-.problem_call(P,P$preconditioner,x,u);.check_tangent(P$manifold,x,z,"Preconditioner output")
  if(.nrm(P$manifold,x,u)>0&&.ip(P$manifold,x,u,z)<=0)
    .stop("Preconditioner must be positive definite")
  z
}
.sample_terms <- function(P,size) {
  if(P$sampling=="without_replacement")
    as.integer(torch::torch_randperm(P$n,dtype=torch::torch_int64())+1)[seq_len(min(size,P$n))]
  else as.integer(torch::torch_randint(1,P$n+1,c(size),dtype=torch::torch_int64()))
}
.variance_optimize <- function(P,x,method,ctl,counts) {
  if(!inherits(P,"riem_finitesum")) .stop(method," requires a finite-sum problem")
  M<-P$manifold
  if(method=="svrg"&&(!isTRUE(M$capabilities$snapshot_transport)||!all(riem.capabilities(M)$value[riem.capabilities(M)$operation=="log"])))
    .stop("SVRG requires a supported local logarithm and endpoint-compatible snapshot transport")
  full<-function(at) .evaluate(P,at,counts,chunk_size=P$evaluation_batch_size)
  initial<-tryCatch(full(x),error=function(e)e)
  if(inherits(initial,"error")) return(.fit(P,x,NA_real_,NULL,method,ctl,counts,0,0,0,
    .error_termination(initial),list(),message=conditionMessage(initial)))
  if(!.finite(initial$value)||is.null(initial$gradient)) return(.fit(P,x,NA_real_,NULL,method,ctl,counts,0,0,0,"nonfinite_objective",list()))
  f<-.scalar(initial$value);g<-initial$gradient;snapshot<-.clone(x);mu<-.clone(g);direction<-.clone(g)
  history<-list(list(iteration=0,objective=f,gradient_norm=.nrm(M,x,g),evaluation="full"))
  accepted<-it<-0L;termination<-"max_iterations";message<-NULL;full_current<-TRUE
  if(.nrm(M,x,g)<=ctl$gradient_tolerance) termination<-"converged_gradient" else
  for(it in seq_len(ctl$max_iterations)) {
    stopped<-.iteration_stop(P,x,f,g,it-1L,ctl,counts)
    if(!is.null(stopped)) {termination<-stopped;break}
    ans<-tryCatch({
      period<-if(method=="svrg") ctl$epoch_length else ctl$snapshot_every
      refresh<-(it-1L)%%period==0L
      if(refresh) {
        ev<-if(it==1L) initial else full(x)
        snapshot<-.clone(x);mu<-.clone(ev$gradient);direction<-.clone(mu)
      } else {
        ids<-.sample_terms(P,ctl$batch_size)
        current<-.evaluate(P,x,counts,indices=ids)$gradient
        if(method=="svrg") {
          old<-.evaluate(P,snapshot,counts,indices=ids)$gradient
          path<-riem.log(M,snapshot,x)
          correction<-riem.transport(M,snapshot,path,x,.add(old,mu,-1))
          direction<-.add(current,correction,-1)
        } else {
          old<-.evaluate(P,previous,counts,indices=ids)$gradient
          direction<-.add(current,riem.transport(M,previous,previous_step,x,.add(direction,old,-1)))
          # The recursive estimator is current - T(old) + T(previous estimator).
        }
      }
      step<-if(is.null(ctl$schedule)) ctl$step_size else ctl$schedule(it-1L)
      .number(step,"scheduled step",strict=FALSE)
      eta<-.scale(direction,-step);y<-torch::with_no_grad(riem.retr(M,x,eta))
      if(!.allfinite(y)||!all(riem.belongs(M,y))) .stop("Variance-reduced step left the manifold")
      list(point=.clone(y),step=eta)
    },error=function(e)e)
    if(inherits(ans,"error")) {termination<-.error_termination(ans);message<-conditionMessage(ans);break}
    previous<-x;previous_step<-ans$step;x<-ans$point;accepted<-accepted+1L
    full_current<-FALSE;f<-NA_real_;g<-NULL
    history[[length(history)+1L]]<-list(iteration=it,evaluation="variance_reduced",objective=NA_real_,gradient_norm=NA_real_,accepted=TRUE)
    if(ctl$full_evaluation_every>0L&&it%%ctl$full_evaluation_every==0L) {
      ev<-tryCatch(full(x),error=function(e)e)
      if(inherits(ev,"error")) {termination<-.error_termination(ev);message<-conditionMessage(ev);break}
      f<-.scalar(ev$value);g<-ev$gradient;full_current<-TRUE
      history[[length(history)]]$objective<-f;history[[length(history)]]$gradient_norm<-.nrm(M,x,g)
      if(.nrm(M,x,g)<=ctl$gradient_tolerance) {termination<-"converged_gradient";break}
    }
  }
  if(!full_current&&!termination%in%c("evaluation_budget","time_budget","user_stopped","numerical_failure")) {
    ev<-tryCatch(full(x),error=function(e)e)
    if(inherits(ev,"error")) {termination<-.error_termination(ev);message<-conditionMessage(ev)} else {
      f<-.scalar(ev$value);g<-ev$gradient
      if(.nrm(M,x,g)<=ctl$gradient_tolerance) termination<-"converged_gradient"
    }
  }
  .fit(P,x,f,g,method,ctl,counts,it,accepted,0L,termination,history,
    diagnostics=list(model="variance_reduction",stationarity="full_gradient"),message=message)
}
#' Run Reproducible Multiple Starts on One Device
#' @param problem An optimization problem.
#' @param initials Optional list of feasible initial points.
#' @param starts Number of random starts when initials is NULL.
#' @param seed Integer initialization seed. NULL uses the current RNG stream.
#' @param method,control Solver and controls passed to [riem.optimize()].
#' @param device,dtype Execution requests resolved once for all starts.
#' @return A list of fits, the index of the best finite feasible fit, its fit,
#'   and execution metadata. Failed fits remain in the result.
#' @examples
#' if (torch::torch_is_installed()) {
#'   P <- riem.problem(manifold.sphere(3),function(x) -x[1])
#'   riem.multistart(P,starts=2,device="cpu")$best$objective
#' }
#' @export
riem.multistart <- function(problem,initials=NULL,starts=5L,seed=1L,
                           method="steepest_descent",control=list(),device=NULL,dtype=NULL) {
  if(!inherits(problem,"riem_problem")) .stop("Expected a riem_problem")
  if(missing(method)) {
    if(inherits(problem,"riem_constrained")) method<-"augmented_lagrangian"
    if(inherits(problem,"riem_composite")) method<-if(problem$cyclic) "cyclic_proximal_point" else "proximal_gradient"
  }
  if(!is.null(seed)) {
    if(length(seed)!=1||!is.finite(seed)||seed!=floor(seed)) .stop("seed must be an integer")
    state<-torch::torch_get_rng_state()
    cuda_states<-if(isTRUE(torch::cuda_is_available())) lapply(seq_len(torch::cuda_device_count()),function(i) torch::cuda_get_rng_state(i-1L)) else list()
    on.exit({torch::torch_set_rng_state(state)
      for(i in seq_along(cuda_states)) torch::cuda_set_rng_state(cuda_states[[i]],i-1L)},add=TRUE)
    torch::torch_manual_seed(seed)
  }
  if(is.null(dtype)&&is.null(initials)&&identical(problem$manifold$specification$field,"complex")) dtype<-"complex128"
  execution<-.resolve_execution(device,dtype,if(is.null(initials)) problem else initials,
    operations=unique(c(.required_operations(problem$manifold),problem$required_operations)))
  P<-.to_execution(problem,execution)
  if(is.null(initials)) {
    starts<-.positive_integer(starts,"starts");if(length(starts)!=1) .stop("starts must be scalar")
    initials<-lapply(seq_len(starts),function(i) .random_execution(P$manifold,integer(),execution))
  }
  if(!is.list(initials)||!length(initials)) .stop("initials must be a nonempty list")
  fits<-lapply(initials,function(x) riem.optimize(P,x,method,control,execution$device,execution$dtype))
  feasible<-vapply(fits,function(f) {
    ok<-is.finite(f$objective)&&all(riem.belongs(P$manifold,f$point))
    if(inherits(P,"riem_constrained")) ok<-ok&&isTRUE(is.finite(f$kkt$feasibility)&&f$kkt$feasibility<=f$control$feasibility_tolerance)
    ok
  },logical(1))
  values<-vapply(seq_along(fits),function(i) if(feasible[i]) fits[[i]]$objective else Inf,numeric(1))
  best<-if(all(!is.finite(values))) NA_integer_ else which.min(values)
  list(fits=fits,feasible=feasible,best_index=best,best=if(is.na(best)) NULL else fits[[best]],execution=.execution_metadata(execution))
}
