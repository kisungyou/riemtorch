.broader_optimize <- function(problem,initial,method,control,device,dtype) {
  ctl<-.outer_controls(control,method)
  execution<-.resolve_execution(device,dtype,initial,
    operations=unique(c(.required_operations(problem$manifold),.probe_operations(problem$required_operations))))
  P<-.to_execution(problem,execution);P$execution<-.execution_metadata(execution)
  x<-.clone(.to_execution(initial,execution));.check_point(P$manifold,x)
  counts<-.new_counts(ctl)
  if(!all(riem.belongs(P$manifold,x))) return(.fit(P,x,NA_real_,NULL,method,ctl,counts,0,0,0,"domain_error",list()))
  if(method=="augmented_lagrangian") .augmented_lagrangian(P,x,ctl,counts) else .proximal_optimize(P,x,method,ctl,counts)
}
.maxabs <- function(v) if(is.null(v)) 0 else .scalar(v$abs()$max())
.positive_violation <- function(v) if(is.null(v)) 0 else max(0,.scalar(v$max()))
.kkt <- function(P,x,lambda,mu,counts,ev=NULL,h=NULL,g=NULL) {
  if(is.null(ev)) ev<-.evaluate(P,x,counts)
  if(is.null(h)) h<-.constraint_values(P,x,"equality",counts)
  if(is.null(g)) g<-.constraint_values(P,x,"inequality",counts)
  grad<-ev$gradient
  if(!is.null(h)) grad<-.add(grad,.constraint_adjoint(P,x,"equality",lambda,counts))
  if(!is.null(g)) grad<-.add(grad,.constraint_adjoint(P,x,"inequality",mu,counts))
  list(value=.scalar(ev$value),gradient=grad,h=h,g=g,
    feasibility=max(.maxabs(h),.positive_violation(g)),stationarity=.nrm(P$manifold,x,grad),
    complementarity=if(is.null(g)) 0 else .maxabs(mu*g),dual_feasibility=if(is.null(mu)) 0 else .positive_violation(-mu))
}
.augmented_lagrangian <- function(P,x,ctl,counts) {
  method<-"augmented_lagrangian";lambda<-mu<-NULL;rho<-ctl$penalty
  history<-inner_results<-list();iteration<-0L;termination<-"max_iterations";message<-NULL;kkt<-NULL
  err<-tryCatch({
    h<-.constraint_values(P,x,"equality",counts);g<-.constraint_values(P,x,"inequality",counts)
    lambda<-if(is.null(h)) NULL else torch::torch_zeros_like(h)
    mu<-if(is.null(g)) NULL else torch::torch_zeros_like(g)
    kkt<-.kkt(P,x,lambda,mu,counts,h=h,g=g)
    previous<-max(kkt$feasibility,kkt$complementarity)
    for(i in seq_len(ctl$max_iterations+1L)) {
      history[[length(history)+1L]]<-list(iteration=iteration,objective=kkt$value,
        feasibility=kkt$feasibility,stationarity=kkt$stationarity,
        complementarity=kkt$complementarity,penalty=rho)
      if(kkt$feasibility<=ctl$feasibility_tolerance&&kkt$stationarity<=ctl$gradient_tolerance&&
         kkt$complementarity<=ctl$complementarity_tolerance&&kkt$dual_feasibility<=ctl$feasibility_tolerance) {
        termination<-"converged_kkt";break
      }
      if(iteration>=ctl$max_iterations) break
      stopped<-.iteration_stop(P,x,kkt$value,kkt$gradient,iteration,ctl,counts)
      if(!is.null(stopped)) {termination<-stopped;break}
      augmented<-function(at) {
        ev<-.evaluate(P,at,counts);value<-ev$value;gradient<-ev$gradient
        h<-.constraint_values(P,at,"equality",counts);g<-.constraint_values(P,at,"inequality",counts)
        if(!is.null(h)) {
          if(h$numel()!=lambda$numel()) .stop("Equality dimension changed")
          value<-value+(lambda*h)$sum()+h$square()$sum()*(rho/2)
          gradient<-.add(gradient,.constraint_adjoint(P,at,"equality",lambda+h*rho,counts))
        }
        if(!is.null(g)) {
          if(g$numel()!=mu$numel()) .stop("Inequality dimension changed")
          w<-(mu+g*rho)$clamp(min=0)
          value<-value+(w$square()-mu$square())$sum()/(2*rho)
          gradient<-.add(gradient,.constraint_adjoint(P,at,"inequality",w,counts))
        }
        list(value=value,gradient=gradient)
      }
      sub<-riem.problem(P$manifold,function(z) augmented(z)$value,value_rgrad=augmented,
        required_operations=P$required_operations,
        preconditioner=if(is.null(P$preconditioner)) NULL else function(z,u) .problem_call(P,P$preconditioner,z,u))
      innerctl<-utils::modifyList(list(gradient_tolerance=max(ctl$gradient_tolerance*.1,
        ctl$inner_accuracy/(10^min(iteration,12L))),max_iterations=100L),ctl$inner_control)
      elapsed<-proc.time()[[3]]-attr(counts,"runtime")$start
      innerctl$max_time<-min(if(is.null(innerctl$max_time)) Inf else innerctl$max_time,ctl$max_time-elapsed)
      .budget_check(counts)
      result<-riem.optimize(sub,x,ctl$inner_method,innerctl,device=.leaves(x)[[1]]$device)
      inner_results[[length(inner_results)+1L]]<-list(termination=result$termination,
        iterations=result$iterations,gradient_norm=result$gradient_norm,objective=result$objective)
      if(result$termination %in% c("numerical_failure","domain_error","nonfinite_objective","line_search_failed","time_budget","evaluation_budget","user_stopped")) {
        termination<-if(result$termination %in% c("time_budget","evaluation_budget","user_stopped")) result$termination else "inner_failure"
        message<-result$message;break
      }
      candidate_x<-.clone(result$point)
      h<-.constraint_values(P,candidate_x,"equality",counts);g<-.constraint_values(P,candidate_x,"inequality",counts)
      candidate_lambda<-if(is.null(h)) NULL else .clone(lambda+h*rho)
      candidate_mu<-if(is.null(g)) NULL else .clone((mu+g*rho)$clamp(min=0))
      candidate_kkt<-.kkt(P,candidate_x,candidate_lambda,candidate_mu,counts,h=h,g=g)
      x<-candidate_x;lambda<-candidate_lambda;mu<-candidate_mu;kkt<-candidate_kkt
      iteration<-iteration+1L
      violation<-max(kkt$feasibility,kkt$complementarity)
      if(violation>ctl$penalty_contraction*previous) rho<-rho*ctl$penalty_increase
      if(!is.finite(rho)) .stop("Penalty overflow")
      previous<-violation
    }
    NULL
  },error=function(e)e)
  if(inherits(err,"error")) {termination<-.error_termination(err);message<-conditionMessage(err)}
  # If a budget interrupted the diagnostics after a new point, never report stale
  # residuals for that point. The last completed KKT point is retained explicitly.
  if(is.null(kkt)) kkt<-list(value=NA_real_,gradient=NULL,feasibility=NA_real_,stationarity=NA_real_,complementarity=NA_real_,dual_feasibility=NA_real_)
  fit<-.fit(P,x,kkt$value,kkt$gradient,method,ctl,counts,iteration,iteration,0L,termination,history,
    diagnostics=list(model="augmented_lagrangian",inner=inner_results,penalty=rho),message=message)
  fit$kkt<-kkt[c("feasibility","stationarity","complementarity","dual_feasibility")]
  fit$multipliers<-list(equality=if(is.null(lambda)) NULL else .clone(lambda),inequality=if(is.null(mu)) NULL else .clone(mu))
  fit$constraint_values<-list(equality=kkt$h,inequality=kkt$g)
  fit$converged<-identical(termination,"converged_kkt");fit
}
.nonsmooth_value <- function(P,x,counts) {
  fs<-if(P$cyclic) P$nonsmooth else list(P$nonsmooth)
  values<-lapply(fs,function(fn) {.count(counts,"fn");.loss_check(.problem_call(P,fn,x),x)})
  value<-Reduce(`+`,values)
  if(is.nan(.scalar(value))||.scalar(value)==-Inf) .stop("Invalid nonsmooth value")
  value
}
.prox_call <- function(P,fn,q,step,counts) {
  .count(counts,"proximal")
  y<-torch::with_no_grad(.problem_call(P,fn,.clone(q),step))
  .check_point(P$manifold,y)
  if(!.allfinite(y)||!all(riem.belongs(P$manifold,y))) .stop("Proximal callback returned an infeasible point")
  .clone(y)
}
.proximal_optimize <- function(P,x,method,ctl,counts) {
  M<-P$manifold;cyclic<-method=="cyclic_proximal_point"
  if(cyclic!=P$cyclic) .stop("Cyclic proximal point requires a list of terms; proximal gradient requires a scalar nonsmooth term")
  caps<-riem.capabilities(M)
  if(!all(vapply(c("exp","log","sqdist"),function(op) all(caps$value[caps$operation==op]),logical(1))))
    .stop("Intrinsic proximal methods require exact exponential, logarithm and distance maps")
  counts$proximal<-0L;f<-NA_real_;g<-NULL;residual<-Inf
  history<-list();iteration<-accepted<-rejected<-0L;termination<-"max_iterations";message<-NULL
  err<-tryCatch({
    ev<-.evaluate(P,x,counts,gradient=!cyclic);g<-ev$gradient
    f<-.scalar(ev$value+.nonsmooth_value(P,x,counts))
    if(!is.finite(f)) .stop("Initial composite objective must be finite")
    for(it in seq_len(ctl$max_iterations)) {
      stopped<-.iteration_stop(P,x,f,NULL,it-1L,ctl,counts)
      if(!is.null(stopped)) {termination<-stopped;break}
      step<-if(is.null(ctl$schedule)) ctl$step_size/(if(cyclic) it^ctl$cycle_power else 1) else ctl$schedule(it-1L)
      .number(step,"proximal step")
      if(cyclic) {
        y<-x
        for(fn in P$prox) y<-.prox_call(P,fn,y,step,counts)
        residual<-.scalar(riem.dist(M,x,y))/step
        fy<-.scalar(.evaluate(P,y,counts,FALSE)$value+.nonsmooth_value(P,y,counts));success<-is.finite(fy)
      } else {
        ev<-.evaluate(P,x,counts);g<-ev$gradient;success<-FALSE
        for(k in seq_len(ctl$max_linesearch)) {
          q<-torch::with_no_grad(riem.exp(M,x,.scale(g,-step)))
          y<-.prox_call(P,P$prox,q,step,counts)
          movement<-riem.log(M,x,y);residual<-.nrm(M,x,movement)/step
          fy<-.scalar(.evaluate(P,y,counts,FALSE)$value+.nonsmooth_value(P,y,counts))
          if(is.finite(fy)&&fy<=f-ctl$armijo*step*residual^2+10*.Machine$double.eps*(1+abs(f))) {success<-TRUE;break}
          rejected<-rejected+1L;step<-step*ctl$backtrack
          if(step<ctl$min_step) break
        }
      }
      if(!success) {termination<-"line_search_failed";break}
      # Residual is evaluated at x. Retain that point when it passes, so that
      # the reported objective and residual refer to the same iterate.
      if(residual<=ctl$proximal_tolerance) {
        termination<-if(cyclic) "stopped_cycle_residual" else "converged_proximal_residual";break
      }
      x<-.clone(y);f<-fy;iteration<-it;accepted<-accepted+1L
      history[[length(history)+1L]]<-list(iteration=it,objective=f,proximal_residual=residual,step=step,
        residual_at="previous_iterate",accepted=TRUE)
    }
    NULL
  },error=function(e)e)
  if(inherits(err,"error")) {termination<-.error_termination(err);message<-conditionMessage(err)}
  fit<-.fit(P,x,f,NULL,method,ctl,counts,iteration,accepted,rejected,termination,history,
    diagnostics=list(model=method,qualified=P$qualified,residual_kind=if(cyclic) "cycle_displacement_over_step" else "intrinsic_gradient_mapping",
      residual_at=if(termination %in% c("stopped_cycle_residual","converged_proximal_residual")) "returned_point" else "last_tested_point"),message=message)
  fit$proximal_residual<-residual;fit$converged<-identical(termination,"converged_proximal_residual");fit
}
