#' Optimize a User-Defined Manifold Objective
#'
#' Generic solvers share metric derivatives, feasibility checks and diagnostics.
#' @param problem A problem from [riem.problem()], [riem.problem.leastsquares()]
#'   or [riem.problem.finitesum()].
#' @param initial A feasible tensor or named product, without batch axes.
#' @param method Solver key; see Details.
#' @param device Execution device request. `NULL` follows [riem.device()]; use
#'   `"cpu"` to force CPU or a value such as `"cuda:1"` to select a GPU.
#' @param dtype Optional floating-point dtype. When omitted, the initial point's
#'   precision is preserved and automatic selection skips incompatible devices.
#' @param control Named options. Common options: `max_iterations` (200),
#'   `gradient_tolerance` (1e-7), `step_tolerance` (1e-12), `step_size` (1),
#'   `line_search` (`"armijo"`, `"adaptive_armijo"`, `"strong_wolfe"`, `"fixed"`),
#'   `max_linesearch` (30), `armijo` (1e-4), `wolfe` (0.9), `backtrack` (0.5),
#'   `min_step` (1e-14), `max_step` (100), and `history` (TRUE).
#'   Execution controls: `callback` (NULL), `max_time` (Inf seconds),
#'   `max_evaluations` (Inf primitive evaluations, excluding the term counter).
#'   Callbacks receive detached iteration states and a final event; TRUE stops
#'   an iteration event. Final events are observational. Time limits are checked
#'   between operations and cannot interrupt a running backend kernel.
#'   `cg_beta` selects PR+, FR, HS+ or DY; `bb_variant` selects BB1, BB2 or
#'   alternating. `epoch_length` and `snapshot_every` default to 100 for SVRG/SRG.
#'   Secant controls: `memory` (10), `restart_every` (50).
#'   Model controls: `trust_radius` (1), `max_trust_radius` (100),
#'   `acceptance_ratio` (0.1), `inner_iterations` (50), `inner_tolerance` (0.1),
#'   `damping` (0.01), `regularization` (1).
#'   Stochastic/block controls: `batch_size` (1), `full_evaluation_every` (0,
#'   meaning only initial and final full evaluations), `schedule` (NULL,
#'   constant step_size), `blocks` (NULL, cyclic factor names).
#'   Search controls: `population` (20), `temperature` (1), `cooling` (0.95),
#'   `proposal_scale` (0.2), `inertia` (0.5), `cognitive` (1), `social` (1),
#'   `elite_fraction` (0.25). Unknown options are errors.
#' @return A `riem_fit`: point, objective, gradient_norm, constraint_residuals,
#'   manifold, method, control, iterations, evaluations, accepted_steps,
#'   rejected_steps, termination, converged, history, diagnostics and message.
#' @details Constrained problems use `augmented_lagrangian`. Additional controls
#'   are `penalty` (10), `penalty_increase` (5), `penalty_contraction` (0.5),
#'   `feasibility_tolerance` and `complementarity_tolerance` (1e-6),
#'   `inner_method` ("lbfgs"), `inner_control` (list(max_iterations=100)), and
#'   `inner_accuracy` (0.1, tightened each outer iteration). Composite problems
#'   use `proximal_gradient` or `cyclic_proximal_point`, with
#'   `proximal_tolerance` (1e-7) and `cycle_power` (0.75, in (0.5,1]).
#'   Diminishing cyclic step sizes alone cannot certify stationarity; a small
#'   cycle residual reports a stop without a convergence claim.
#'
#'   First-order methods are `steepest_descent`, `conjugate_gradient`
#'   (transported PR+ with descent restarts), `barzilai_borwein` (BB1), and `lbfgs`
#'   (transported secants and curvature rejection). Quasi-Newton secants and
#'   stored vectors move to the accepted point before reuse; projection transports
#'   give practical variants without a universal convergence guarantee.
#'
#'   `trust_regions` uses Steihaug truncated CG; `newton_cg` uses truncated CG
#'   plus line search. `adaptive_regularization_cubics` uses a Cauchy-initialized
#'   approximate cubic subproblem. These require an exact `rhess` callback or
#'   verified automatic Hessian conversion. `gauss_newton` and
#'   `levenberg_marquardt` require a least-squares problem, use the metric adjoint,
#'   and report Gauss--Newton rather than exact Hessian models. LM adapts damping
#'   with actual/predicted reduction. None of these certifies a global optimum.
#'
#'   `svrg` and `srg` add finite-sum variance reduction. SVRG requires
#'   an exact local logarithm to identify its snapshot transport path.
#'   `stochastic_gradient` requires a finite sum. It uses chunked full initial
#'   and final evaluations; positive `full_evaluation_every` requests additional
#'   full checks. Other iterations report explicitly labelled minibatch
#'   diagnostics, which cannot establish gradient convergence.
#'   `alternating_gradient` updates one named product block per iteration.
#'   Derivative-free methods are `particle_swarm`, `nelder_mead`,
#'   `stiefel_annealing`, and `grassmann_macg`; see the installed mathematical
#'   notes for their local movement and sampling rules. They return no
#'   gradient-based convergence claim.
#'
#'   Strong Wolfe differentiates the actual retraction curve. It is enabled only
#'   on geometries advertising that derivative. Small-step termination is
#'   `stopped_small_step`, distinct from `converged_gradient`. Rejected domain
#'   trials are counted; accepted iterates are never silently repaired.
#' @examples
#' if (torch::torch_is_installed()) {
#'   torch::torch_manual_seed(1)
#'   M <- manifold.sphere(3)
#'   a <- torch::torch_tensor(c(1, 2, 3), dtype = torch::torch_float64())
#'   P <- riem.problem(M, function(x, data) -torch::torch_sum(data*x), data = a)
#'   fit <- riem.optimize(P, riem.random(M, device = "cpu"),
#'                        "conjugate_gradient", device = "cpu")
#'   fit
#'   fit$point
#' }
#' @export
riem.optimize <- function(problem,initial,method="steepest_descent",control=list(),
                          device=NULL,dtype=NULL) {
  if(!inherits(problem,"riem_problem")) .stop("problem must be a riem_problem")
  if(inherits(problem,"riem_constrained")||inherits(problem,"riem_composite")) {
    expected<-if(inherits(problem,"riem_constrained")) "augmented_lagrangian" else
      if(problem$cyclic) "cyclic_proximal_point" else "proximal_gradient"
    if(missing(method)) method<-expected
    if(!identical(method,expected)) .stop("This problem requires method=",expected)
    return(.broader_optimize(problem,initial,method,control,device,dtype))
  }
  methods<-c("steepest_descent","conjugate_gradient","barzilai_borwein","lbfgs",
    "trust_regions","newton_cg","adaptive_regularization_cubics","gauss_newton",
    "levenberg_marquardt","stochastic_gradient","svrg","srg","alternating_gradient",
    "particle_swarm","nelder_mead","stiefel_annealing","grassmann_macg")
  method<-match.arg(method,methods);ctl<-.controls(control)
  execution<-.resolve_execution(device=device,dtype=dtype,reference=initial,operations=unique(c(.required_operations(problem$manifold),.probe_operations(problem$required_operations))))
  P<-.to_execution(problem,execution);initial<-.to_execution(initial,execution)
  P$execution<-.execution_metadata(execution)
  M<-P$manifold
  .check_point(M,initial);x<-.clone(initial);counts<-.new_counts(ctl)
  if(!all(riem.belongs(M,x))) return(.fit(P,x,NA_real_,NULL,method,ctl,counts,0L,0L,0L,"domain_error",list(),message="Initial point is outside the manifold"))
  if(method %in% c("svrg","srg")) return(.variance_optimize(P,x,method,ctl,counts))
  if(method %in% c("particle_swarm","nelder_mead","stiefel_annealing","grassmann_macg")) return(.search_optimize(P,x,method,ctl,counts))
  exact<-method %in% c("trust_regions","newton_cg","adaptive_regularization_cubics")
  lsq<-method %in% c("gauss_newton","levenberg_marquardt")
  if(exact && is.null(P$rhess) && !isTRUE(M$capabilities$hessian)) .stop("Method requires exact rhess or a geometry with verified Hessian conversion")
  if(lsq && !inherits(P,"riem_leastsquares")) .stop("Method requires riem.problem.leastsquares")
  if(method=="stochastic_gradient"&&!inherits(P,"riem_finitesum")) .stop("stochastic_gradient requires riem.problem.finitesum")
  if(method=="alternating_gradient"&&!inherits(M,"riem_product")) .stop("alternating_gradient requires a product manifold")
  if(ctl$line_search=="strong_wolfe"&&!isTRUE(M$capabilities$curve_derivative)) .stop("Strong Wolfe requires a differentiable retraction")
  blocks<-if(is.null(ctl$blocks)) names(M$factors) else ctl$blocks
  if(method=="alternating_gradient" && (!setequal(blocks,names(M$factors))||anyDuplicated(blocks))) .stop("blocks must list each factor exactly once")
  full_chunk<-if(inherits(P,"riem_finitesum")) P$evaluation_batch_size else NULL
  evaluated<-tryCatch(.evaluate(P,x,counts,chunk_size=full_chunk),error=function(e) e)
  if(inherits(evaluated,"error")) return(.fit(P,x,NA_real_,NULL,method,ctl,counts,0L,0L,0L,.error_termination(evaluated),list(),message=conditionMessage(evaluated)))
  f<-.scalar(evaluated$value);g<-evaluated$gradient
  if(!is.finite(f)) return(.fit(P,x,f,NULL,method,ctl,counts,0L,0L,0L,"nonfinite_objective",list()))
  accepted<-rejected<-iterations<-0L;termination<-"max_iterations";message<-NULL
  history<-list();pairs<-list();direction<-NULL;alpha<-ctl$step_size
  radius<-ctl$trust_radius;lambda<-ctl$damping;sigma<-ctl$regularization
  diagnostics<-list(restarts=0L,skipped_secants=0L,subproblems=list(),model=if(lsq) "gauss_newton" else if(exact) "exact_hessian" else "first_order")
  ng<-.nrm(M,x,g)
  if(!is.finite(ng)) return(.fit(P,x,f,g,method,ctl,counts,0L,0L,0L,"numerical_failure",list(),message="Nonfinite metric gradient norm"))
  history[[1]]<-list(iteration=0L,objective=f,gradient_norm=ng,accepted=NA,
    step=0,evaluation=if(method=="stochastic_gradient") "full" else NULL)
  full_at_current<-TRUE
  if(ng<=ctl$gradient_tolerance) termination<-"converged_gradient" else
  for(it in seq_len(ctl$max_iterations)) {
    stopped<-.iteration_stop(P,x,f,g,it-1L,ctl,counts)
    if(!is.null(stopped)) {termination<-stopped;break}
    iterations<-it;oldx<-x;oldg<-g;oldf<-f;ratio<-NA_real_;sub<-NULL
    ans<-tryCatch({
      if(method=="stochastic_gradient") {
        if(P$sampling=="without_replacement") {
          ids<-as.integer(torch::torch_randperm(P$n,dtype=torch::torch_int64()) + 1)[seq_len(min(ctl$batch_size,P$n))]
        } else ids<-as.integer(torch::torch_randint(low=1,high=P$n+1,size=c(ctl$batch_size),dtype=torch::torch_int64()))
        minibatch<-.evaluate(P,x,counts,indices=ids)
        sg<-minibatch$gradient
        if(is.null(sg)) .stop("Nonfinite minibatch objective")
        batch_f<-.scalar(minibatch$value);batch_ng<-.nrm(M,x,sg)
        if(!is.finite(batch_f)||!is.finite(batch_ng)) .stop("Nonfinite minibatch objective or gradient")
        step<-if(is.null(ctl$schedule)) ctl$step_size else ctl$schedule(it-1)
        .number(step,"scheduled step",strict=FALSE)
        direction<-.scale(sg,-1)
        trial<-tryCatch({
          y<-torch::with_no_grad(riem.retr(M,x,direction,step))
          ok<-.allfinite(y)&&all(riem.belongs(M,y))
          list(ok=ok,x=y,reason=if(ok) NULL else "domain_error")
        },error=function(e) list(ok=FALSE,reason="domain_error",message=conditionMessage(e)))
        trial$step<-step;trial$direction<-direction
        trial$rejected<-if(trial$ok) 0L else 1L
        trial$batch_objective<-batch_f;trial$batch_gradient_norm<-batch_ng
        trial$indices<-ids
      } else if(exact||lsq) {
        H<-if(lsq) function(u) .gn(P,x,u,counts) else function(u) .hessian(P,x,u,counts)
        if(method=="adaptive_regularization_cubics") {
          sub<-.cubic(M,x,g,H,sigma,ctl);d<-sub$step;pred<-sub$predicted
        } else {
          Hd<-if(method=="levenberg_marquardt") function(u) .add(H(u),u,lambda) else H
          sub<-.tcg(M,x,g,Hd,ctl,if(method=="trust_regions") radius else Inf,precondition=function(u) .precondition(P,x,u));d<-sub$step
          pred<--.ip(M,x,g,d)-.ip(M,x,d,H(d))/2
        }
        if(method %in% c("newton_cg","gauss_newton")) {
          if(.ip(M,x,g,d)>=0) {d<-.scale(g,-1);diagnostics$restarts<-diagnostics$restarts+1L}
          trial<-.linesearch(P,x,f,g,d,ctl,counts)
        } else {
          trial<-.trial(P,x,d,1,counts);trial$step<-1
          ratio<-if(trial$ok&&is.finite(pred)&&pred>0) (f-trial$f)/pred else -Inf
          trial$ok<-trial$ok&&is.finite(ratio)&&ratio>ctl$acceptance_ratio
          trial$rejected<-if(trial$ok) 0L else 1L
          sub$predicted_reduction<-pred;sub$ratio<-ratio
          if(method=="trust_regions") {
            if(ratio<0.25) radius<-radius/4 else if(ratio>0.75&&.nrm(M,x,d)>0.9*radius) radius<-min(2*radius,ctl$max_trust_radius)
          }
          if(method=="levenberg_marquardt") lambda<-if(ratio>0.75) max(lambda/2,1e-14) else if(ratio<0.25) min(lambda*4,1e16) else lambda
          if(method=="adaptive_regularization_cubics") sigma<-if(ratio>0.75) max(sigma/2,1e-14) else if(ratio<0.25) min(sigma*2,1e16) else sigma
        }
        trial$direction<-d
      } else {
        d<-.scale(g,-1)
        if(method=="conjugate_gradient"&&!is.null(direction)) d<-direction
        if(method=="lbfgs"&&length(pairs)) {
          q<-.clone(g);aa<-numeric(length(pairs))
          for(j in rev(seq_along(pairs))) {aa[j]<-.ip(M,x,pairs[[j]]$s,q)/pairs[[j]]$sy;q<-.add(q,pairs[[j]]$y,-aa[j])}
          last<-pairs[[length(pairs)]];r<-.scale(q,last$sy/.ip(M,x,last$y,last$y))
          for(j in seq_along(pairs)) {b<-.ip(M,x,pairs[[j]]$y,r)/pairs[[j]]$sy;r<-.add(r,pairs[[j]]$s,aa[j]-b)}
          d<-.scale(r,-1)
        }
        if(method=="alternating_gradient") {
          block<-blocks[(it-1)%%length(blocks)+1];d<-.zeros(g);d[[block]]<-.scale(g[[block]],-1)
          if(.nrm(M,x,d)<=ctl$gradient_tolerance) {
            trial<-list(ok=TRUE,x=x,f=f,step=0,rejected=0L,direction=d)
          }
        }
        if(method!="alternating_gradient"||.nrm(M,x,d)>ctl$gradient_tolerance) {
          if(.ip(M,x,g,d)>=-1e-12*ng*.nrm(M,x,d)) {d<-.scale(g,-1);pairs<-list();diagnostics$restarts<-diagnostics$restarts+1L}
          trial<-.linesearch(P,x,f,g,d,ctl,counts,alpha)
          trial$direction<-d
        }
      }
      trial
    },error=function(e) e)
    if(inherits(ans,"error")) {termination<-.error_termination(ans);message<-conditionMessage(ans);break}
    rejected<-rejected+ans$rejected
    if(!is.null(sub)) diagnostics$subproblems[[length(diagnostics$subproblems)+1L]]<-sub[setdiff(names(sub),"step")]
    if(!ans$ok) {
      if(method %in% c("trust_regions","levenberg_marquardt","adaptive_regularization_cubics")) {
        history[[length(history)+1L]]<-list(iteration=it,objective=f,gradient_norm=ng,accepted=FALSE,step=0,ratio=ratio)
        if(radius<ctl$min_step||lambda>=1e16||sigma>=1e16) {termination<-"stopped_small_step";break}
        next
      }
      termination<-if(method=="stochastic_gradient") ans$reason else "line_search_failed";message<-ans$message;break
    }
    x<-.clone(ans$x);accepted<-accepted+1L
    if(method=="stochastic_gradient") {
      full_at_current<-FALSE
      stepvec<-.scale(ans$direction,ans$step);stepnorm<-.nrm(M,oldx,stepvec)
      full_now<-ctl$full_evaluation_every>0&&it%%ctl$full_evaluation_every==0
      if(full_now) {
        ev<-tryCatch(.evaluate(P,x,counts,chunk_size=full_chunk),error=function(e) e)
        if(inherits(ev,"error")) {termination<-.error_termination(ev);message<-conditionMessage(ev);f<-NA_real_;g<-NULL;break}
        f<-.scalar(ev$value);g<-ev$gradient
        if(!is.finite(f)) {termination<-"nonfinite_objective";break}
        ng<-.nrm(M,x,g);full_at_current<-TRUE
      } else {f<-NA_real_;g<-NULL;ng<-NA_real_}
      history[[length(history)+1L]]<-list(iteration=it,objective=f,
        gradient_norm=ng,minibatch_objective=ans$batch_objective,
        minibatch_gradient_norm=ans$batch_gradient_norm,accepted=TRUE,
        step=ans$step,evaluation=if(full_now) "full" else "minibatch")
      if(full_now&&!is.finite(ng)) {termination<-"numerical_failure";message<-"Nonfinite metric gradient norm";break}
      if(full_now&&ng<=ctl$gradient_tolerance) {termination<-"converged_gradient";break}
      if(stepnorm<=ctl$step_tolerance) {termination<-"stopped_small_step";break}
      next
    }
    ev<-if(!is.null(ans$evaluation)) ans$evaluation else tryCatch(.evaluate(P,x,counts,chunk_size=full_chunk),error=function(e) e)
    if(inherits(ev,"error")) {termination<-.error_termination(ev);message<-conditionMessage(ev);f<-ans$f;g<-NULL;break}
    f<-.scalar(ev$value);g<-ev$gradient
    if(!is.finite(f)) {termination<-"nonfinite_objective";break}
    ng<-.nrm(M,x,g);stepvec<-.scale(ans$direction,ans$step);stepnorm<-.nrm(M,oldx,stepvec)
    history[[length(history)+1L]]<-list(iteration=it,objective=f,gradient_norm=ng,accepted=TRUE,step=ans$step,ratio=ratio)
    if(!is.finite(ng)) {termination<-"numerical_failure";message<-"Nonfinite metric gradient norm";break}
    if(ng<=ctl$gradient_tolerance) {termination<-"converged_gradient";break}
    if(stepnorm<=ctl$step_tolerance&&method!="alternating_gradient") {termination<-"stopped_small_step";break}
    update<-tryCatch({
      if(method %in% c("conjugate_gradient","lbfgs","barzilai_borwein")) {
        transport<-function(v) .clone(riem.transport(M,oldx,stepvec,x,v))
        tg<-transport(oldg);s<-transport(stepvec);y<-.add(g,tg,-1)
        if(method=="conjugate_gradient") {
          den<-max(.ip(M,oldx,oldg,oldg),1e-30)
          dyden<-.ip(M,x,transport(ans$direction),y)
          beta<-switch(ctl$cg_beta,"PR+"=max(0,.ip(M,x,g,y)/den),
            FR=.ip(M,x,g,g)/den,"HS+"=if(dyden>1e-30) max(0,.ip(M,x,g,y)/dyden) else 0,
            DY=if(dyden>1e-30) .ip(M,x,g,g)/dyden else 0)
          if(it%%ctl$restart_every==0) {beta<-0;diagnostics$restarts<-diagnostics$restarts+1L}
          direction<-.add(.scale(g,-1),transport(ans$direction),beta)
          if(.ip(M,x,g,direction)>=-1e-12*ng*.nrm(M,x,direction)) {direction<-.scale(g,-1);diagnostics$restarts<-diagnostics$restarts+1L}
        } else {
          sy<-.ip(M,x,s,y)
          if(method=="barzilai_borwein") {
            variant<-if(ctl$bb_variant=="alternating") if(it%%2) "BB1" else "BB2" else ctl$bb_variant
            proposal<-if(variant=="BB1") .ip(M,x,s,s)/sy else sy/max(.ip(M,x,y,y),1e-30)
            alpha<-if(sy>1e-12*.nrm(M,x,s)*.nrm(M,x,y)) min(ctl$max_step,max(ctl$min_step,proposal)) else ctl$step_size
          }
          if(method=="lbfgs") {
            pairs<-lapply(pairs,function(pair) {pair$s<-transport(pair$s);pair$y<-transport(pair$y);pair$sy<-.ip(M,x,pair$s,pair$y);pair})
            pairs<-Filter(function(pair) is.finite(pair$sy)&&pair$sy>1e-12*.nrm(M,x,pair$s)*.nrm(M,x,pair$y),pairs)
            if(is.finite(sy)&&sy>1e-12*.nrm(M,x,s)*.nrm(M,x,y)) pairs<-tail(c(pairs,list(list(s=s,y=y,sy=sy))),ctl$memory) else diagnostics$skipped_secants<-diagnostics$skipped_secants+1L
          }
        }
      }
      if(method!="barzilai_borwein") alpha<-if(ctl$line_search=="adaptive_armijo") min(ctl$max_step,ans$step*1.5) else ctl$step_size
      TRUE
    },error=function(e) e)
    if(inherits(update,"error")) {termination<-"numerical_failure";message<-conditionMessage(update);break}
  }
  if(method=="stochastic_gradient"&&!full_at_current&&
     !termination%in%c("domain_error","numerical_failure","nonfinite_objective","evaluation_budget","time_budget","user_stopped")) {
    final<-tryCatch(.evaluate(P,x,counts,chunk_size=full_chunk),error=function(e) e)
    if(inherits(final,"error")) {
      termination<-"numerical_failure";message<-conditionMessage(final);f<-NA_real_;g<-NULL
    } else {
      f<-.scalar(final$value);g<-final$gradient
      if(!is.finite(f)) termination<-"nonfinite_objective" else {
        ng<-.nrm(M,x,g)
        if(!is.finite(ng)) {termination<-"numerical_failure";message<-"Nonfinite metric gradient norm"}
        else if(ng<=ctl$gradient_tolerance) termination<-"converged_gradient"
      }
      if(length(history)) {
        history[[length(history)]]$objective<-f
        history[[length(history)]]$gradient_norm<-if(is.null(g)) NA_real_ else ng
        history[[length(history)]]$final_full_evaluation<-TRUE
      }
    }
  }
  diagnostics$trust_radius<-radius;diagnostics$damping<-lambda;diagnostics$regularization<-sigma
  if(inherits(P,"riem_finitesum")) diagnostics$sampling<-list(
    normalization=P$normalization,policy=P$sampling,batch_size=ctl$batch_size,
    evaluation_batch_size=P$evaluation_batch_size,
    full_evaluation_every=ctl$full_evaluation_every,
    stationarity="full_gradient",intermediate="minibatch")
  .fit(P,x,f,g,method,ctl,counts,iterations,accepted,rejected,termination,history,diagnostics,message)
}
