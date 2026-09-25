#' Learning-Rate Schedules
#' @param initial Positive initial rate.
#' @param type `"constant"`, `"polynomial"`, or `"cosine"`.
#' @param decay Nonnegative polynomial decay coefficient.
#' @param power Positive polynomial exponent.
#' @param total_iterations Positive cosine horizon.
#' @param minimum Nonnegative minimum rate, at most initial.
#' @return A function of the zero-based iteration number.
#' @examples
#' schedule <- riem.schedule(0.1, "polynomial", decay = 0.01)
#' schedule(10)
#' @export
riem.schedule <- function(initial,type=c("constant","polynomial","cosine"),
                          decay=0.01,power=0.5,total_iterations=100,minimum=0) {
  .number(initial,"initial");.number(decay,"decay",strict=FALSE);.number(power,"power")
  .number(minimum,"minimum",strict=FALSE);.number(total_iterations,"total_iterations")
  if(minimum>initial) .stop("minimum cannot exceed initial")
  type<-match.arg(type);force(initial);force(decay);force(power);force(total_iterations);force(minimum)
  function(iteration) {
    .number(iteration,"iteration",strict=FALSE)
    switch(type,constant=initial,polynomial=max(minimum,initial/(1+decay*iteration)^power),
      cosine=minimum+(initial-minimum)*(1+cos(pi*min(iteration,total_iterations)/total_iterations))/2)
  }
}
.controls<-function(control) {
  defaults<-list(max_iterations=200L,gradient_tolerance=1e-7,step_tolerance=1e-12,
    step_size=1,line_search="armijo",armijo=1e-4,wolfe=0.9,backtrack=0.5,
    max_linesearch=30L,memory=10L,min_step=1e-14,max_step=100,
    restart_every=50L,trust_radius=1,max_trust_radius=100,acceptance_ratio=0.1,
    inner_iterations=50L,inner_tolerance=0.1,damping=1e-2,regularization=1,
    batch_size=1L,full_evaluation_every=0L,schedule=NULL,history=TRUE,blocks=NULL,
    callback=NULL,max_time=Inf,max_evaluations=Inf,cg_beta="PR+",bb_variant="BB1",
    epoch_length=100L,snapshot_every=100L,
    population=20L,temperature=1,cooling=0.95,proposal_scale=0.2,
    inertia=0.5,cognitive=1,social=1,elite_fraction=0.25)
  if(!is.list(control)|| (length(control)&&is.null(names(control)))) .stop("control must be a named list")
  unknown<-setdiff(names(control),names(defaults));if(length(unknown)) .stop("Unknown control: ",paste(unknown,collapse=", "))
  z<-utils::modifyList(defaults,control,keep.null=TRUE)
  for(k in c("max_iterations","max_linesearch","memory","restart_every","inner_iterations","batch_size","full_evaluation_every","population")) {
    if(k%in%c("max_iterations","full_evaluation_every") && is.numeric(z[[k]]) &&
       length(z[[k]])==1 && !is.na(z[[k]]) && z[[k]]==0) next
    v<-.positive_integer(z[[k]],k);if(length(v)!=1) .stop(k," must be scalar")
  }
  for(k in c("gradient_tolerance","step_tolerance")) .number(z[[k]],k,strict=FALSE)
  for(k in c("step_size","min_step","max_step","trust_radius","max_trust_radius","inner_tolerance","damping","regularization","temperature","proposal_scale")) .number(z[[k]],k)
  for(k in c("armijo","wolfe","backtrack","acceptance_ratio","cooling","elite_fraction")) {
    .number(z[[k]],k);if(z[[k]]>=1) .stop(k," must be smaller than one")
  }
  if(z$wolfe<=z$armijo) .stop("wolfe must exceed armijo")
  for(k in c("inertia","cognitive","social")) .number(z[[k]],k,strict=FALSE)
  if(z$min_step>z$max_step || z$trust_radius>z$max_trust_radius) .stop("Inconsistent step/radius bounds")
  z$cg_beta<-match.arg(z$cg_beta,c("PR+","FR","HS+","DY"))
  z$bb_variant<-match.arg(z$bb_variant,c("BB1","BB2","alternating"))
  for(k in c("epoch_length","snapshot_every")) if(length(.positive_integer(z[[k]],k))!=1) .stop(k," must be scalar")
  if(!is.null(z$callback)&&!is.function(z$callback)) .stop("callback must be a function")
  for(k in c("max_time","max_evaluations")) if(!is.numeric(z[[k]])||length(z[[k]])!=1||is.na(z[[k]])||z[[k]]<=0) .stop("Invalid ",k)
  if(is.finite(z$max_evaluations)&&z$max_evaluations!=floor(z$max_evaluations)) .stop("max_evaluations must be integral")
  z$line_search<-match.arg(z$line_search,c("armijo","adaptive_armijo","strong_wolfe","fixed"))
  if(!is.null(z$schedule)&&!is.function(z$schedule)) .stop("schedule must be a function")
  if(!is.logical(z$history)||length(z$history)!=1||is.na(z$history)) .stop("history must be TRUE or FALSE")
  z
}
.ip<-function(M,x,u,v) .scalar(riem.inner(M,x,u,v))
.nrm<-function(M,x,u) .scalar(riem.norm(M,x,u))
.constraint<-function(M,x) {
  if(inherits(M,"riem_product")) return(Map(.constraint,M$factors,x))
  if(!.allfinite(x) || !all(riem.belongs(M,x))) return(Inf)
  if(!is.null(M$operations$residual)) M$operations$residual(x) else
    if(all(riem.belongs(M,x))) 0 else Inf
}
.fit<-function(P,x,f,g,method,ctl,counts,iterations,accepted,rejected,termination,history,diagnostics=list(),message=NULL) {
  elapsed<-if(is.null(attr(counts,"runtime"))) 0 else proc.time()[[3]]-attr(counts,"runtime")$start
  if(!is.null(ctl$callback)) {
    response<-ctl$callback(list(point=.clone(x),objective=f,gradient=if(is.null(g)) NULL else .clone(g),
      iteration=iterations,evaluations=as.list(counts),elapsed=elapsed,event="final",termination=termination))
    if(!is.null(response)&&(!is.logical(response)||length(response)!=1||is.na(response))) .stop("callback must return TRUE, FALSE, or NULL")
  }
  structure(list(point=.clone(x),objective=f,elapsed=elapsed,gradient_norm=if(is.null(g)) NA_real_ else .nrm(P$manifold,x,g),
    constraint_residuals=.constraint(P$manifold,x),manifold=P$manifold,method=method,
    control=ctl,iterations=iterations,evaluations=as.list(counts),accepted_steps=accepted,
    rejected_steps=rejected,termination=termination,converged=identical(termination,"converged_gradient"),
    history=if(ctl$history) history else NULL,diagnostics=diagnostics,
    execution=P$execution,message=message),class="riem_fit")
}
#' @export
print.riem_fit <- function(x,...) {
  cat("<riem_fit>",x$method,"on",x$manifold$name,"\n",
      "Objective:",format(x$objective,digits=8)," | gradient norm:",format(x$gradient_norm,digits=4),"\n",
      "Termination:",x$termination," | iterations:",x$iterations,"\n")
  invisible(x)
}
