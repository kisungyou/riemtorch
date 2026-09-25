.trial<-function(P,x,d,a,counts) {
  tryCatch({
    y<-torch::with_no_grad(riem.retr(P$manifold,x,d,a))
    if(!.allfinite(y)||!all(riem.belongs(P$manifold,y))) return(list(ok=FALSE,reason="domain_error"))
    ev<-if(!is.null(P$value_rgrad)) .evaluate(P,y,counts) else NULL
    f<-if(is.null(ev)) torch::with_no_grad(.scalar(.value(P,y,counts))) else .scalar(ev$value)
    list(ok=is.finite(f),x=y,f=f,evaluation=ev,reason=if(is.finite(f)) NULL else "nonfinite_objective")
  },error=function(e) {if(inherits(e,"riem_stop")) stop(e);list(ok=FALSE,reason="domain_error",message=conditionMessage(e))})
}
.curve_slope<-function(P,x,d,a,counts) {
  torch::with_enable_grad({
    ref<-.leaves(x)[[1]];t<-torch::torch_tensor(a,dtype=.real_dtype(ref$dtype),device=ref$device,requires_grad=TRUE)
    y<-riem.retr(P$manifold,x,d,t);v<-.value(P,y,counts)
    .count(counts,"gradient")
    if(!v$requires_grad) return(0)
    .scalar(torch::autograd_grad(v,t)[[1]])
  })
}
.linesearch<-function(P,x,f,g,d,ctl,counts,initial=ctl$step_size) {
  M<-P$manifold;slope<-.ip(M,x,g,d);a<-min(ctl$max_step,max(ctl$min_step,initial));rejected<-0L;last<-NULL
  if(!is.finite(slope)||slope>=0) return(list(ok=FALSE,rejected=0L,reason="non_descent_direction"))
  if(ctl$line_search=="strong_wolfe") {
    if(!isTRUE(M$capabilities$curve_derivative)) .stop("Strong Wolfe requires a differentiable retraction on this geometry")
    if(!is.null(P$egrad)||!is.null(P$rgrad)) .stop("Strong Wolfe currently requires an autodifferentiable objective")
    low<-0;high<-Inf;previous_f<-f
    for(i in seq_len(ctl$max_linesearch)) {
      trial<-.trial(P,x,d,a,counts);last<-trial
      armijo<-trial$ok && trial$f<=f+ctl$armijo*a*slope
      if(!armijo || (low>0 && trial$f>=previous_f)) high<-a else {
        ds<-tryCatch(.curve_slope(P,x,d,a,counts),error=function(e) {if(inherits(e,"riem_stop")) stop(e);NA_real_})
        if(is.finite(ds)&&abs(ds)<=-ctl$wolfe*slope) return(c(trial,list(step=a,rejected=rejected)))
        if(!is.finite(ds)||ds>=0) high<-a else {low<-a;previous_f<-trial$f}
      }
      rejected<-rejected+1L
      a<-if(is.finite(high)) (low+high)/2 else min(2*a,ctl$max_step)
      if(a<ctl$min_step || (is.finite(high)&&high-low<ctl$min_step)) break
    }
  } else {
    for(i in seq_len(ctl$max_linesearch)) {
      trial<-.trial(P,x,d,a,counts);last<-trial
      if(trial$ok&&(ctl$line_search=="fixed"||trial$f<=f+ctl$armijo*a*slope)) return(c(trial,list(step=a,rejected=rejected)))
      rejected<-rejected+1L;a<-a*ctl$backtrack
      if(a<ctl$min_step||ctl$line_search=="fixed") break
    }
  }
  list(ok=FALSE,rejected=rejected,reason=if(is.null(last$reason)) "insufficient_decrease" else last$reason,message=last$message)
}
