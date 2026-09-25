.training_manifold <- function(p) manifold.euclidean(p$shape,if(p$is_complex()) "complex" else "real")
.factor_norm2 <- function(M,p,g) {
  if(M$name=="power") riem.inner(M$base,p,g,g) else riem.inner(M,p,g,g)
}
.factor_expand <- function(M,a) {
  if(M$name=="power"&&inherits(a,"torch_tensor")&&a$numel()>1)
    a$reshape(c(M$copies,rep(1L,length(M$base$shape)))) else a
}
.training_update <- function(p,gradient,M,previous,group,algorithm,clock=NULL) {
  if(!.finite(gradient)) .stop("Nonfinite gradient; no parameters were updated")
  g<-riem.egrad2rgrad(M,p,gradient)/group$metric_weight
  .check_tangent(M,p,g,"Training gradient")
  t<-if(is.null(previous)) 1L else previous$step+1L
  if(is.null(clock)) clock<-t
  old<-if(is.null(previous)) torch::torch_zeros_like(p) else previous$momentum
  .check_tangent(M,p,old,"Stored momentum")
  if(algorithm=="rsgd") {m<-old*group$momentum+g;d<-m;v<-NULL;maximum<-NULL} else {
    n2<-.factor_norm2(M,p,g)*group$metric_weight
    v0<-if(is.null(previous)) torch::torch_zeros_like(n2) else previous$variance
    maximum<-NULL
    if(algorithm=="radagrad") {m<-g;v<-v0+n2;d<-m/.factor_expand(M,v$sqrt()+group$eps)} else {
      b1<-group$betas[1];b2<-group$betas[2]
      m<-old*b1+g*(1-b1);v<-v0*b2+n2*(1-b2)
      denominator<-v
      if(isTRUE(group$amsgrad)) {
        maximum<-torch::torch_maximum(if(is.null(previous)) torch::torch_zeros_like(v) else previous$max_variance,v)
        denominator<-maximum
      }
      correction<-function(beta) if(inherits(clock,"torch_tensor")) -(clock$to(dtype=n2$dtype)*log(beta))$exp()+1 else 1-beta^clock
      c1<-if(b1==0) 1 else correction(b1);c2<-if(b2==0) 1 else correction(b2)
      d<-(m/.factor_expand(M,c1))/.factor_expand(M,(denominator/c2)$sqrt()+group$eps)
    }
  }
  eta<-d*(-group$lr);y<-riem.retr(M,p,eta)
  if(!.finite(y)||!all(riem.belongs(M,y))) .stop("Training step leaves the manifold; no parameters were updated")
  transported<-riem.transport(M,p,eta,y,m)
  .check_tangent(M,y,transported,"Transported momentum")
  state<-list(step=t,momentum=.clone(transported))
  if(!is.null(v)) state$variance<-.clone(v)
  if(!is.null(maximum)) state$max_variance<-.clone(maximum)
  list(point=.clone(y),state=state)
}
.sparse_base <- function(M,p) {
  if(length(p$shape)!=2) .stop("Sparse updates require a row-wise matrix parameter")
  base<-if(M$name=="power") M$base else if(M$name=="euclidean") manifold.euclidean(p$shape[2]) else NULL
  if(is.null(base)||!base$name %in% c("euclidean","sphere","hyperbolic")||length(base$shape)!=1)
    .stop("Sparse updates support row-wise real Euclidean, sphere and hyperbolic embeddings")
  base
}
.training_sparse <- function(p,gradient,M,previous,group,algorithm) {
  base<-.sparse_base(M,p)
  g<-gradient$coalesce();ids<-g$indices();values<-g$values()
  if(ids$shape[1]!=1||length(values$shape)!=2) .stop("Sparse gradients must have one sparse row axis")
  ids<-ids$squeeze(1);n<-ids$numel()
  if(n==0) return(list(point=.clone(p),state=previous))
  if(!is.null(previous)&&!identical(previous$layout,"sparse_rows")) .stop("Cannot mix dense and sparse update layouts")
  at<-p$index_select(1,ids);subM<-manifold.power(base,n)
  state<-if(is.null(previous)) list(step=0L,layout="sparse_rows",momentum=torch::torch_zeros_like(p),
    row_steps=torch::torch_zeros(p$shape[1],dtype=torch::torch_int64(),device=p$device)) else
      lapply(previous,function(x) if(inherits(x,"torch_tensor")) .clone(x) else x)
  if(algorithm %in% c("radam","radagrad")&&is.null(state$variance))
    state$variance<-torch::torch_zeros(p$shape[1],dtype=.real_dtype(p$dtype),device=p$device)
  if(isTRUE(group$amsgrad)&&is.null(state$max_variance)) state$max_variance<-torch::torch_zeros_like(state$variance)
  clocks<-state$row_steps$index_select(1,ids)+1L
  sub<-list(step=state$step,momentum=state$momentum$index_select(1,ids))
  for(key in intersect(c("variance","max_variance"),names(state))) sub[[key]]<-state[[key]]$index_select(1,ids)
  candidate<-.training_update(at,values,subM,sub,group,algorithm,clocks)
  y<-.clone(p);y$index_copy_(1,ids,candidate$point)
  for(key in intersect(c("momentum","variance","max_variance"),names(candidate$state)))
    state[[key]]$index_copy_(1,ids,candidate$state[[key]])
  state$row_steps$index_copy_(1,ids,clocks);state$step<-state$step+1L
  list(point=y,state=state)
}
.training_linesearch <- function(opt,closure) {
  if(!is.function(closure)) .stop("Line-search optimizer requires a closure")
  parameters<-unlist(lapply(opt$param_groups,`[[`,"params"),recursive=FALSE)
  saved<-lapply(parameters,.clone)
  gradients<-lapply(parameters,function(p) if(is.null(p$grad)||torch::is_undefined_tensor(p$grad)) NULL else .clone(p$grad))
  committed<-FALSE
  on.exit(if(!committed) torch::with_no_grad(for(i in seq_along(parameters)) {
    parameters[[i]]$copy_(saved[[i]]);parameters[[i]]$grad<-gradients[[i]]
  }),add=TRUE)
  initial<-torch::with_enable_grad(closure())
  .loss_check(initial,parameters[[1]])
  if(!.finite(initial)) .stop("Nonfinite line-search objective")
  candidates<-list();slope<-0;index<-0L
  for(group in opt$param_groups) {
    .training_group(group)
    for(p in group$params) {
      index<-index+1L
      if(!p$requires_grad||is.null(p$grad)||torch::is_undefined_tensor(p$grad)) next
      if(p$grad$is_sparse()) .stop("Line-search optimizer requires dense gradients")
      M<-if(is.null(group$manifold)) .training_manifold(p) else group$manifold
      if(!.finite(p$grad)) .stop("Nonfinite line-search gradient")
      g<-riem.egrad2rgrad(M,p,p$grad)/group$metric_weight
      eta<-g*(-group$lr)
      slope<-slope+.scalar(riem.inner(M,p,g,eta))*group$metric_weight
      candidates[[length(candidates)+1L]]<-list(p=p,x=saved[[index]],M=M,eta=.clone(eta),g=.clone(g))
    }
  }
  if(!length(candidates)) {committed<-TRUE;return(invisible(initial))}
  controls<-opt$param_groups[[1]];alpha<-1;accepted<-FALSE
  for(k in 0:controls$max_backtracks) {
    trial<-tryCatch({
      torch::with_no_grad(for(z in candidates) {
        y<-riem.retr(z$M,z$x,z$eta*alpha)
        if(!all(riem.belongs(z$M,y))) .stop("Invalid line-search trial")
        z$p$copy_(y)
      })
      value<-torch::with_enable_grad(closure());.loss_check(value,parameters[[1]]);value
    },error=function(e) {stop(e)})
    if(.finite(trial)&&.scalar(trial)<=.scalar(initial)+controls$armijo*alpha*slope) {accepted<-TRUE;break}
    alpha<-alpha*controls$backtrack
  }
  if(!accepted) .stop("Line search failed; parameters restored")
  staged<-lapply(candidates,function(z) {
    m<-riem.transport(z$M,z$x,z$eta*alpha,z$p,z$g);.check_tangent(z$M,z$p,m,"Line-search momentum")
    old<-opt$state$get(z$p)
    list(p=z$p,state=list(step=if(is.null(old)) 1L else old$step+1L,momentum=.clone(m),step_size=alpha))
  })
  for(z in staged) opt$state$set(z$p,z$state)
  committed<-TRUE;invisible(trial)
}
