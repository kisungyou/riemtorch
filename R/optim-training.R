.geometry_config<-function(M) {
  if(inherits(M,"riem_product")) return(list(name=M$name,metric=M$metric,weights=M$weights,factors=lapply(M$factors,.geometry_config)))
  if(!is.null(M$base)) return(list(name=M$name,base=.geometry_config(M$base),specification=M$specification))
  list(name=M$name,shape=M$shape,dimension=M$dimension,metric=M$metric,specification=M$specification)
}
.same_config<-function(a,b) {
  if(inherits(a,"torch_tensor")||inherits(b,"torch_tensor")) return(inherits(a,"torch_tensor")&&inherits(b,"torch_tensor")&&
    identical(a$shape,b$shape)&&a$dtype==b$dtype&&
    identical(a$device$type,b$device$type)&&identical(a$device$index,b$device$index)&&
    isTRUE(torch::torch_equal(a,b)))
  if(is.list(a)) a<-a[!vapply(a,is.null,logical(1))]
  if(is.list(b)) b<-b[!vapply(b,is.null,logical(1))]
  if(is.list(a)||is.list(b)) return(is.list(a)&&is.list(b)&&identical(names(a),names(b))&&length(a)==length(b)&&all(mapply(.same_config,a,b)))
  identical(a,b)
}
.training_group<-function(group) {
  .number(group$lr,"learning rate");.number(group$metric_weight,"metric_weight")
  if(!is.null(group$weight_decay)&&group$weight_decay!=0) .stop("Use an explicit geometric regularizer; manifold weight_decay is unsupported")
  .number(group$momentum,"momentum",strict=FALSE)
  if(group$momentum>=1) .stop("momentum must be less than one")
  if(length(group$betas)!=2||any(!is.finite(group$betas)|group$betas<0|group$betas>=1)) .stop("betas must be in [0,1)")
  .number(group$eps,"eps")
  if(!is.logical(group$amsgrad)||length(group$amsgrad)!=1||is.na(group$amsgrad)) .stop("amsgrad must be TRUE or FALSE")
  if(group$backtrack<=0||group$backtrack>=1||group$armijo<=0||group$armijo>=1) .stop("Invalid line-search constants")
  .positive_integer(group$max_backtracks,"max_backtracks")
  for(p in group$params) {
    M<-if(is.null(group$manifold)) .training_manifold(p) else group$manifold
    if(inherits(M,"riem_product")) .stop("Each tensor parameter is one factor; use separate parameter groups for products")
    .check_point(M,p)
    if(!all(riem.belongs(M,p))) .stop("A training parameter is outside its manifold")
  }
}
.training_state_dict<-function(opt) {
  groups<-list();states<-list();index<-0L
  for(i in seq_along(opt$param_groups)) {
    group<-opt$param_groups[[i]];configs<-list();keys<-character(length(group$params))
    for(j in seq_along(group$params)) {
      p<-group$params[[j]];index<-index+1L
      M<-if(is.null(group$manifold)) .training_manifold(p) else group$manifold
      key<-if(!is.null(group$keys)) group$keys[j] else paste0("group",i,".parameter",j)
      keys[j]<-key;configs[[j]]<-.geometry_config(M)
      states[index]<-list(opt$state$get(p))
    }
    options<-group[setdiff(names(group),c("params","manifold","keys"))]
    groups[[i]]<-list(keys=keys,geometry=configs,options=options)
  }
  allkeys<-unlist(lapply(groups,`[[`,"keys"))
  if(anyNA(allkeys)||any(!nzchar(allkeys))||anyDuplicated(allkeys)) .stop("Checkpoint parameter keys must be unique and nonempty")
  list(schema_version=2L,algorithm=opt$algorithm,groups=groups,state=states)
}
.validate_training_state<-function(opt,dict,points=NULL) {
  if(!is.list(dict)||length(dict$schema_version)!=1||!(dict$schema_version %in% c(1L,2L))||!identical(dict$algorithm,opt$algorithm)) .stop("Incompatible optimizer checkpoint schema or algorithm")
  current<-.training_state_dict(opt)
  if(length(dict$groups)!=length(current$groups)) .stop("Checkpoint group count differs")
  params<-unlist(lapply(opt$param_groups,`[[`,"params"),recursive=FALSE)
  if(length(dict$state)!=length(params)) .stop("Checkpoint state count differs")
  index<-0L
  for(i in seq_along(current$groups)) {
    a<-current$groups[[i]];b<-dict$groups[[i]]
    geometry_ok<-.same_config(a$geometry,b$geometry)
    if(!geometry_ok&&identical(dict$schema_version,1L)) {
      legacy<-lapply(opt$param_groups[[i]]$params,function(p) {
        M<-opt$param_groups[[i]]$manifold;if(is.null(M)) M<-.training_manifold(p)
        list(name=M$name,shape=M$shape,dimension=M$dimension,metric=M$metric,specification=M$specification)
      })
      geometry_ok<-.same_config(legacy,b$geometry)
    }
    if(!identical(a$keys,b$keys)||!geometry_ok) .stop("Checkpoint keys or geometry/metric configuration differ")
    group<-opt$param_groups[[i]]
    # Validate saved options through the same group contract before mutation.
    proposed<-utils::modifyList(group,b$options);.training_group(proposed)
    for(j in seq_along(group$params)) {
      index<-index+1L;p<-group$params[[j]];M<-if(is.null(group$manifold)) .training_manifold(p) else group$manifold
      at<-if(is.null(points)) p else points[[index]]
      .check_point(M,at)
      if(!(at$dtype==p$dtype)||!identical(at$device$type,p$device$type)||!identical(at$device$index,p$device$index)||!all(riem.belongs(M,at))) .stop("Checkpoint point has incompatible placement or geometry")
      s<-dict$state[[index]]
      if(!is.null(s)) {
        if(!is.list(s)||is.null(s$step)||s$step<1||!is.finite(s$step)) .stop("Invalid optimizer step counter")
        .check_tangent(M,at,s$momentum,"Checkpoint momentum")
        if(!(s$momentum$dtype==p$dtype)||
           !identical(s$momentum$device$type,p$device$type)||
           !identical(s$momentum$device$index,p$device$index))
          .stop("Checkpoint state placement differs")
        if(opt$algorithm %in% c("radam","radagrad")) {
          n<-if(identical(s$layout,"sparse_rows")) p$shape[1] else if(M$name=="power") M$copies else 1L
          if(identical(dict$schema_version,1L)&&!is.null(s$variance)&&s$variance$numel()==1L) n<-1L
          for(key in c("variance",if(isTRUE(proposed$amsgrad)) "max_variance")) {
            v<-s[[key]]
            if(is.null(v)||v$numel()!=n||!(v$dtype==.real_dtype(p$dtype))||
               .device_string(v$device)!=.device_string(p$device)||!.finite(v)||.scalar(v$min())<0)
              .stop("Invalid adaptive variance state")
          }
        }
        if(identical(s$layout,"sparse_rows")) {
          r<-s$row_steps
          if(is.null(r)||r$numel()!=p$shape[1]||!(r$dtype==torch::torch_int64())||
             .device_string(r$device)!=.device_string(p$device)||.scalar(r$min())<0||.scalar(r$max())>s$step)
            .stop("Invalid sparse row clocks")
        }
      }
    }
  }
  invisible(TRUE)
}
.riem_optimizer <- torch::optimizer(
  "riem_optimizer",
  algorithm=NULL,
  initialize=function(params,lr,manifold=NULL,momentum=0,betas=c(0.9,0.999),eps=1e-8,metric_weight=1,algorithm="rsgd",amsgrad=FALSE,backtrack=0.5,armijo=1e-4,max_backtracks=25L) {
    self$algorithm<-algorithm
    defaults<-list(lr=lr,manifold=manifold,momentum=momentum,betas=betas,eps=eps,metric_weight=metric_weight,amsgrad=amsgrad,backtrack=backtrack,armijo=armijo,max_backtracks=max_backtracks)
    super$initialize(params,defaults)
    # Optimizer construction happens after managed module placement. Rebuild
    # device-bound built-in geometry (notably generalized frame manifolds) on
    # the parameter placement before validating the group contract.
    for(i in seq_along(self$param_groups)) {
      group<-self$param_groups[[i]]
      if(!is.null(group$manifold)&&length(group$params)) {
        reference<-group$params[[1]]
        self$param_groups[[i]]$manifold<-.to_execution(group$manifold,
          list(device=reference$device,dtype=reference$dtype))
      }
    }
    lapply(self$param_groups,.training_group)
    seen <- list()
    for (group in self$param_groups) for (p in group$params) {
      if (any(vapply(seen, identical, logical(1), p))) .stop("A parameter cannot belong to multiple groups")
      seen[[length(seen)+1L]] <- p
    }
    .training_state_dict(self)
    invisible(self)
  },
  step=function(closure=NULL) {
    if(self$algorithm=="rlinesearch") return(.training_linesearch(self,closure))
    loss<-if(is.null(closure)) NULL else torch::with_enable_grad(closure())
    torch::with_no_grad({
      staged<-list()
      for(group in self$param_groups) {
        .training_group(group)
        for(p in group$params) {
          if(!p$requires_grad||is.null(p$grad)||torch::is_undefined_tensor(p$grad)) next
          M<-if(is.null(group$manifold)) .training_manifold(p) else group$manifold
          previous<-self$state$get(p)
          candidate<-if(p$grad$is_sparse()) .training_sparse(p,p$grad,M,previous,group,self$algorithm) else {
            if(!is.null(previous)&&identical(previous$layout,"sparse_rows")) .stop("Cannot mix dense and sparse update layouts")
            .training_update(p,p$grad,M,previous,group,self$algorithm)
          }
          y<-candidate$point;state<-candidate$state
          staged[[length(staged)+1L]]<-list(param=p,point=.clone(y),state=state)
        }
      }
      # All candidate points and states passed validation before any copy.
      for(z in staged) {z$param$copy_(z$point);self$state$set(z$param,z$state)}
    })
    invisible(loss)
  },
  state_dict=function() .training_state_dict(self),
  load_state_dict=function(state_dict,...) {
    .validate_training_state(self,state_dict)
    index<-0L
    for(i in seq_along(self$param_groups)) {
      for(p in self$param_groups[[i]]$params) {
        index<-index+1L;s<-state_dict$state[[index]]
        if(!is.null(s)) s<-lapply(s,function(z) if(inherits(z,"torch_tensor")) .clone(z) else z)
        M<-self$param_groups[[i]]$manifold
        if(identical(state_dict$schema_version,1L)&&!is.null(M)&&M$name=="power"&&!is.null(s$variance)&&s$variance$numel()==1L)
          s$variance<-s$variance$expand(c(M$copies))$clone()
        self$state$set(p,s)
      }
      for(k in names(state_dict$groups[[i]]$options)) self$param_groups[[i]][[k]]<-state_dict$groups[[i]]$options[[k]]
    }
    invisible(self)
  }
)
#' Native Torch Riemannian Training Optimizers
#' @param params Tensor, list of tensors, or torch parameter groups. A group can
#'   supply `manifold`, `lr`, `metric_weight`, and unique character `keys`.
#'   A missing manifold means Euclidean geometry with the parameter's shape.
#' @param lr Positive learning rate.
#' @param manifold Default geometry for each tensor parameter.
#' @param momentum RSGD coefficient in [0,1).
#' @param betas RADAM first- and second-moment coefficients in [0,1).
#' @param eps Positive constant outside the bias-corrected variance square root.
#' @param amsgrad Use the running maximum of uncorrected second moments.
#' @param metric_weight Positive factor metric weight, shared by group parameters.
#' @return A native torch optimizer with `step()`, `zero_grad()`, `state_dict()`
#'   and `load_state_dict()`. Parameter identity is preserved by in-place copies.
#' @details RSGD uses d=momentum*m+grad, retracts -lr*d, and transports d to the
#' new point. RADAM uses a transported first moment and one scalar second moment
#' of the weighted squared gradient norm per tensor factor (independent states
#' for factors of a power manifold). Sparse row gradients are coalesced and
#' use row-local bias-correction clocks; untouched rows and states do not change. Both moments have
#' standard bias corrections. This is factorwise Riemannian Adam, not rectified
#' Adam. It does not perform coordinatewise ambient scaling or weight decay.
#' Frozen and missing-gradient parameters are skipped. A failing update validates
#' all parameters before committing any of them. Place tensors on their device
#' before optimizer construction. See [riem.save()] for complete checkpoints.
#' @examples
#' if (torch::torch_is_installed()) {
#'   x <- torch::nn_parameter(torch::torch_tensor(c(1, 0, 0),
#'                            dtype = torch::torch_float64()))
#'   opt <- optim_rsgd(list(list(params = list(x), manifold = manifold.sphere(3))),
#'                     lr = 0.1, momentum = 0.8)
#'   opt$zero_grad()
#'   (-x[2])$backward()
#'   opt$step()
#'   riem.belongs(manifold.sphere(3), x)
#'   adaptive <- optim_radam(list(x), lr = 0.01, manifold = manifold.sphere(3))
#'   adagrad <- optim_radagrad(list(x), lr = 0.01, manifold = manifold.sphere(3))
#' }
#' @export
optim_rsgd <- function(params,lr=0.01,manifold=NULL,momentum=0,metric_weight=1)
  .riem_optimizer(params,lr,manifold,momentum=momentum,metric_weight=metric_weight,algorithm="rsgd")
#' @rdname optim_rsgd
#' @export
optim_radam <- function(params,lr=0.001,manifold=NULL,betas=c(0.9,0.999),eps=1e-8,metric_weight=1,amsgrad=FALSE)
  .riem_optimizer(params,lr,manifold,betas=betas,eps=eps,metric_weight=metric_weight,algorithm="radam",amsgrad=amsgrad)

#' @rdname optim_rsgd
#' @export
optim_radagrad <- function(params,lr=0.01,manifold=NULL,eps=1e-8,metric_weight=1)
  .riem_optimizer(params,lr,manifold,eps=eps,metric_weight=metric_weight,algorithm="radagrad")
#' Armijo Riemannian Training Optimizer
#' @inheritParams optim_rsgd
#' @param backtrack Step reduction factor in (0,1).
#' @param armijo Sufficient decrease coefficient in (0,1).
#' @param max_backtracks Maximum reductions before failure.
#' @return A torch optimizer. `step(closure)` requires a deterministic closure
#'   that clears gradients, computes a real scalar loss, calls backward and
#'   returns the loss. Failed searches restore all parameter values and gradients.
#' @details Closures must not change buffers, random state, or external state.
#'   Each search varies all parameter groups together. Sparse gradients are not
#'   supported by this optimizer. Parameter identity is preserved.
#' @examples
#' if(torch::torch_is_installed()) {
#'   x <- torch::nn_parameter(torch::torch_ones(2,dtype=torch::torch_float64()))
#'   opt <- optim_rlinesearch(list(x))
#'   opt$step(function() {opt$zero_grad();z<-x$square()$sum();z$backward();z})
#' }
#' @export
optim_rlinesearch <- function(params,lr=1,manifold=NULL,metric_weight=1,
                             backtrack=0.5,armijo=1e-4,max_backtracks=25L)
  .riem_optimizer(params,lr,manifold,metric_weight=metric_weight,algorithm="rlinesearch",
    backtrack=backtrack,armijo=armijo,max_backtracks=max_backtracks)
