#' Save and Restore Training Parameters and Optimizer State
#' @param optimizer An optimizer returned by [optim_rsgd()] or [optim_radam()].
#' @param path Checkpoint file path.
#' @param model Optional torch module whose parameters and buffers are included.
#' @param training_state Optional named list of progress and caller-owned state,
#'   such as epoch, global step, or data-order metadata.
#' @param include_rng Whether to save R, torch CPU and visible CUDA RNG states.
#' @param device Device on which to load tensor storage. `NULL` uses the existing
#'   optimizer parameter device.
#' @param restore_rng Whether to restore saved RNG states.
#' @return `riem.save()` invisibly returns the path. `riem.load()` invisibly
#'   returns the optimizer after restoring its existing parameters and states.
#' @details Checkpoints use torch-aware serialization and contain schema version,
#' named parameter mapping, parameter tensors, geometry/metric configurations,
#' factor structure, optimizer options and tensor states. Version 2 optionally
#' includes complete module state, training progress and RNG state. Loading
#' validates points, momentum and module state before updating parameters. Create
#' a matching module and optimizer on the target device first. Parameter keys
#' default to stable group/position keys; explicit keys detect reordering.
#'
#' Deterministic continuation assumes identical objectives and minibatch order.
#' Dataset position is caller-owned and can be placed in `training_state`.
#' Reproducible continuation requires matching runtimes and data order; results
#' across device types need not be bitwise identical. Custom geometries must be
#' reconstructed by the caller before loading; callbacks are not deserialized.
#' @examples
#' if (torch::torch_is_installed()) {
#'   x <- torch::nn_parameter(torch::torch_tensor(c(1, 0),
#'                           dtype = torch::torch_float64()))
#'   opt <- optim_rsgd(list(x), manifold = manifold.sphere(2))
#'   path <- tempfile(fileext = ".pt")
#'   riem.save(opt, path)
#'   riem.load(opt, path)
#'   unlink(path)
#' }
#' @export
riem.save <- function(optimizer,path,model=NULL,training_state=NULL,include_rng=TRUE) {
  if(!inherits(optimizer,"riem_optimizer")) .stop("Expected a riemtorch optimizer")
  if(!is.null(model)&&!inherits(model,"nn_module")) .stop("model must be a torch nn_module")
  if(!is.null(training_state)&&!is.list(training_state)) .stop("training_state must be a list")
  if(!is.logical(include_rng)||length(include_rng)!=1||is.na(include_rng)) .stop("include_rng must be TRUE or FALSE")
  dict<-optimizer$state_dict()
  params<-unlist(lapply(optimizer$param_groups,`[[`,"params"),recursive=FALSE)
  points<-lapply(params,.clone)
  names(points)<-unlist(lapply(dict$groups,`[[`,"keys"))
  model_state<-if(is.null(model)) NULL else lapply(model$state_dict(),.clone)
  rng<-NULL
  if(include_rng) {
    r_state<-if(exists(".Random.seed",envir=.GlobalEnv,inherits=FALSE))
      get(".Random.seed",envir=.GlobalEnv,inherits=FALSE) else NULL
    cpu_state<-as.integer(torch::torch_get_rng_state())
    cuda_state<-list()
    if(isTRUE(torch::cuda_is_available())) {
      count<-torch::cuda_device_count()
      if(count>0) for(i in seq_len(count))
        cuda_state[[i]]<-as.integer(torch::cuda_get_rng_state(i-1L))
    }
    rng<-list(R=r_state,torch_cpu=cpu_state,cuda=cuda_state)
  }
  torch::torch_save(list(schema_version=2L,parameters=points,optimizer=dict,
    model=model_state,training_state=training_state,rng=rng),path)
  invisible(path)
}
#' @rdname riem.save
#' @export
riem.load <- function(optimizer,path,device=NULL,model=NULL,restore_rng=FALSE) {
  if(!inherits(optimizer,"riem_optimizer")) .stop("Expected a riemtorch optimizer")
  if(!is.logical(restore_rng)||length(restore_rng)!=1||is.na(restore_rng)) .stop("restore_rng must be TRUE or FALSE")
  params<-unlist(lapply(optimizer$param_groups,`[[`,"params"),recursive=FALSE)
  if(is.null(device)) device<-params[[1]]$device
  saved<-torch::torch_load(path,device=device)
  if(!is.list(saved)||is.null(saved$schema_version)||
     !(saved$schema_version%in%c(1L,2L))) .stop("Unsupported checkpoint schema")
  keys<-unlist(lapply(optimizer$state_dict()$groups,`[[`,"keys"))
  if(!identical(names(saved$parameters),keys)) .stop("Checkpoint parameter mapping differs")
  .validate_training_state(optimizer,saved$optimizer,saved$parameters)
  if(identical(saved$schema_version,2L)&&!is.null(saved$model)) {
    if(is.null(model)||!inherits(model,"nn_module")) .stop("Checkpoint contains module state; supply the matching model")
    current<-model$state_dict()
    if(!identical(names(current),names(saved$model))||length(current)!=length(saved$model))
      .stop("Checkpoint module mapping differs")
    for(i in seq_along(current)) {
      a<-current[[i]];b<-saved$model[[i]]
      if(!inherits(a,"torch_tensor")||!inherits(b,"torch_tensor")||
         !identical(a$shape,b$shape)||!(a$dtype==b$dtype)||
         !identical(a$device$type,b$device$type)||
         !identical(a$device$index,b$device$index))
        .stop("Checkpoint module state has incompatible shape, dtype or device")
    }
  }
  if(identical(saved$schema_version,2L)&&!is.null(saved$model)) model$load_state_dict(saved$model)
  torch::with_no_grad(for(i in seq_along(params)) params[[i]]$copy_(saved$parameters[[i]]))
  optimizer$load_state_dict(saved$optimizer)
  restored<-FALSE
  if(identical(saved$schema_version,2L)&&restore_rng&&!is.null(saved$rng)) {
    if(!is.null(saved$rng$R)) assign(".Random.seed",saved$rng$R,envir=.GlobalEnv)
    if(!is.null(saved$rng$torch_cpu)) torch::torch_set_rng_state(
      torch::torch_tensor(saved$rng$torch_cpu,dtype=torch::torch_uint8(),device="cpu"))
    if(length(saved$rng$cuda)&&isTRUE(torch::cuda_is_available())) {
      count<-min(length(saved$rng$cuda),torch::cuda_device_count())
      if(count>0) for(i in seq_len(count)) torch::cuda_set_rng_state(
        torch::torch_tensor(saved$rng$cuda[[i]],dtype=torch::torch_uint8(),device="cpu"),i-1L)
    }
    restored<-TRUE
  }
  optimizer$checkpoint<-list(schema_version=saved$schema_version,
    training_state=if(identical(saved$schema_version,2L)) saved$training_state else NULL,
    rng_restored=restored)
  invisible(optimizer)
}
#' Inspect Published Geometry and Solver Evidence
#' @param ledger `"geometry"`, `"solvers"`, `"support"`, `"devices"`, or `"competitors"`.
#' @return A data frame from the installed coverage ledger. Status distinguishes
#'   supported, experimental, planned and unsupported capabilities.
#' @examples
#' head(riem.support("geometry"))
#' @export
riem.support <- function(ledger=c("geometry","solvers","support","devices","competitors")) {
  ledger<-match.arg(ledger)
  path<-system.file("coverage",paste0(ledger,".csv"),package="riemtorch")
  if(!nzchar(path)) .stop("Coverage ledger is missing from the installation")
  utils::read.csv(path,stringsAsFactors=FALSE)
}
