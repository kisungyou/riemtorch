#' Train a Torch Module on One Resolved Device
#'
#' @param model A torch `nn_module` instance.
#' @param dataloader A torch dataloader.
#' @param loss Function `loss(model, batch, ...)` returning one finite scalar
#'   torch tensor.
#' @param optimizer_factory Function called as `optimizer_factory(model)` after
#'   the model has been moved to the selected device and dtype. It must return a
#'   torch optimizer, such as [optim_rsgd()] or [optim_radam()].
#' @param epochs Positive number of complete dataloader passes.
#' @param device Device request. `NULL` follows the package selection policy;
#'   use `"cpu"` to force CPU or, for example, `"cuda:1"` for a specific GPU.
#' @param dtype Optional floating-point dtype request. Existing model precision
#'   is preserved when it is omitted.
#' @param checkpoint Optional checkpoint path to restore after model placement
#'   and optimizer construction. This permits, for example, resuming a GPU
#'   checkpoint with `device = "cpu"`.
#' @param restore_rng Whether to restore RNG state from `checkpoint`.
#' @param ... Additional arguments passed to `loss`.
#' @param validation Optional `function(model, epoch)` returning finite numeric
#'   validation metrics; evaluated in evaluation mode without gradient recording.
#' @param callback Optional `function(state)` called after each epoch; return
#'   TRUE to stop. The state contains detached history, metrics and progress.
#' @param clip_norm Optional positive bound on the total Riemannian gradient norm.
#' @param accumulate Number of minibatches averaged into each optimizer update.
#' @param scheduler Optional `function(optimizer, epoch, metrics)` called after
#'   validation. Caller-owned scheduler state can be included in `data_state`.
#' @param data_state Optional list with `get()` and `set(state)` callbacks for
#'   epoch-boundary data-order or scheduler state. Restoration happens before
#'   the next iterator is created. Pass the returned `training_state` to
#'   [riem.save()] to resume progress together with model, optimizer and RNG state.
#' @return A `riem_training_fit` containing `model`, `optimizer`, per-epoch
#'   `history`, total `steps`, and resolved `execution` metadata.
#' @details Device resolution and model placement happen once, before optimizer
#' construction. Each minibatch is moved immediately before use. Integer index
#' tensors and logical masks retain their dtype. Tensor arguments supplied in
#' `...` are moved once with the model. The mean reported for an epoch is the
#' unweighted mean of its scalar minibatch losses.
#' @examples
#' if (torch::torch_is_installed()) {
#'   x <- torch::torch_tensor(matrix(c(0, 1, 1, 2), ncol = 1),
#'                            dtype = torch::torch_float64())$clone()
#'   y <- (x * 2)$clone()
#'   loader <- torch::dataloader(torch::tensor_dataset(x, y), batch_size = 2)
#'   net_type <- torch::nn_module("tiny_linear",
#'     initialize = function() self$weight <- torch::nn_parameter(
#'       torch::torch_zeros(1, 1, dtype = torch::torch_float64())),
#'     forward = function(z) z$matmul(self$weight))
#'   fit <- riem.train(net_type(), loader,
#'     loss = function(model, batch) (model(batch[[1]]) - batch[[2]])$square()$mean(),
#'     optimizer_factory = function(model) optim_rsgd(model$parameters, lr = 0.1),
#'     epochs = 2, device = "cpu")
#'   fit$history
#' }
#' @export
riem.train <- function(model,dataloader,loss,optimizer_factory,epochs=1L,
                       device=NULL,dtype=NULL,checkpoint=NULL,
                       restore_rng=FALSE,...,validation=NULL,callback=NULL,clip_norm=NULL,
                       accumulate=1L,scheduler=NULL,data_state=NULL) {
  if(!inherits(model,"nn_module")) .stop("model must be a torch nn_module")
  if(!is.function(loss)) .stop("loss must be a function")
  if(!is.function(optimizer_factory)) .stop("optimizer_factory must be a function")
  epochs<-.positive_integer(epochs,"epochs")
  if(length(epochs)!=1) .stop("epochs must be scalar")
  accumulate<-.positive_integer(accumulate,"accumulate")
  if(length(accumulate)!=1) .stop("accumulate must be scalar")
  if(!is.null(clip_norm)) .number(clip_norm,"clip_norm")
  for(fn in list(validation,callback,scheduler)) if(!is.null(fn)&&!is.function(fn)) .stop("Training hooks must be functions")
  if(!is.null(data_state)&&(!is.list(data_state)||!is.function(data_state$get)||!is.function(data_state$set)))
    .stop("data_state must supply get and set functions")
  parameters<-model$parameters
  if(!length(parameters)) .stop("model must have at least one parameter")
  floating<-Filter(function(parameter) (parameter$is_floating_point()||parameter$is_complex()),parameters)
  if(!length(floating)) .stop("model must have at least one floating-point parameter")
  reference<-floating[[1]]
  execution<-.resolve_execution(device=device,dtype=dtype,reference=reference)
  # With no dtype request, moving only the device preserves a module's existing
  # floating-point precisions (including intentionally mixed-precision state).
  if(is.null(dtype)) model$to(device=execution$device) else
    model$to(device=execution$device,dtype=execution$dtype)
  optimizer<-optimizer_factory(model)
  if(!inherits(optimizer,"torch_optimizer"))
    .stop("optimizer_factory must return a torch optimizer")
  resumed<-NULL;start_epoch<-0L;steps<-0L
  if(!is.null(checkpoint)) {
    if(!inherits(optimizer,"riem_optimizer"))
      .stop("checkpoint restoration requires a riemtorch optimizer")
    riem.load(optimizer,checkpoint,device=execution$device,model=model,
              restore_rng=restore_rng)
    resumed<-optimizer$checkpoint$training_state
    if(is.list(resumed)&&length(resumed$epoch)==1&&is.finite(resumed$epoch))
      start_epoch<-as.integer(resumed$epoch)
    if(is.list(resumed)&&length(resumed$step)==1&&is.finite(resumed$step))
      steps<-as.integer(resumed$step)
  }
  if(!is.null(data_state)&&!is.null(resumed$data_order)) data_state$set(resumed$data_order)
  if(inherits(optimizer,"riem_optimizer")&&optimizer$algorithm=="rlinesearch")
    .stop("riem.train requires minibatch optimizers; use optim_rlinesearch directly with a deterministic closure")
  dots<-.to_execution(list(...),execution);history<-vector("list",epochs)
  metrics<-list();termination<-"epochs_completed";training_state<-NULL
  model$train()
  for(epoch in seq_len(epochs)) {
    iterator<-torch::dataloader_make_iter(dataloader)
    total<-0;batches<-0L;pending<-0L
    optimizer$zero_grad()
    update<-function(n) {
      .training_prepare_gradient(optimizer,n,clip_norm)
      optimizer$step();optimizer$zero_grad()
    }
    repeat {
      batch<-torch::dataloader_next(iterator)
      if(is.null(batch)) break
      batch<-riem.to(batch,device=execution$device,dtype=execution$dtype)
      objective<-do.call(loss,c(list(model,batch),dots))
      if(!inherits(objective,"torch_tensor")||objective$numel()!=1||
         !objective$is_floating_point())
        .stop("loss must return a floating-point scalar torch tensor")
      if(!identical(objective$device$type,execution$device$type)||
         !identical(objective$device$index,execution$device$index))
        .stop("loss must remain on the selected execution device")
      if(!.finite(objective)) .stop("loss returned a nonfinite value")
      objective$backward();pending<-pending+1L
      if(pending==accumulate) {update(pending);steps<-steps+1L;pending<-0L}
      total<-total+.scalar(objective$detach());batches<-batches+1L
    }
    if(!batches) .stop("dataloader produced no minibatches")
    if(pending>0) {update(pending);steps<-steps+1L}
    history[[epoch]]<-data.frame(epoch=start_epoch+epoch,loss=total/batches,
                                 batches=batches,steps=steps)
    valid<-NULL
    if(!is.null(validation)) {
      model$eval()
      valid<-tryCatch(torch::with_no_grad(validation(model,start_epoch+epoch)),finally=model$train())
      if(inherits(valid,"torch_tensor")) valid<-as.numeric(valid$detach()$to(device="cpu"))
      if(!is.numeric(valid)||any(!is.finite(valid))) .stop("validation must return finite numeric metrics")
    }
    metrics[epoch]<-list(valid)
    if(!is.null(scheduler)) scheduler(optimizer,start_epoch+epoch,valid)
    training_state<-list(epoch=start_epoch+epoch,step=steps,
      data_order=if(is.null(data_state)) NULL else data_state$get())
    if(!is.null(callback)) {
      stopped<-callback(list(epoch=start_epoch+epoch,steps=steps,history=history[[epoch]],validation=valid,training_state=training_state))
      if(!is.logical(stopped)||length(stopped)!=1||is.na(stopped)) .stop("training callback must return TRUE or FALSE")
      if(stopped) {termination<-"user_stopped";history<-history[seq_len(epoch)];break}
    }
  }
  history<-do.call(rbind,history);row.names(history)<-NULL
  metadata<-.execution_metadata(execution)
  structure(list(model=model,optimizer=optimizer,history=history,steps=steps,
    execution=metadata,resumed=resumed,validation=metrics,
    training_state=training_state,termination=termination),class="riem_training_fit")
}

.training_prepare_gradient <- function(optimizer,n,clip_norm) {
  torch::with_no_grad({
    norm2<-0;gradients<-list()
    for(group in optimizer$param_groups) for(p in group$params) {
      if(!p$requires_grad||is.null(p$grad)||torch::is_undefined_tensor(p$grad)) next
      gradient<-p$grad
      if(gradient$is_sparse()) {gradient<-gradient$coalesce();p$grad<-gradient;gradient$values()$div_(n)} else gradient$div_(n)
      if(!is.null(clip_norm)) {
        M<-if(is.null(group$manifold)) .training_manifold(p) else group$manifold
        weight<-if(is.null(group$metric_weight)) 1 else group$metric_weight
        if(gradient$is_sparse()) {
          base<-.sparse_base(M,p);ids<-gradient$indices()$squeeze(1)
          if(ids$numel()==0) next
          at<-p$index_select(1,ids);M<-manifold.power(base,ids$numel());E<-gradient$values()
        } else {at<-p;E<-gradient}
        g<-riem.egrad2rgrad(M,at,E)/weight
        norm2<-norm2+.scalar(riem.inner(M,at,g,g))*weight
      }
      gradients[[length(gradients)+1L]]<-gradient
    }
    if(!is.null(clip_norm)) {
      if(!is.finite(norm2)||norm2<0) .stop("Invalid metric gradient norm")
      ratio<-min(1,clip_norm/max(sqrt(norm2),.Machine$double.eps))
      for(g in gradients) if(g$is_sparse()) g$values()$mul_(ratio) else g$mul_(ratio)
    }
  })
}
