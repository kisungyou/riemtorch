# Install the current source package, then run this file on the target host.
# Rscript development/qualification/devices.R [output-directory]
library(torch);library(riemtorch)
torch_set_num_threads(1);torch_manual_seed(1601)
args<-commandArgs(TRUE);out<-if(length(args)) args[1] else "development/qualification/results"
dir.create(out,recursive=TRUE,showWarnings=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"session.txt"))
if(nzchar(Sys.which("nvidia-smi")))
 writeLines(system2(Sys.which("nvidia-smi"),c("--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv"),stdout=TRUE),file.path(out,"nvidia.csv"))
inventory<-riem.devices("float64");write.csv(inventory,file.path(out,"inventory.csv"),row.names=FALSE)
requests<-c("auto","cpu")
if(cuda_is_available()) requests<-c(requests,paste0("cuda:",seq_len(cuda_device_count())-1L))
if("mps" %in% riem.devices("float32")$type) requests<-c(requests,"mps")
rows<-list()
for(request in requests) for(precision in c("float64","float32")) {
 if(request=="mps"&&precision=="float64") next
 record<-tryCatch({
  dtype<-if(precision=="float64") torch_float64() else torch_float32()
  x<-torch_tensor(c(1,0,0),dtype=dtype);target<-torch_tensor(c(.3,.8,.4),dtype=dtype)
  P<-riem.problem(manifold.sphere(3),function(z,data) -(z*data)$sum(),data=target,required_operations="basic")
  tol<-if(precision=="float64") 1e-6 else 2e-4
  cpu<-riem.optimize(P,x,device="cpu",control=list(gradient_tolerance=tol))
  resolved<-riem.device(request,precision,operations="basic")
  if(request=="auto"&&cuda_is_available()&&resolved$type!="cuda") stop("Auto did not select available CUDA")
  transfer_start<-proc.time()[[3]];moved<-riem.to(x,device=resolved)
  if(resolved$type=="cuda") cuda_synchronize(resolved)
  transfer<-proc.time()[[3]]-transfer_start
  start<-proc.time()[[3]]
  fit<-riem.optimize(P,moved,device=resolved,control=list(gradient_tolerance=tol))
  if(resolved$type=="cuda") cuda_synchronize(resolved)
  elapsed<-proc.time()[[3]]-start
  point_error<-max(abs(as.numeric(fit$point$to(device="cpu"))-as.numeric(cpu$point)))
  stopifnot(point_error<tol*20,abs(fit$objective-cpu$objective)<tol*20)
  # Explicit CPU overrides package defaults, including a CUDA preference.
  old<-getOption("riemtorch.device");options(riemtorch.device="cuda")
  forced<-tryCatch(riem.device("cpu",precision,operations="basic"),finally=options(riemtorch.device=old))
  stopifnot(forced$type=="cpu")
  p<-nn_parameter(riem.to(x,device=resolved)$detach()$clone());opt<-optim_radam(list(p),manifold=P$manifold,amsgrad=TRUE)
  step<-function(o,z) {o$zero_grad();(-z[2])$backward();o$step()}
  step(opt,p);path<-file.path(out,paste0(gsub(":","-",request),"-",precision,".pt"));riem.save(opt,path)
  q<-nn_parameter(x$detach()$clone());other<-optim_radam(list(q),manifold=P$manifold,amsgrad=TRUE);riem.load(other,path,device="cpu")
  step(opt,p);step(other,q)
  checkpoint_error<-max(abs(as.numeric(p$to(device="cpu"))-as.numeric(q)))
  stopifnot(checkpoint_error<tol*20)
  data.frame(request=request,precision=precision,selected=fit$execution$device,status="verified",
    model=fit$execution$model,libtorch=fit$execution$libtorch,reason=fit$execution$reason,
    point_error=point_error,checkpoint_error=checkpoint_error,transfer_seconds=transfer,seconds=elapsed,error="")
 },error=function(e) data.frame(request=request,precision=precision,selected=NA,status="failed",
    model=NA,libtorch=NA,reason=NA,point_error=NA,checkpoint_error=NA,transfer_seconds=NA,seconds=NA,error=conditionMessage(e)))
 rows[[length(rows)+1L]]<-record
}
results<-do.call(rbind,rows);write.csv(results,file.path(out,"devices.csv"),row.names=FALSE);print(results)
if(any(results$status=="failed")) stop("Device qualification failed; see saved report")
# Absent accelerators are explicitly recorded as unexercised.
writeLines(c(if(!cuda_is_available()) "CUDA: awaiting hardware validation",
 if(!"mps" %in% inventory$type) "MPS: awaiting hardware validation"),file.path(out,"unexercised.txt"))
