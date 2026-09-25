library(torch);devtools::load_all(quiet=TRUE);torch_set_num_threads(1)
a<-commandArgs(TRUE);n<-as.integer(a[1]);mode<-a[2];out<-a[3]
M<-manifold.euclidean(256)
P<-riem.problem.finitesum(M,function(x,i) (x-i/n)$square()$mean(),n=n,
 batch_fn=function(x,indices) {
  shift<-torch_tensor(indices/n,dtype=x$dtype,device=x$device)$unsqueeze(2)
  (x$unsqueeze(1)-shift)$square()$mean(dim=2)
 },evaluation_batch_size=64)
x<-torch_ones(256,dtype=torch_float64());u<-torch_ones_like(x)
invisible(riem.evaluate(P,x));gc();file.create(paste0(out,".ready"))
start<-proc.time()[[3]]
value<-switch(mode,
 diagnostic_value=riem.evaluate(P,x,FALSE)$value,
 gradient=riem.evaluate(P,x)$gradient,
 hessian_vector=riem.hessian(P,x,u),
 differentiable_value={z<-x$clone()$requires_grad_(TRUE);riemtorch:::.value(P,z,riemtorch:::.new_counts())})
write.csv(data.frame(n=n,mode=mode,chunk=64,dimension=256,seconds=proc.time()[[3]]-start,
 checksum=as.numeric(value$detach()$sum())),out,row.names=FALSE)
# Keep the result alive until measurements are collected by the parent process.
Sys.sleep(.1)
