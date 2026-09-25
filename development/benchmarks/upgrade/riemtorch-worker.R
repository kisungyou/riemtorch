library(torch)
devtools::load_all(quiet=TRUE)
torch_set_num_threads(1)
args<-commandArgs(TRUE);case<-args[1];folder<-args[2];output<-args[3]
read<-function(name) as.matrix(read.table(file.path(folder,paste0(name,".csv")),sep=","))
t<-function(z) torch_tensor(z,dtype=torch_float64())
x<-NULL;method<-"lbfgs";control<-list(max_iterations=500,gradient_tolerance=1e-6)
if(case=="eigen") {
 M<-manifold.sphere(6);A<-t(read("eigen_A"));x<-t(as.numeric(read("eigen_x")))
 P<-riem.problem(M,function(z) -(z*A$matmul(z))$sum()/2);optimum<- -max(eigen(read("eigen_A"),symmetric=TRUE,only.values=TRUE)$values)/2
} else if(case=="procrustes") {
 M<-manifold.stiefel(6,2,"euclidean");A<-t(read("procrustes_A"));x<-t(read("procrustes_x"))
 P<-riem.problem(M,function(z) (z-A)$square()$sum()/2)
 sv<-svd(read("procrustes_A"));optimum<-sum((sv$u%*%base::t(sv$v)-read("procrustes_A"))^2)/2
} else if(case=="completion") {
 M<-manifold.fixedrank(6,5,2,"svd");A<-t(read("completion_A"));mask<-t(read("completion_mask"))
 x<-list(U=t(read("completion_U")),S=torch_diag(t(as.numeric(read("completion_S")))),V=t(read("completion_V")))
 P<-riem.problem(M,function(z) ((riem.materialize(M,z)-A)*mask)$square()$sum()/2);optimum<-0
} else if(case=="spd") {
 M<-manifold.spd(3,"lerm");C<-t(read("spd_C"));x<-t(read("spd_x"))
 P<-riem.problem(M,function(z) (riem.matrix.function(z,"log")-C)$square()$sum()/2);optimum<-0
} else if(case=="robust") {
 M<-manifold.euclidean(2);A<-t(read("robust_A"));b<-t(as.numeric(read("robust_b")));x<-t(as.numeric(read("robust_x")))
 P<-riem.problem.leastsquares(M,function(z) A$matmul(z)-b,loss="huber",loss_scale=.2);method<-"levenberg_marquardt";optimum<-NA_real_
} else if(case=="hyperbolic") {
 M<-manifold.power(manifold.hyperbolic(2),5);A<-t(read("hyperbolic_A"));x<-t(read("hyperbolic_x"))
 P<-riem.problem(M,function(z) riem.sqdist(M,z,A)/2);optimum<-0
} else if(case=="phase") {
 M<-manifold.complexcircle(5);aa<-read("phase_A");xx<-read("phase_x")
 a<-torch_complex(t(aa[,1]),t(aa[,2]));x<-torch_complex(t(xx[,1]),t(xx[,2]))
 P<-riem.problem(M,function(z) -(a$conj()*z)$sum()$abs()$square()/2);optimum<- -25/2
} else if(case=="constrained") {
 M<-manifold.euclidean(3);A<-t(as.numeric(read("constrained_A")));x<-t(as.numeric(read("constrained_x")))
 P<-riem.problem.constrained(M,function(z) (z-A)$square()$sum()/2,equality=function(z) z$sum()$reshape(1)-1,inequality=function(z) -z)
 method<-"augmented_lagrangian";control<-list(max_iterations=40,gradient_tolerance=1e-6,inner_control=list(max_iterations=150))
 optimum<-sum((c(1,0,0)-as.numeric(read("constrained_A")))^2)/2
}
# Warm-up uses the same device and placement, excluded from timing.
invisible(riem.optimize(P,x,method,control=utils::modifyList(control,list(max_iterations=1)),device="cpu"))
file.create(paste0(output,".ready"))
start<-proc.time()[[3]]
fit<-riem.optimize(P,x,method,control=control,device="cpu")
elapsed<-proc.time()[[3]]-start
feasibility<-if(case=="constrained") fit$kkt$feasibility else max(unlist(fit$constraint_residuals))
gradient<-if(case=="constrained") fit$kkt$stationarity else fit$gradient_norm
row<-data.frame(case=case,implementation="riemtorch",version=as.character(packageVersion("riemtorch")),
 algorithm=method,seconds=elapsed,objective=fit$objective,gradient_norm=gradient,feasibility=feasibility,
 optimum=optimum,fn=fit$evaluations$fn,gradient=fit$evaluations$gradient,termination=fit$termination,
 device=fit$execution$device,dtype=fit$execution$dtype,transfer_seconds=0,
 seed=as.integer(sub("seed","",basename(folder))))
write.csv(row,output,row.names=FALSE)
