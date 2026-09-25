# Run from the package root after installation, or with RIEMTORCH_LIB set.
lib <- Sys.getenv("RIEMTORCH_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
library(riemtorch)
library(torch)
torch_set_num_threads(1)
torch_manual_seed(604)
outdir <- if (dir.exists("development/benchmarks")) "development/benchmarks" else "."
make_cases <- function() {
  sphere <- local({
    M <- manifold.sphere(30)
    a <- torch_linspace(1,2,30,dtype=torch_float64())
    list(P=riem.problem(M,function(x)-(x*a)$sum()),x=riem.random(M),
         optimum=-(a*a)$sum()$sqrt()$item())
  })
  frame <- local({
    M <- manifold.stiefel(8,3,"euclidean");A<-torch_randn(c(8,3),dtype=torch_float64())
    list(P=riem.problem(M,function(x)-(x*A)$sum()),x=riem.random(M),
         optimum=-linalg_svdvals(A)$sum()$item())
  })
  spd <- local({
    M<-manifold.spd(4,"lerm");target<-torch_diag(torch_linspace(-.2,.4,4,dtype=torch_float64()))
    list(P=riem.problem(M,function(x)((riem.matrix.function(x,"log")-target)^2)$sum()/2),
         x=torch_eye(4,dtype=torch_float64()),optimum=0)
  })
  list(sphere=sphere,procrustes=frame,spd_log_coordinates=spd)
}
cases<-make_cases();rows<-list();profiles<-list()
for(name in names(cases)) {
  case<-cases[[name]]
  invisible(riem.evaluate(case$P,case$x))
  value_time<-system.time(for(i in 1:20)riem.evaluate(case$P,case$x,FALSE))[["elapsed"]]/20
  gradient_time<-system.time(for(i in 1:20)riem.evaluate(case$P,case$x,TRUE))[["elapsed"]]/20
  profiles[[name]]<-data.frame(problem=name,value_seconds=value_time,value_gradient_seconds=gradient_time)
  for(method in c("steepest_descent","conjugate_gradient","lbfgs")) {
    invisible(riem.optimize(case$P,case$x,method,list(max_iterations=2)))
    elapsed<-system.time(fit<-riem.optimize(case$P,case$x,method,
      list(max_iterations=250,gradient_tolerance=1e-6)))[["elapsed"]]
    residual<-max(unlist(fit$constraint_residuals))
    accuracy<-fit$objective-case$optimum
    rows[[length(rows)+1]]<-data.frame(problem=name,method=method,seconds=elapsed,
      objective_gap=accuracy,gradient_norm=fit$gradient_norm,constraint_residual=residual,
      matched_accuracy=abs(accuracy)<=1e-8&&fit$gradient_norm<=1e-5&&residual<=1e-7,
      iterations=fit$iterations,fn_evaluations=fit$evaluations$fn,
      gradient_evaluations=fit$evaluations$gradient,termination=fit$termination)
  }
}
utils::write.csv(do.call(rbind,rows),file.path(outdir,"results.csv"),row.names=FALSE)
utils::write.csv(do.call(rbind,profiles),file.path(outdir,"kernel-costs.csv"),row.names=FALSE)
writeLines(c(capture.output(sessionInfo()),"CPU only; one torch thread; warmups excluded.",
  "Times are descriptive single-run observations, not stable speed rankings.",
  "R allocator statistics do not measure native torch memory; no memory claim is made."),file.path(outdir,"environment.txt"))
print(do.call(rbind,rows))
