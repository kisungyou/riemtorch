.search_noise<-function(M,x,scale=1) riem.tangent(M,x,.tree_map(x,function(z) torch::torch_randn_like(z)*scale))
.displacement<-function(M,x,y) {
  if(inherits(M,"riem_product")) return(Map(.displacement,M$factors,x,y))
  if(!is.null(M$operations$log)) return(riem.log(M,x,y))
  if(M$name %in% c("grassmann","grassmann.generalized","landmark") && M$specification$embedding %in% "frame" || M$name %in% c("grassmann.generalized","landmark")) {
    cross<-if(M$name=="grassmann.generalized") y$t()$matmul(M$specification$B)$matmul(x) else y$t()$matmul(x)
    s<-torch::linalg_svd(cross);q<-s[[1]]$matmul(s[[3]])
    if(M$name=="landmark"&&!isTRUE(M$specification$reflections)&&.scalar(torch::linalg_det(q))<0) {
      d<-torch::torch_ones(q$shape[1],dtype=q$dtype,device=q$device);d[q$shape[1]]<--1
      q<-s[[1]]$matmul(.diag(d))$matmul(s[[3]])
    }
    y<-y$matmul(q)
  }
  riem.tangent(M,x,.add(y,x,-1))
}
.search_candidate<-function(P,x,v,a,counts) .trial(P,x,v,a,counts)
.search_optimize<-function(P,x,method,ctl,counts) {
  M<-P$manifold
  if(method=="stiefel_annealing"&&M$name!="stiefel") .stop("stiefel_annealing requires a Stiefel manifold")
  if(method=="grassmann_macg"&&(M$name!="grassmann"||M$specification$embedding!="frame")) .stop("grassmann_macg requires Grassmann frames")
  initial<-tryCatch(.scalar(.value(P,x,counts)),error=function(e) e)
  if(inherits(initial,"error")) return(.fit(P,x,NA_real_,NULL,method,ctl,counts,0L,0L,0L,.error_termination(initial),list(),message=conditionMessage(initial)))
  if(!is.finite(initial)) return(.fit(P,x,initial,NULL,method,ctl,counts,0L,0L,0L,"nonfinite_objective",list()))
  best<-x;bestf<-initial;f<-initial;accepted<-rejected<-it<-0L
  termination<-"max_iterations";message<-NULL;history<-list(list(iteration=0L,objective=f));diagnostics<-list(heuristic=TRUE)
  localpoint<-function() {
    for(j in seq_len(ctl$max_linesearch)) {
      v<-.search_noise(M,x,ctl$proposal_scale*ctl$backtrack^(j-1));z<-.search_candidate(P,x,v,1,counts)
      if(z$ok) return(z)
      rejected<<-rejected+1L
    }
    .stop("Could not construct a feasible local search population")
  }
  setup<-tryCatch({
    if(method=="particle_swarm") {
      population<-c(list(list(x=x,f=f,ok=TRUE)),lapply(seq_len(ctl$population-1L),function(i) localpoint()))
      personal<-population;velocity<-lapply(population,function(z) .zeros(z$x))
    }
    if(method=="nelder_mead") {
      # Orthonormal tangent simplex, dimension + 1 vertices.
      basis<-list();population<-list(list(x=x,f=f,ok=TRUE))
      if(M$dimension<1) .stop("Nelder-Mead needs positive intrinsic dimension")
      for(j in seq_len(M$dimension)) {
        found<-FALSE
        for(attempt in seq_len(30)) {
          v<-.search_noise(M,x)
          for(b in basis) v<-.add(v,b,-.ip(M,x,b,v))
          nv<-.nrm(M,x,v)
          if(nv>1e-8) {found<-TRUE;break}
        }
        if(!found) .stop("Could not form an independent tangent simplex")
        v<-.scale(v,1/nv);basis[[j]]<-v;step<-ctl$proposal_scale
        for(attempt in seq_len(ctl$max_linesearch)) {
          z<-.search_candidate(P,x,v,step,counts);if(z$ok) break
          rejected<-rejected+1L;step<-step/2
        }
        if(!z$ok) .stop("Could not retract initial simplex")
        population[[j+1L]]<-z
      }
    }
    if(method=="grassmann_macg") A<-.eye(x,x$shape[1])
    TRUE
  },error=function(e)e)
  if(inherits(setup,"error")) return(.fit(P,best,bestf,NULL,method,ctl,counts,0L,accepted,rejected,"domain_error",history,diagnostics,conditionMessage(setup)))
  update_best<-function(z) {if(z$ok&&z$f<bestf) {best<<-.clone(z$x);bestf<<-z$f}}
  if(exists("population",inherits=FALSE)) for(z in population) update_best(z)
  for(it in seq_len(ctl$max_iterations)) {
    stopped<-.iteration_stop(P,best,bestf,NULL,it-1L,ctl,counts)
    if(!is.null(stopped)) {termination<-stopped;break}
    result<-tryCatch({
      if(method=="stiefel_annealing") {
        ref<-x;p<-x$shape[1];a<-torch::torch_randn(c(p,p),dtype=x$dtype,device=x$device)
        y<-torch::torch_matrix_exp(.skew(a)*ctl$proposal_scale)$matmul(x)
        fy<-.scalar(.value(P,y,counts));temp<-ctl$temperature*ctl$cooling^(it-1)
        take<-is.finite(fy)&&(fy<=f||.scalar(torch::torch_rand(1))<exp(min(0,(f-fy)/max(temp,1e-300))))
        if(take) {x<-y;f<-fy;accepted<-accepted+1L} else rejected<-rejected+1L
        update_best(list(ok=is.finite(fy),x=y,f=fy))
        diagnostics$temperature<-temp
      } else if(method=="particle_swarm") {
        for(j in seq_along(population)) {
          point<-population[[j]]$x
          r1<-.scalar(torch::torch_rand(1));r2<-.scalar(torch::torch_rand(1))
          v<-.add(.scale(velocity[[j]],ctl$inertia),.displacement(M,point,personal[[j]]$x),ctl$cognitive*r1)
          v<-.add(v,.displacement(M,point,best),ctl$social*r2)
          z<-.search_candidate(P,point,v,1,counts)
          if(z$ok) {
            velocity[[j]]<-riem.transport(M,point,v,z$x,v);population[[j]]<-z;accepted<-accepted+1L
            if(z$f<personal[[j]]$f) personal[[j]]<-z
            update_best(z)
          } else {velocity[[j]]<-.zeros(point);rejected<-rejected+1L}
        }
      } else if(method=="nelder_mead") {
        order<-order(vapply(population,`[[`,numeric(1),"f"));population<-population[order];n<-length(population)
        anchor<-population[[1]]$x
        v<-.zeros(anchor)
        for(j in seq_len(n-1)) v<-.add(v,.displacement(M,anchor,population[[j]]$x),1/(n-1))
        centroid<-riem.retr(M,anchor,v);worst<-population[[n]]
        d<-.displacement(M,centroid,worst$x)
        reflected<-.search_candidate(P,centroid,d,-1,counts)
        if(reflected$ok&&reflected$f<population[[1]]$f) {
          expanded<-.search_candidate(P,centroid,d,-2,counts)
          chosen<-if(expanded$ok&&expanded$f<reflected$f) expanded else reflected
        } else if(reflected$ok&&reflected$f<population[[n-1]]$f) chosen<-reflected else {
          outside<-reflected$ok&&reflected$f<worst$f
          contracted<-.search_candidate(P,centroid,d,if(outside) -0.5 else 0.5,counts)
          if(contracted$ok&&contracted$f<(if(outside) reflected$f else worst$f)) chosen<-contracted else chosen<-NULL
        }
        if(!is.null(chosen)) {population[[n]]<-chosen;accepted<-accepted+1L;update_best(chosen)} else {
          for(j in 2:n) {
            z<-.search_candidate(P,anchor,.displacement(M,anchor,population[[j]]$x),0.5,counts)
            if(z$ok) {population[[j]]<-z;accepted<-accepted+1L;update_best(z)} else rejected<-rejected+1L
          }
        }
        diameter<-max(vapply(population,function(z) .nrm(M,best,.displacement(M,best,z$x)),numeric(1)))
        diagnostics$simplex_diameter<-diameter
        if(diameter<ctl$step_tolerance) termination<-"stopped_small_step"
      } else {
        p<-x$shape[1];k<-x$shape[2];root<-riem.matrix.function(A,"sqrt")
        population<-lapply(seq_len(ctl$population),function(j) {
          y<-.polar(root$matmul(torch::torch_randn_like(x)))
          fy<-.scalar(.value(P,y,counts));list(x=y,f=fy,ok=is.finite(fy))
        })
        for(z in population) {if(z$ok) accepted<-accepted+1L else rejected<-rejected+1L;update_best(z)}
        good<-Filter(function(z)z$ok,population)
        if(!length(good)) .stop("All MACG sample objectives are nonfinite")
        good<-good[order(vapply(good,`[[`,numeric(1),"f"))]
        elite<-head(good,max(1,floor(length(good)*ctl$elite_fraction)))
        # Shrinkage-regularized MACG fixed-point fit; scale fixed by trace p.
        for(inner in seq_len(20)) {
          C<-Reduce(`+`,lapply(elite,function(z) {q<-z$x;q$matmul(.solve(q$t()$matmul(.solve(A,q)),q$t()))})) * p/(k*length(elite))
          C<-C+1e-6*.eye(C);C<-C*p/C$trace();res<-.scalar(.fnorm(C-A));A<-C
          if(res<1e-7) break
        }
        diagnostics$macg_fit_residual<-res;diagnostics$macg_shrinkage<-1e-6
      }
      TRUE
    },error=function(e)e)
    if(inherits(result,"error")) {termination<-.error_termination(result);message<-conditionMessage(result);break}
    history[[length(history)+1L]]<-list(iteration=it,objective=bestf)
    if(termination=="stopped_small_step") break
  }
  .fit(P,best,bestf,NULL,method,ctl,counts,it,accepted,rejected,termination,history,diagnostics,message)
}
