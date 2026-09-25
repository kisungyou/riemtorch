.boundary_tau<-function(M,x,s,d,radius) {
  a<-.ip(M,x,d,d);b<-2*.ip(M,x,s,d);c<-.ip(M,x,s,s)-radius^2
  max(0,(-b+sqrt(max(0,b*b-4*a*c)))/(2*a))
}
.tcg<-function(M,x,g,H,ctl,radius=Inf,precondition=function(u) u) {
  s<-.zeros(g);r<-.clone(g);z<-precondition(r);d<-.scale(z,-1);rr<-.ip(M,x,r,z);initial<-.nrm(M,x,r)
  status<-"max_inner_iterations";i<-0L
  if(initial==0) return(list(step=s,status="zero_gradient",iterations=0L))
  for(i in seq_len(ctl$inner_iterations)) {
    Hd<-H(d);curv<-.ip(M,x,d,Hd)
    if(!is.finite(curv)) .stop("Nonfinite subproblem curvature")
    if(curv<=0) {
      if(is.finite(radius)) s<-.add(s,d,.boundary_tau(M,x,s,d,radius)) else if(i==1) s<-d
      status<-"negative_curvature";break
    }
    a<-rr/curv;candidate<-.add(s,d,a)
    if(is.finite(radius)&&.nrm(M,x,candidate)>=radius) {
      s<-.add(s,d,.boundary_tau(M,x,s,d,radius));status<-"boundary";break
    }
    s<-candidate;rnew<-.add(r,Hd,a);znew<-precondition(rnew);rrnew<-.ip(M,x,rnew,znew)
    if(.nrm(M,x,rnew)<=ctl$inner_tolerance*initial) {status<-"residual_tolerance";break}
    d<-.add(.scale(znew,-1),d,rrnew/rr);r<-rnew;z<-znew;rr<-rrnew
  }
  list(step=s,status=status,iterations=i)
}
.cubic<-function(M,x,g,H,sigma,ctl) {
  ng<-.nrm(M,x,g);d<-.scale(g,-1/ng);curv<-.ip(M,x,d,H(d));root<-sqrt(curv^2+4*sigma*ng)
  t<-if(curv>0) 2*ng/(curv+root) else (root-curv)/(2*sigma)
  s<-.scale(d,t)
  model<-function(v) {Hv<-H(v);nv<-.nrm(M,x,v);list(value=.ip(M,x,g,v)+.ip(M,x,v,Hv)/2+sigma*nv^3/3,
    gradient=.add(.add(g,Hv),v,sigma*nv))}
  q<-model(s);status<-"cauchy_decrease";i<-0L
  for(i in seq_len(ctl$inner_iterations)) {
    normg<-.nrm(M,x,q$gradient)
    if(normg<=ctl$inner_tolerance*ng) {status<-"model_gradient_tolerance";break}
    a<-1;found<-FALSE
    for(j in seq_len(20)) {
      trial<-.add(s,q$gradient,-a);qt<-model(trial)
      if(is.finite(qt$value)&&qt$value<=q$value-1e-4*a*normg^2) {found<-TRUE;break}
      a<-a/2
    }
    if(!found) break
    s<-trial;q<-qt
  }
  list(step=s,status=status,iterations=i,predicted=-q$value,model_gradient=.nrm(M,x,q$gradient))
}
