"""Independent Pymanopt/SciPy comparisons; no code imported from riemtorch."""
import sys, pathlib, time, csv
import numpy as np
import scipy
from scipy.optimize import minimize, least_squares
import pymanopt
from pymanopt.manifolds import Sphere, Stiefel, FixedRankEmbedded, ComplexCircle
from pymanopt.optimizers import ConjugateGradient
case, folder, output=sys.argv[1:]
folder=pathlib.Path(folder)
read=lambda name: np.loadtxt(folder/(name+'.csv'),delimiter=',')
counts={'fn':0,'gradient':0}
def counted(key,fn):
 def wrapped(*args):
  counts[key]+=1
  return fn(*args)
 return wrapped
optimum=np.nan;feasibility=0.;algorithm='';implementation='';version=''
if case in ['eigen','procrustes','completion','phase']:
 implementation='Pymanopt';version=pymanopt.__version__;algorithm='Riemannian CG PR+ (default retraction)'
 if case=='eigen':
  M=Sphere(6);A=read('eigen_A');x=read('eigen_x');optimum=-np.linalg.eigvalsh(A)[-1]/2
  cost=lambda z: -z@A@z/2
  grad=lambda z: -A@z
 elif case=='procrustes':
  M=Stiefel(6,2,retraction='polar');A=read('procrustes_A');x=read('procrustes_x')
  cost=lambda z: np.sum((z-A)**2)/2
  grad=lambda z: z-A
  U,_,V=np.linalg.svd(A,full_matrices=False);optimum=np.sum((U@V-A)**2)/2
 elif case=='completion':
  M=FixedRankEmbedded(6,5,2);A=read('completion_A');mask=read('completion_mask')
  x=(read('completion_U'),read('completion_S'),read('completion_V').T);optimum=0
  def cost(U,s,V):return np.sum(((U@np.diag(s)@V-A)*mask)**2)/2
  def grad(U,s,V):
   R=(U@np.diag(s)@V-A)*mask
   return (R@V.T*s,np.diag(U.T@R@V.T),s[:,None]*(U.T@R))
 else:
  M=ComplexCircle(5);aa=read('phase_A');xx=read('phase_x');a=aa[:,0]+1j*aa[:,1];x=xx[:,0]+1j*xx[:,1];optimum=-12.5
  cost=lambda z: -abs(np.vdot(a,z))**2/2
  grad=lambda z: -a*np.vdot(a,z)
 # Decorators preserve fixed arity, needed for factor points.
 if case=='completion':
  @pymanopt.function.numpy(M)
  def f(U,s,V):return counted('fn',cost)(U,s,V)
  @pymanopt.function.numpy(M)
  def g(U,s,V):return counted('gradient',grad)(U,s,V)
 else:
  @pymanopt.function.numpy(M)
  def f(z):return counted('fn',cost)(z)
  @pymanopt.function.numpy(M)
  def g(z):return counted('gradient',grad)(z)
 P=pymanopt.Problem(M,cost=f,euclidean_gradient=g)
 ConjugateGradient(beta_rule='PolakRibiere',max_iterations=1,verbosity=0).run(P,initial_point=x)
 counts={'fn':0,'gradient':0};pathlib.Path(output+'.ready').touch();start=time.perf_counter()
 result=ConjugateGradient(beta_rule='PolakRibiere',max_iterations=500,min_gradient_norm=1e-6,verbosity=0).run(P,initial_point=x)
 elapsed=time.perf_counter()-start;value=result.cost;gradient=result.gradient_norm;termination=result.stopping_criterion
 if case=='procrustes':feasibility=float(np.linalg.norm(result.point.T@result.point-np.eye(2)))
elif case=='robust':
 implementation='SciPy';version=scipy.__version__;algorithm='trust-region reflective robust least squares'
 A=read('robust_A');b=read('robust_b');x=read('robust_x')
 fun=counted('fn',lambda z:A@z-b);jac=counted('gradient',lambda z:A.copy())
 pathlib.Path(output+'.ready').touch();start=time.perf_counter()
 result=least_squares(fun,x,jac=jac,loss='huber',f_scale=.2,gtol=1e-9,ftol=1e-12,xtol=1e-12,max_nfev=500)
 elapsed=time.perf_counter()-start;value=result.cost;gradient=float(np.linalg.norm(result.grad));termination=result.message
elif case=='spd':
 implementation='SciPy';version=scipy.__version__;algorithm='BFGS in log-Euclidean coordinates'
 C=read('spd_C');x=np.zeros(9);optimum=0
 cost=counted('fn',lambda z:np.sum((z.reshape(3,3)-C)**2)/2)
 grad=counted('gradient',lambda z:(z.reshape(3,3)-C).ravel())
 pathlib.Path(output+'.ready').touch();start=time.perf_counter()
 result=minimize(cost,x,jac=grad,method='BFGS',options={'gtol':1e-6,'maxiter':500})
 elapsed=time.perf_counter()-start;value=result.fun;gradient=float(np.linalg.norm(result.jac));termination=result.message
elif case=='hyperbolic':
 implementation='SciPy';version=scipy.__version__;algorithm='BFGS in global Poincare coordinates; independent distance formula'
 A=read('hyperbolic_A');X=read('hyperbolic_x');r=np.linalg.norm(X,axis=1,keepdims=True);x=(X*np.arctanh(r)/r).ravel();optimum=0
 def to_ball(z):
  Z=z.reshape(5,2);r=np.linalg.norm(Z,axis=1,keepdims=True)
  return Z*np.divide(np.tanh(r),r,out=np.ones_like(r),where=r>0)
 def cost(z):
  Q=to_ball(z);den=(1-np.sum(Q*Q,axis=1))*(1-np.sum(A*A,axis=1))
  return np.sum(np.arccosh(1+2*np.sum((Q-A)**2,axis=1)/den)**2)/2
 fun=counted('fn',cost)
 pathlib.Path(output+'.ready').touch();start=time.perf_counter()
 result=minimize(fun,x,method='BFGS',options={'gtol':1e-6,'maxiter':500})
 elapsed=time.perf_counter()-start;value=result.fun;termination=result.message
 # Metric norm from independent real-coordinate central differences.
 Q=to_ball(result.x);ambient=np.empty_like(Q);h=1e-6
 def loss_point(Q):
  den=(1-np.sum(Q*Q,axis=1))*(1-np.sum(A*A,axis=1))
  return np.sum(np.arccosh(1+2*np.sum((Q-A)**2,axis=1)/den)**2)/2
 for i in range(5):
  for j in range(2):
   plus=Q.copy();minus=Q.copy();plus[i,j]+=h;minus[i,j]-=h
   ambient[i,j]=(loss_point(plus)-loss_point(minus))/(2*h)
 gradient=float(np.sqrt(np.sum(ambient**2*(1-np.sum(Q*Q,axis=1,keepdims=True))**2/4)))
else:
 implementation='SciPy';version=scipy.__version__;algorithm='SLSQP with analytic derivatives'
 A=read('constrained_A');x=read('constrained_x');optimum=np.sum((np.array([1,0,0])-A)**2)/2
 fun=counted('fn',lambda z:np.sum((z-A)**2)/2);jac=counted('gradient',lambda z:z-A)
 pathlib.Path(output+'.ready').touch();start=time.perf_counter()
 result=minimize(fun,x,jac=jac,method='SLSQP',bounds=[(0,None)]*3,
  constraints={'type':'eq','fun':lambda z:z.sum()-1,'jac':lambda z:np.ones(3)},options={'ftol':1e-12,'maxiter':500})
 elapsed=time.perf_counter()-start;value=result.fun;termination=result.message
 z=result.x;active=z<1e-7;lam=float(np.mean((A-z)[~active]));mu=np.maximum(z-A+lam,0)*active
 gradient=float(np.linalg.norm(z-A+lam-mu));feasibility=max(abs(z.sum()-1),max(0,-z.min()))
row=dict(case=case,implementation=implementation,version=version,algorithm=algorithm,seconds=elapsed,
 objective=value,gradient_norm=gradient,feasibility=feasibility,optimum=optimum,fn=counts['fn'],gradient=counts['gradient'],
 termination=termination,device='cpu',dtype='complex128' if case=='phase' else 'float64',transfer_seconds=0,seed=int(folder.name[4:]))
with open(output,'w') as f:
 writer=csv.DictWriter(f,fieldnames=row.keys());writer.writeheader();writer.writerow(row)
