"""Generate the installed inventories; support records cite executed test cases."""
from pathlib import Path
import csv
root = Path(__file__).resolve().parents[1]
rows = [
('G01','Euclidean','manifold.euclidean','euclidean','tensor','euclidean'),
('G02','Sphere','manifold.sphere','round','unit vector','sphere'),
('G03','SphereExtrinsic','manifold.sphere','round','identity embedding; alias of G02','sphere'),
('G04','Oblique','manifold.oblique','column_round','p x k unit columns','oblique'),
('G05','ProbabilitySimplex','manifold.multinomial','fisher_rao','positive probability vector','multinomial'),
('G06','PoincareBall','manifold.hyperbolic','poincare; negative curvature','d-vector inside ball','poincare'),
('G07','Hyperboloid','manifold.hyperbolic','lorentz; negative curvature','d+1 vector positive time','hyperboloid'),
('G08','Torus','manifold.torus','flat_angles','d angles modulo 2pi','torus'),
('G09','Stiefel','manifold.stiefel','canonical','p x k frame','stiefel_canonical'),
('G10','StiefelEuclidean','manifold.stiefel','euclidean','p x k frame','stiefel_euclidean'),
('G11','Grassmann','manifold.grassmann','canonical','p x k quotient frame','grassmann'),
('G12','GrassmannProjection','manifold.grassmann','projection_half_frobenius','p x p projector; equivalent to G11','grassmann_projection'),
('G13','GeneralizedStiefel','manifold.stiefel.generalized','B_euclidean','p x k; X transpose B X = I','generalized_stiefel'),
('G14','GeneralizedGrassmann','manifold.grassmann.generalized','B_euclidean','p x k quotient; fixed SPD B','generalized_grassmann'),
('G15','SpecialOrthogonal','manifold.rotation','frobenius','p x p det +1','rotation'),
('G16','SpecialEuclidean','manifold.rigidmotion','weighted_product','named rotation and translation','rigidmotion'),
('G17','SPDLogEuclidean','manifold.spd','lerm','p x p symmetric SPD','spd_lerm'),
('G18','SPDAffineInvariant','manifold.spd','airm','p x p symmetric SPD','spd_airm'),
('G19','SPDBuresWasserstein','manifold.spd','wasserstein','p x p symmetric SPD','spd_wasserstein'),
('G20','FixedRank','manifold.fixedrank','embedded','m x p rank k','fixedrank'),
('G21','RankKPSD','manifold.spdk','embedded','p x p rank k PSD','spdk_embedded'),
('G22','RankKPSDBuresWasserstein','manifold.spdk','wasserstein','p x p rank k PSD','spdk_wasserstein'),
('G23','Elliptope','manifold.elliptope','embedded','p x p rank k PSD; unit diagonal','elliptope'),
('G24','Spectrahedron','manifold.spectrahedron','embedded','p x p rank k PSD; unit trace','spectrahedron'),
('G25','CorrelationECM','manifold.correlation','ecm','full rank correlation; unit-lower Cholesky chart','correlation_ecm'),
('G26','CorrelationLEC','manifold.correlation','lec','full rank correlation; nilpotent log chart','correlation_lec'),
('G27','CorrelationAffineQuotient','manifold.correlation','affine_quotient','full rank correlation; AIRM diagonal quotient','correlation_affine'),
('G28','KendallShape','manifold.landmark','kendall','centered unit k x p full rank preshape modulo SO(p)','landmark'),
('G29','Product','manifold.product','weighted_product','named nested tensor tree','weighted_product'),
('R-landmark-O','Riemann landmark','manifold.landmark','kendall; reflections=TRUE','centered unit preshape modulo O(p)','landmark_reflections'),
('R-spdk-factor','Riemann spdk','manifold.spdk','wasserstein','p x k full column rank factor modulo O(k)','spdk_factor')]

def write(name, fields, data):
    with (root/'inst'/'coverage'/name).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(data)
write('geometry.csv',['id','source_geometry','constructor','metric','representation','status','evidence'],
      [(i,s,c,m,r,'supported','test-geometry-contracts; test-solver-evidence: '+e) for i,s,c,m,r,e in rows])
solvers=[
('steepest_descent','Armijo retraction descent','gradient; retraction','supported'),
('conjugate_gradient','transported PR+; descent and periodic restart','gradient; transport','supported'),
('barzilai_borwein','transported BB1 with safeguarded step and Armijo','gradient; transport','supported'),
('lbfgs','transported limited-memory secants; curvature rejection','gradient; transport','supported'),
('trust_regions','Steihaug truncated CG and acceptance ratio','exact Riemannian HVP','supported'),
('newton_cg','truncated CG; negative-curvature safeguard; line search','exact Riemannian HVP','supported'),
('adaptive_regularization_cubics','Cauchy-initialized iterative cubic model minimization','exact Riemannian HVP','experimental'),
('gauss_newton','metric-adjoint normal operator and CG','residual differential and metric adjoint','supported'),
('levenberg_marquardt','damped metric normal operator with ratio updates','residual differential and metric adjoint','supported'),
('stochastic_gradient','vectorized minibatch; explicit sum/mean normalization; chunked full diagnostics','finite-sum autodiff','supported'),
('alternating_gradient','cyclic product block gradient','named product; gradient','supported'),
('particle_swarm','transported velocity; local or aligned chord displacement','tangent; retraction; transport','experimental'),
('nelder_mead','tangent-centroid simplex; retraction reflection and shrink','tangent; retraction; local displacement','experimental'),
('stiefel_annealing','reversible left orthogonal proposals; cooling Metropolis','Stiefel frames; value only','experimental'),
('grassmann_macg','elite MACG sampling; shrinkage fixed-point shape fit','Grassmann frames; value only','experimental')]
write('solvers.csv',['method','variant','requirements','status','reference','evidence'],
      [(m,v,r,s,'inst/math/solvers.md',
        'test-scalable-training' if m == 'stochastic_gradient' else 'test-solver-evidence')
       for m,v,r,s in solvers])
support=[]
for i,s,c,m,r,e in rows:
    evidence='test-solver-evidence: every geometry decreases a custom scalar objective'
    if e=='weighted_product': evidence='test-geometry-contracts: weighted products'
    if e=='landmark_reflections': evidence='test-geometry-contracts: reflection quotient'
    if e=='rigidmotion': evidence='test-geometry-contracts: nested products and rigid motions'
    support.append((i,m,'steepest_descent','ambient reverse-mode','cpu','float64','supported',evidence,'local smooth domain; no blanket higher-order claim'))
for method,variant,req,status in solvers:
    geometry='G01'
    derivative='ambient reverse-mode'
    evidence='test-solver-evidence'
    if method in ['trust_regions','newton_cg','adaptive_regularization_cubics']:
        derivative='exact automatic HVP'
        geometry='G01;G02;G04;G08;G10;G11;G12;G13;G14;G15;G18;G29'
        evidence='test-solver-evidence; test-math-expansion'
    if method in ['gauss_newton','levenberg_marquardt']: derivative='residual JVP and metric adjoint'
    if method in ['conjugate_gradient','barzilai_borwein','lbfgs','steepest_descent']: geometry='G02'
    if method=='alternating_gradient': geometry='G29'
    if method=='stiefel_annealing': geometry='G10';derivative='value only'
    if method=='grassmann_macg': geometry='G11';derivative='value only'
    if method in ['particle_swarm','nelder_mead']:derivative='value only'
    if method == 'stochastic_gradient': evidence='test-scalable-training'
    support.append((geometry,'see geometry ledger',method,derivative,'cpu','float64',status,evidence,req))
for method,geometry,evidence in [('optim_rsgd','G09;G18;G01','mixed matrix and Euclidean training'),('optim_radam','G09;G18;G01;G11','mixed matrix training; factorwise basis invariance')]:
    support.append((geometry,'see geometry ledger',method,'ambient reverse-mode','cpu','float64','supported','test-optim-serialization: '+evidence,'one adaptive scalar per tensor factor'))
support.append(('G01;G10','see geometry ledger','riem.train','ambient reverse-mode','cpu','float64','supported','test-scalable-training: managed module and minibatch placement','one resolved device per run'))
write('support.csv',['geometry_id','metric','method','derivative','device','dtype','status','evidence','qualification'],support)

devices=[
('cpu','float64','supported','test-device-runtime; full package suite','reference validation backend'),
('cpu','float32','supported','test-device-runtime; solver smoke tests','precision-specific tolerances apply'),
('cuda','float64','runtime_checked','riem.devices probe; conditional accelerator tests','requires a CUDA-enabled R torch runtime and compatible visible GPU'),
('cuda','float32','runtime_checked','riem.devices probe; conditional accelerator tests','requires a CUDA-enabled R torch runtime and compatible visible GPU'),
('mps','float32','runtime_checked','riem.devices probe; conditional accelerator tests','requires an MPS-enabled R torch runtime; operation coverage is runtime-qualified'),
('mps','float64','unsupported','device compatibility probe','MPS does not support the preserved float64 default')]
write('devices.csv',['backend','dtype','status','evidence','qualification'],devices)

# Staged upgrade inventory. IDs are capability rows, not a count of spaces.
new_geometry=[
('G30','Repeated factors','manifold.power','sum of base metrics','one tensor with leading factor axis','power'),
('G31','Fixed metric scaling','manifold.scaled','base metric times scale squared','base representation','scaled'),
('G32','Positive tensors','manifold.positive','log_euclidean','positive tensor','positive'),
('G33','Affine subspace','manifold.affine','euclidean','ambient vector with orthonormal basis and offset','affine'),
('G34','Full orthogonal group','manifold.orthogonal','euclidean','square frame; both determinant signs','orthogonal'),
('G35','Positive doubly stochastic','manifold.doublystochastic','fisher','positive matrix with row/column sums one','doublystochastic'),
('G36','Compact fixed rank','manifold.fixedrank','embedded','U S V transpose; full invertible small core; separate tangent factors','compact'),
('G37','Complex Euclidean','manifold.euclidean','real_hermitian','complex tensor','complex_euclidean'),
('G38','Complex sphere','manifold.sphere','round real_hermitian','unit complex vector','complex_sphere'),
('G39','Complex circle','manifold.complexcircle','real_hermitian','unit-modulus complex vector','complex_circle'),
('G40','Complex Stiefel','manifold.stiefel','real_hermitian','complex orthonormal frame','complex_stiefel'),
('G41','Complex Grassmann','manifold.grassmann','real_hermitian','complex quotient frame','complex_grassmann'),
('G42','Unitary group','manifold.unitary','real_hermitian','square complex frame','unitary')]
all_geometry=rows+new_geometry
space_alias={'G03':'G02','G07':'G06','G12':'G11','G34':'G10 (square case)',
 'G36':'G20','G37':'G01 (twice real dimension)','G38':'G02 (twice ambient dimension)',
 'G39':'G08','G42':'G40 (square case)','R-spdk-factor':'G22'}
write('geometry.csv',['id','source_geometry','constructor','metric','representation','status','evidence','space_equivalence'],
 [(i,s,c,m,r,'verified','test-smooth-upgrade; test-training-complex-upgrade; test-upgrade-stress' if (i,s,c,m,r,e) in new_geometry else
   'test-geometry-contracts; test-solver-evidence: '+e,space_alias.get(i,'distinct metric/space family or parameterized construction')) for i,s,c,m,r,e in all_geometry])
more_solvers=[
 ('svrg','snapshot finite-sum gradient with endpoint-compatible transport','local log and qualified snapshot transport','verified'),
 ('srg','recursive transported estimator with periodic full refresh','finite sum; transport along last accepted step','verified'),
 ('augmented_lagrangian','Powell-Hestenes-Rockafellar; adaptive penalty; smooth inner solver','equality/inequality values and metric adjoints','verified'),
 ('proximal_gradient','unaccelerated exponential gradient step and intrinsic prox; backtracking','exact maps; caller-supplied manifold-distance proximal map','verified'),
 ('cyclic_proximal_point','diminishing steps; deterministic cyclic order','matching nonsmooth values and exact proximal maps; cycle stop is not stationarity','verified')]
solvers=[(m,('transported PR+/FR/HS+/DY; descent restarts' if m=='conjugate_gradient' else
 'BB1/BB2/alternating with safeguarded Armijo' if m=='barzilai_borwein' else v),r,('verified' if s=='supported' else s)) for m,v,r,s in solvers]+more_solvers
write('solvers.csv',['method','variant','requirements','status','reference','evidence'],
 [(m,v,r,s,'inst/math/solvers.md; development/competitors/audit.md','test-broader-optimization' if m in [x[0] for x in more_solvers[2:]] else
 'test-smooth-upgrade; test-solver-evidence') for m,v,r,s in solvers])
for i,s,c,m,r,e in new_geometry:
 support.append((i,m,'geometry contract','primitive-qualified','cpu','float64; selected float32','verified',
 'test-smooth-upgrade; test-training-complex-upgrade; test-upgrade-stress','no blanket Hessian or accelerator qualification'))
for method,variant,req,status in more_solvers:
 support.append(('qualified subset','see geometry ledger',method,'problem contract','cpu','float64',status,
 'test-broader-optimization' if method in ['augmented_lagrangian','proximal_gradient','cyclic_proximal_point'] else 'test-smooth-upgrade',req))
for method in ['optim_radam (AMSGrad)','optim_radagrad','optim_rlinesearch','sparse SGD/Adam']:
 support.append(('qualified tensor and row-wise factors','see geometry ledger',method,'ambient reverse-mode','cpu','float64','verified',
 'test-training-complex-upgrade','power factors independent; sparse row-local clocks; line-search closures deterministic'))
write('support.csv',['geometry_id','metric','method','derivative','device','dtype','status','evidence','qualification'],
 [tuple('verified' if z=='supported' else z for z in row) for row in support])
write('devices.csv',['backend','dtype','status','evidence','qualification'],[
 ('cpu','float64; complex128','verified','full package suite on macOS arm64','complex automatic Hessians only for qualified geometries'),
 ('cpu','float32; complex64','verified','test-upgrade-stress; test-device-runtime','selected geometry and solver coverage; not full float32 suite'),
 ('cuda','float64; float32','awaiting_hardware_validation','development/qualification/devices.R','no CUDA hardware exercised locally'),
 ('mps','float32','awaiting_hardware_validation','development/qualification/devices.R','no MPS-enabled R torch runtime exercised locally'),
 ('mps','float64','unsupported','R torch device precision contract','explicit incompatible requests fail'),
 ('Linux CPU','float64','awaiting_hardware_validation','.github/workflows/R-CMD-check.yaml','workflow supplied; not executed from this workspace'),
 ('Windows CPU','float64','awaiting_hardware_validation','.github/workflows/R-CMD-check.yaml','workflow supplied; not executed from this workspace')])
