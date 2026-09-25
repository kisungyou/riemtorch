"""Run from the package root using an isolated reference Python environment."""
import csv, pathlib, subprocess, sys, time, os, psutil, platform, json
root=pathlib.Path.cwd();here=root/'development/benchmarks/upgrade'
out=here/'evidence';out.mkdir(exist_ok=True);inputs=out/'inputs'
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
subprocess.run(['Rscript',str(here/'prepare.R'),str(inputs)],check=True,env=env,stdout=subprocess.DEVNULL)
rows=[]
for seed in range(1,4):
 for case in ['eigen','procrustes','completion','spd','robust','hyperbolic','phase','constrained']:
  for impl in ['riemtorch','reference']:
   output=out/f'{case}-{seed}-{impl}.csv';ready=pathlib.Path(str(output)+'.ready');ready.unlink(missing_ok=True)
   cmd=(['Rscript',str(here/'riemtorch-worker.R')] if impl=='riemtorch' else [sys.executable,str(here/'reference-worker.py')])+[case,str(inputs/f'seed{seed}'),str(output)]
   logpath=out/f'{case}-{seed}-{impl}.log';baseline=None;peak=0;peak_run=0
   with logpath.open('w') as log:
    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=log);ps=psutil.Process(proc.pid)
    while proc.poll() is None:
     try:
      rss=ps.memory_info().rss;peak=max(peak,rss)
      if ready.exists():
       if baseline is None:baseline=rss
       peak_run=max(peak_run,rss)
     except psutil.NoSuchProcess:pass
     time.sleep(.02)
   if proc.returncode!=0:
    row=dict(case=case,seed=seed,implementation=impl,termination='worker_failure',error=logpath.read_text()[-1800:])
   else:
    with output.open() as f:row=next(csv.DictReader(f))
   row.update(peak_rss_bytes=peak,baseline_rss_bytes=baseline,peak_run_rss_bytes=peak_run if baseline is not None else None,
              memory_status="sampled" if baseline is not None else "short_run_unobserved",
              memory_method='process RSS sampled every 20 ms; includes interpreter/runtime, may miss brief peaks')
   rows.append(row);print(case,seed,impl,row['termination'],flush=True)
keys=list(dict.fromkeys(k for row in rows for k in row))
with (out/'results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
import pymanopt,numpy,scipy
(out/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
 pymanopt=pymanopt.__version__,numpy=numpy.__version__,scipy=scipy.__version__,seeds=[1,2,3],precision='float64 / complex128',device='CPU'),indent=2))
