import pathlib,subprocess,time,csv,psutil,os
root=pathlib.Path.cwd();here=root/'development/benchmarks/upgrade';out=here/'evidence';out.mkdir(exist_ok=True)
rows=[]
for mode in ['diagnostic_value','gradient','hessian_vector','differentiable_value']:
 for n in [1024,65536]:
  file=out/f'memory-{mode}-{n}.csv';ready=pathlib.Path(str(file)+'.ready');ready.unlink(missing_ok=True)
  baseline=None;peak=0
  with (out/f'memory-{mode}-{n}.log').open('w') as log:
   p=subprocess.Popen(['Rscript',str(here/'memory-worker.R'),str(n),mode,str(file)],stdout=log,stderr=log);ps=psutil.Process(p.pid)
   while p.poll() is None:
    try:
     rss=ps.memory_info().rss
     if ready.exists():
      if baseline is None:baseline=rss
      peak=max(peak,rss)
    except psutil.NoSuchProcess:pass
    time.sleep(.005)
  if p.returncode:raise RuntimeError(f'{mode} {n} failed')
  with file.open() as f:row=next(csv.DictReader(f))
  row.update(baseline_rss_bytes=baseline,peak_rss_bytes=peak,increment_bytes=max(0,peak-(baseline or 0)),sampling_ms=5)
  rows.append(row)
with (out/'memory.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(rows)
