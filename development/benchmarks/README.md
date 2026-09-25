# Matched-accuracy benchmarks

Run `Rscript development/benchmarks/run.R` from the package root with riemtorch
and torch installed. `results.csv` retains every run, including failures to meet
accuracy. All methods share the same initialization, objective, metric, dtype,
CPU device and stopping budget per problem. The reported accuracy requires
objective gap <=1e-8, metric gradient norm <=1e-5 and constraint residual <=1e-7.
Warmups are excluded; these single-run local timings are descriptive and do not
establish stable performance rankings or comparisons with other packages.

`kernel-costs.csv` separately measures value and value-plus-gradient evaluations.
The difference between those and full solve time is not attributed entirely to
solver overhead, because line searches and validation perform different work.
`environment.txt` records the environment. R memory accounting excludes native
torch allocations, so no memory or accelerator speedup claim is made.

The initial nine runs (three solvers on sphere, Procrustes and SPD log-coordinate
problems) all reached the stated matched-accuracy criterion. Training throughput
is demonstrated in the vignette, without asserting a benchmark speedup.

`device-run.R` is the portable accelerator smoke benchmark. It accepts
`--device=auto`, `--device=cpu`, or an indexed CUDA request, plus a `--dtype`
and output path. It records the selected device and separates initialization
transfer time from synchronized computation. Run it independently on each
workstation; an available accelerator is not itself evidence of a speedup.
