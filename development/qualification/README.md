# Platform qualification

The workflow runs standard package checks and vignettes on Linux, macOS and
Windows with a CPU Torch installation. No remote workflow has been dispatched
from this local workspace. Those OS labels require actual uploaded successful
logs before changing the evidence ledger.

On a CUDA workstation, install the source archive and run:

```
Rscript development/qualification/devices.R /path/to/evidence
```

The runner tests automatic selection, explicit CPU, every visible CUDA index,
float64/float32 agreement, synchronized timing, transfer costs and checkpoint
transfer/continuation to CPU. Compatible MPS float32 is included where exposed by
R torch. A failed request records its error and fails the runner. Missing
accelerators are recorded as awaiting validation, not passed tests. Driver and
GPU-model inventory is saved from nvidia-smi when available. CUDA-visible indices
can differ from physical GPU indices under CUDA_VISIBLE_DEVICES.

The CPU tests include the full float64 regression suite and additional float32
contracts for the new real/complex geometries and smooth solvers. Device smoke
qualification is narrower than the full mathematical suite; probes only test
the requested operation set. No silent device change occurs within a run.
