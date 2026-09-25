# Release checks

1. Install R torch/runtime explicitly, then use an isolated R library. Confirm
   that loading riemtorch neither downloads a runtime nor initializes a device.
2. Generate documentation with roxygen2; keep generated NAMESPACE and man files.
   Confirm the device, placement, managed-training and checkpoint APIs are
   exported and that DESCRIPTION, NEWS and the archive all report version 0.1.0.
3. Run testthat, including mathematical contracts, expanded Hessian/map tests,
   native batch agreement, selection precedence, registered-data movement,
   scalable finite sums, managed training and both checkpoint schemas.
4. On the CPU reference host, test float64 fully and the documented float32 smoke
   paths. On each claimed accelerator host, record `riem.devices()`, run the
   conditional suite, select a nondefault indexed CUDA device where available,
   and prove `device = "cpu"` overrides automatic selection.
5. Build the source tarball with all 15 executed vignettes. Rebuild the
   applications/device vignette on CPU so its output is portable.
6. Run standard and `--as-cran` checks on the tarball and inspect every error,
   warning and note. Run installed examples and the reference manual separately.
7. Run matched-accuracy benchmarks; retain failed runs and separate warm-up,
   transfers, synchronized computation and total time. Do not infer an
   accelerator ranking from discovery probes or a single run.
8. Verify every public help topic has a runnable example. Check that captured
   callback tensors are described as caller-owned and registered data is used in
   portable examples.
9. Regenerate all ledgers, including `devices.csv`, from
   `development/generate-ledgers.py` if the tested inventory changes. Leave CUDA
   and MPS as runtime-checked until executed hardware evidence is recorded.
10. Record versions, platform, check status, expectation counts, archive hash and
    remaining limitations in the implementation and verification reports.
    Add evidence for the current source archive after it has completed the
    checks above. Preserve historical version labels, measurements and hashes;
    the current version reset is documented in
    [verification/version-0.1.0.md](verification/version-0.1.0.md).
11. Before public release, separately confirm package-name availability and
    review attribution/license obligations. Local implementation is not CRAN
    publication, registration or a claim of legal clearance.
