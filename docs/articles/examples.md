# Example gallery

Start with a foundation example to learn the problem interface, or
choose a public-data application close to your work. Every page contains
runnable code, figures, and checks of the numerical result. The six
applications use data already included with R or bundled in this
website; no dataset download is needed when building the documentation.

## Foundations

These short introductions isolate the optimization ideas. The sphere
example uses two iris measurements; the other three construct small
synthetic problems with known reference answers.

[![Centered iris measurements with the optimized principal
direction.](example-sphere-pca_files/figure-html/direction-plot-1.png)](https://www.kisungyou.com/riemtorch/articles/example-sphere-pca.md)

Sphere **Find a principal direction**

Recover the direction of greatest variation in two iris measurements.

**Dataset** Iris

Introductory CPU render: 7.0 s

Read example →

[![Synthetic observations and fitted quadratic and Huber regression
lines.](example-robust-regression_files/figure-html/fits-plot-1.png)](https://www.kisungyou.com/riemtorch/articles/example-robust-regression.md)

Euclidean **Compare quadratic and Huber loss**

See how Huber loss changes a regression fit when observations contain
outliers.

**Dataset** Synthetic regression

Introductory CPU render: 6.1 s

Read example →

[![Ellipses comparing the input matrices and their geometric and
arithmetic
averages.](example-spd-mean_files/figure-html/ellipses-1.png)](https://www.kisungyou.com/riemtorch/articles/example-spd-mean.md)

Positive-definite matrices **Compute a geometric midpoint**

Compare a geometric midpoint with the arithmetic average of two
covariance shapes.

**Dataset** Two synthetic matrices

Intermediate CPU render: 8.8 s

Read example →

[![Heatmaps of the true, partially observed, and recovered rank-two
matrix.](example-matrix-completion_files/figure-html/matrix-panels-1.png)](https://www.kisungyou.com/riemtorch/articles/example-matrix-completion.md)

Compact fixed rank **Complete a low-rank matrix**

Recover missing entries while keeping the matrix in a compact rank-two
representation.

**Dataset** Synthetic rank-two matrix

Intermediate CPU render: 8.0 s

Read example →

## Public-data applications

Each application starts with the observations and explains how the
objective relates to the question being asked. The independent checks
are available after the main workflow, so you can follow the data
analysis before studying its numerical details.

[![Penguin observations projected onto two dimensions and colored by
species.](example-penguin-subspace_files/figure-html/penguin-projection-1.png)](https://www.kisungyou.com/riemtorch/articles/example-penguin-subspace.md)

Grassmann **Find a penguin subspace**

Project four penguin body measurements onto a two-dimensional plane and
compare with PCA.

**Dataset** Palmer Penguins

Intermediate CPU render: 10.7 s

Read example →

[![Fitted values and residuals for quadratic and Huber industrial
regression.](example-stackloss-regression_files/figure-html/fitted-residuals-1.png)](https://www.kisungyou.com/riemtorch/articles/example-stackloss-regression.md)

Euclidean **Fit robust industrial regression**

Fit quadratic and Huber models to industrial measurements and inspect
their residuals.

**Dataset** Stack loss

Introductory CPU render: 8.6 s

Read example →

[![Heatmaps separating missing and withheld measurements and showing
their
reconstruction.](example-airquality-completion_files/figure-html/patterns-1.png)](https://www.kisungyou.com/riemtorch/articles/example-airquality-completion.md)

Compact fixed rank **Reconstruct missing measurements**

Fill gaps in environmental measurements and assess predictions on
held-out observations.

**Dataset** Air quality

Intermediate CPU render: 9.3 s

Read example →

[![Heatmaps comparing arithmetic, log-Euclidean, and affine-invariant
covariance
means.](example-covariance-means_files/figure-html/covariance-heatmaps-1.png)](https://www.kisungyou.com/riemtorch/articles/example-covariance-means.md)

SPD: log-Euclidean and affine-invariant **Average observed covariances**

Compare geometric averages of covariance matrices estimated from
stock-index returns.

**Dataset** European stock indices

Intermediate CPU render: 9.3 s

Read example →

## Constraints and sparsity

Euclidean space also fits the manifold interface. These examples use it
to make equality constraints, inequalities, and an exact proximal
operator easy to understand and verify.

[![Observed tree volume against fitted volume with a perfect-fit
reference
line.](example-tree-constraints_files/figure-html/fit-plot-1.png)](https://www.kisungyou.com/riemtorch/articles/example-tree-constraints.md)

Euclidean with equality/inequality constraints **Constrain a tree-volume
model**

Fit a log-volume model with nonnegative slopes that sum to three, then
check the constraints.

**Dataset** Black cherry trees

Intermediate CPU render: 11.8 s

Read example →

[![Five standardized regression coefficients along a decreasing sequence
of L1
penalties.](example-mtcars-proximal_files/figure-html/coefficient-path-1.png)](https://www.kisungyou.com/riemtorch/articles/example-mtcars-proximal.md)

Euclidean with L1 penalty **Trace a sparse regression path**

Follow regression coefficients across five L1 penalties using an exact
soft-thresholding step.

**Dataset** Motor Trend cars

Intermediate CPU render: 17.9 s

Read example →

## Reading the results

The CPU times are measured complete article render times, not
solver-only benchmarks or speed guarantees. They include data
preparation, independent checks, figures, and HTML rendering in fresh R
sessions. Values are the mean of two serial runs on the reference
machine: macOS arm64, R 4.5.2, torch 0.17.0, CPU float64, one torch
thread. The [timing
table](https://www.kisungyou.com/riemtorch/articles/data/example-runtimes.csv)
contains both measurements.

Each public-data example reports its termination reason and the
residuals appropriate to the method. A small training loss is not proof
of accurate missing-value predictions, and a stopped iteration budget is
not convergence. The articles discuss these distinctions using their
actual results.

Use `device = "auto"` to request automatic device selection or
`device = "cpu"` to force CPU execution, including on a GPU workstation.
An explicit CUDA device must be available and support the requested
precision. See the [device
guide](https://www.kisungyou.com/riemtorch/articles/applications-and-devices.md).

For citations, units, missing values, the penguin CSV, and the
reproducibility procedure, see [Data sources and
reproducibility](https://www.kisungyou.com/riemtorch/articles/example-data-sources.md).
