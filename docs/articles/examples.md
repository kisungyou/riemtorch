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

| Application | Dataset | Geometry | Level | CPU render |
|:---|:---|:---|:---|:---|
| [Find a principal direction](https://www.kisungyou.com/riemtorch/articles/example-sphere-pca.md) | Iris | Sphere | Introductory | 7.0 s |
| [Compare quadratic and Huber loss](https://www.kisungyou.com/riemtorch/articles/example-robust-regression.md) | Synthetic regression | Euclidean | Introductory | 6.1 s |
| [Compute a geometric midpoint](https://www.kisungyou.com/riemtorch/articles/example-spd-mean.md) | Two synthetic matrices | Positive-definite matrices | Intermediate | 8.8 s |
| [Complete a low-rank matrix](https://www.kisungyou.com/riemtorch/articles/example-matrix-completion.md) | Synthetic rank-two matrix | Compact fixed rank | Intermediate | 8.0 s |

## Public-data applications

Each application starts with the observations and explains how the
objective relates to the question being asked. The independent checks
are available after the main workflow, so you can follow the data
analysis before studying its numerical details.

| Application | Dataset | Geometry | Level | CPU render |
|:---|:---|:---|:---|:---|
| [Find a penguin subspace](https://www.kisungyou.com/riemtorch/articles/example-penguin-subspace.md) | Palmer Penguins | Grassmann | Intermediate | 10.7 s |
| [Fit robust industrial regression](https://www.kisungyou.com/riemtorch/articles/example-stackloss-regression.md) | Stack loss | Euclidean | Introductory | 8.6 s |
| [Reconstruct missing measurements](https://www.kisungyou.com/riemtorch/articles/example-airquality-completion.md) | Air quality | Compact fixed rank | Intermediate | 9.3 s |
| [Average observed covariances](https://www.kisungyou.com/riemtorch/articles/example-covariance-means.md) | European stock indices | SPD: log-Euclidean and affine-invariant | Intermediate | 9.3 s |

## Constraints and sparsity

Euclidean space also fits the manifold interface. These examples use it
to make equality constraints, inequalities, and an exact proximal
operator easy to understand and verify.

| Application | Dataset | Geometry | Level | CPU render |
|:---|:---|:---|:---|:---|
| [Constrain a tree-volume model](https://www.kisungyou.com/riemtorch/articles/example-tree-constraints.md) | Black cherry trees | Euclidean with equality/inequality constraints | Intermediate | 11.8 s |
| [Trace a sparse regression path](https://www.kisungyou.com/riemtorch/articles/example-mtcars-proximal.md) | Motor Trend cars | Euclidean with L1 penalty | Intermediate | 17.9 s |

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
