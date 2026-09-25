# Compact low-rank optimization

Compact fixed-rank points store orthonormal U and V factors and an
invertible square core S. The core may vary off-diagonally under
autodiff, avoiding the singular coordinate chart of diagonal-only SVD
parameters at repeated spectra. Tangents use different geometric
meanings for the same three named leaves; only retractions turn them
into new points.

``` r

M <- manifold.fixedrank(20,15,2,representation="svd")
x <- riem.random(M,device="cpu")
rows <- rep(1:20,each=15)
columns <- rep(1:15,times=20)
target <- riem.random(M,device="cpu")
data <- list(rows=rows,columns=columns,
             values=riem.materialize(M,target,rows,columns))
P <- riem.problem(M,function(x,data) {
  residual <- riem.materialize(M,x,data$rows,data$columns)-data$values
  residual$square()$mean()/2
},data=data)
fit <- riem.optimize(P,x,"lbfgs",control=list(max_iterations=30),device="cpu")
fit$objective
#> [1] 3.50596e-13
riem.belongs(M,fit$point)
#> [1] TRUE
```

Selected entries and small QR/SVD retractions avoid dense ambient
allocations. Objectives must be invariant to factor changes that
preserve U S V’. Automatic second-order differentiation and independent
batches are not advertised for this representation; use the dense
geometry when those requirements matter.
