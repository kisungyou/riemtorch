# Sparse embeddings and resumable training

A power manifold represents rows as independent factors within one
point. Riemannian Adam and AdaGrad keep one variance per factor. Sparse
SGD and Adam coalesce repeated row indices; untouched rows, momenta,
variances and clocks remain unchanged. Adam uses row-local clocks for
bias correction. Dense and sparse state layouts cannot be mixed for one
parameter.

``` r

M <- manifold.power(manifold.sphere(3),8)
embedding <- nn_embedding(8,3,sparse=TRUE)$to(dtype=torch_float64())
with_no_grad(embedding$weight$copy_(riem.random(M,device="cpu")))
#> torch_tensor
#>  0.2315  0.9176  0.3232
#>  0.9086 -0.3998  0.1212
#> -0.4193  0.5456  0.7256
#>  0.5379  0.2588  0.8023
#> -0.4297 -0.3071  0.8491
#>  0.2290  0.7948 -0.5621
#> -0.3930  0.7603  0.5172
#>  0.7106  0.6922 -0.1260
#> [ CPUDoubleType{8,3} ][ requires_grad = TRUE ]
opt <- optim_radam(list(embedding$weight),manifold=M,lr=0.01,amsgrad=TRUE)
opt$zero_grad()
y <- embedding(torch_tensor(c(2L,2L,5L),dtype=torch_int64()))
(-y[,1]$sum())$backward()
opt$step()
opt$state$get(embedding$weight)$row_steps
#> torch_tensor
#>  0
#>  1
#>  0
#>  0
#>  1
#>  0
#>  0
#>  0
#> [ CPULongType{8} ]
riem.belongs(M,embedding$weight)
#> [1] TRUE
```

[`riem.train()`](https://www.kisungyou.com/riemtorch/reference/riem.train.md)
accepts `accumulate` (average minibatch gradients, including the final
partial group), `clip_norm` (total metric norm), `validation`,
`scheduler`, and an epoch callback. A callback returning TRUE requests a
user stop. Validation runs in evaluation mode without a gradient graph.

The returned `training_state` includes epoch and step counters. Optional
`data_state=list(get=function() state, set=function(x) ...)` hooks
capture and restore caller-owned data-order or scheduler state before
the next iterator. Save model, optimizer, that progress and RNG together
for epoch-boundary resume. Older optimizer checkpoints remain readable.
Mid-epoch replay is not claimed.

``` r

path <- tempfile(fileext=".pt")
riem.save(opt,path,model=embedding,training_state=list(epoch=1L,step=1L))
riem.load(opt,path,model=embedding,restore_rng=TRUE)
unlink(path)
```

Line-search training uses
[`optim_rlinesearch()`](https://www.kisungyou.com/riemtorch/reference/optim_rlinesearch.md)
with a deterministic closure that clears gradients and recomputes both
loss and backward pass. Failed searches restore all parameters and
gradients. Closures must not mutate buffers or RNG.
