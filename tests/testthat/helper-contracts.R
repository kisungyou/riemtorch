library(torch)
local_edition(3)
if (torch::torch_is_installed()) torch::torch_set_num_threads(1)
tensor <- function(x) torch::torch_tensor(x, dtype=torch::torch_float64())
scalar <- function(x) as.numeric(x$item())
normf <- function(x) scalar((x*x)$sum()$sqrt())
geom_cases <- function() {
  B<-torch::torch_diag(tensor(c(1,2,3)))
  list(euclidean=manifold.euclidean(c(3,2)),sphere=manifold.sphere(3),
    oblique=manifold.oblique(3,2),multinomial=manifold.multinomial(4),
    poincare=manifold.hyperbolic(2),hyperboloid=manifold.hyperbolic(2,"hyperboloid"),
    torus=manifold.torus(2),stiefel_euclidean=manifold.stiefel(4,2,"euclidean"),
    stiefel_canonical=manifold.stiefel(4,2,"canonical"),grassmann=manifold.grassmann(4,2),
    grassmann_projection=manifold.grassmann(4,2,"projection"),
    generalized_stiefel=manifold.stiefel.generalized(3,2,B),
    generalized_grassmann=manifold.grassmann.generalized(3,2,B),rotation=manifold.rotation(3),
    spd_airm=manifold.spd(3,"airm"),spd_lerm=manifold.spd(3,"lerm"),spd_wasserstein=manifold.spd(3,"wasserstein"),
    fixedrank=manifold.fixedrank(4,3,2),spdk_embedded=manifold.spdk(4,2,"embedded"),
    spdk_wasserstein=manifold.spdk(4,2,"wasserstein"),spdk_factor=manifold.spdk(4,2,"wasserstein","factor"),
    elliptope=manifold.elliptope(4,2),spectrahedron=manifold.spectrahedron(4,2),
    correlation_ecm=manifold.correlation(3,"ecm"),correlation_lec=manifold.correlation(3,"lec"),
    correlation_affine=manifold.correlation(3,"affine_quotient"),landmark=manifold.landmark(5,2))
}
geom_loss <- function(M,target) {
  if(M$name %in% c("grassmann","grassmann.generalized","landmark") && M$metric!="projection_half_frobenius" || M$name=="spdk"&&M$specification$representation=="factor") {
    T<-target$matmul(target$t())
    return(function(x) ((x$matmul(x$t())-T)^2)$sum()/2)
  }
  if(M$name=="torus") return(function(x) (1-(x-target)$cos())$sum())
  function(x) ((x-target)^2)$sum()/2
}
