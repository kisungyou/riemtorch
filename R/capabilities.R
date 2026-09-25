#' Inspect Primitive-Level Geometry Capabilities
#' @param manifold A geometry specification.
#' @return A data frame with operation, value, first_derivative,
#' second_derivative, and native_batch columns. FALSE derivative entries mean unsupported or
#' unverified, not necessarily mathematical nonsmoothness. Membership/random
#' operations are not differentiated. Domain restrictions still apply.
#' @details A first-order optimizer usually only needs values of retraction and
#' gradient conversion. It does not imply differentiating those primitives.
#' A true value for `ehess2rhess` records a verified exact conversion from an
#' ambient Hessian-vector product, rather than a projected approximation.
#' Product records take the intersection of all factors' capabilities.
#' @examples
#' riem.capabilities(manifold.stiefel(4, 2, "euclidean"))
#' @export
riem.capabilities <- function(manifold) {
  .check_manifold(manifold)
  ops<-c("belongs","tangent","inner","egrad2rgrad","retr","transport","ehess2rhess","exp","log","sqdist","project","random")
  if(inherits(manifold,"riem_product")) {
    tabs<-lapply(manifold$factors,riem.capabilities);ans<-tabs[[1]]
    for(j in seq.int(2,ncol(ans))) ans[[j]]<-Reduce(`&`,lapply(tabs,function(z)z[[j]]))
    return(ans)
  }
  value<-ops %in% names(manifold$operations)
  value[ops=="transport"]<-TRUE
  value[ops=="random"]<-value[ops=="random"]||value[ops=="project"]
  value[ops=="ehess2rhess"]<-value[ops=="ehess2rhess"]&&isTRUE(manifold$capabilities$hessian)
  order<-setNames(rep(0L,length(ops)),ops)
  declared <- manifold$capabilities$derivatives
  if(length(declared)) order[intersect(names(declared),ops)] <- declared[intersect(names(declared),ops)]
  batch<-utils::modifyList(.builtin_batch_operations(manifold),manifold$batch_operations)
  data.frame(operation=ops,value=value,first_derivative=value & order>=1L,
             second_derivative=value & order>=2L,
             native_batch=ops%in%names(batch),row.names=NULL)
}
