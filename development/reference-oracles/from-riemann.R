# Run only in a separate development session; Riemann is not a runtime dependency.
# Working directory: riemtorch package root.
stopifnot(requireNamespace("Riemann",quietly=TRUE))
stopifnot(as.character(utils::packageVersion("Riemann"))=="0.1.7")
x<-matrix(c(2,.3,.3,1),2);y<-matrix(c(1.2,.1,.1,3),2)
sphere<-list(c(1,0,0),c(1,2,2)/3)
pairs<-list(spd_airm=list(x=x,y=y),spd_lerm=list(x=x,y=y),
            sphere=list(x=sphere[[1]],y=sphere[[2]]))
D<-Riemann::wrap.spd(list(x,y))
pairs$spd_airm$distance<-Riemann::riem.pdist(D,"intrinsic")[1,2]
pairs$spd_lerm$distance<-Riemann::riem.pdist(D,"extrinsic")[1,2]
pairs$sphere$distance<-Riemann::riem.pdist(Riemann::wrap.sphere(sphere))[1,2]
saveRDS(pairs,"inst/extdata/riemann-distance-fixtures.rds",version=2)
writeLines(c("Generated using installed Riemann 0.1.7 in an independent R session.",
 "Source baseline audited separately: e847692576f8b250c8140e1343b145f72caa379f.",
 "The installed binary does not expose a source commit; do not infer build provenance.",
 "Inputs are deterministic constants; expected distances come from Riemann::riem.pdist.",
 capture.output(sessionInfo())),"development/reference-oracles/environment.txt")
print(vapply(pairs,`[[`,numeric(1),"distance"))
