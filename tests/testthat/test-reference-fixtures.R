test_that("portable Riemann distance fixtures supplement analytic tests", {
  skip_if_not(torch::torch_is_installed())
  fixtures<-readRDS(system.file("extdata","riemann-distance-fixtures.rds",package="riemtorch"))
  geometries<-list(spd_airm=manifold.spd(2,"airm"),spd_lerm=manifold.spd(2,"lerm"),sphere=manifold.sphere(3))
  for(name in names(fixtures)) {
    z<-fixtures[[name]]
    expect_equal(scalar(riem.dist(geometries[[name]],tensor(z$x),tensor(z$y))),z$distance,tolerance=1e-10)
  }
})
