library(rgl)

sname <- 'GLYMA_05G092200_suplevel_gene_c2014'
sname <- 'GLYMA_05G092200_suplevel_gene_c368'
setwd(paste('~/documents/mcarto/animation/D2/', sname, sep=''))

data <- read.csv('coords_kde.csv')
diag <- read.csv('persistence_diagram.csv')

coords <- as.matrix(data[, c(3,1,2)])
vals <- as.vector(data[, 4])
bar <- rbind(apply(coords, 2, min), apply(coords, 2, max))
thresh <- c(0, sort(unique(unlist(as.vector(diag[,-1])))), 256)
thresh <- base::seq(0,256, by=4)

presorted <- vector(length = length(vals))
for( i in 1:(length(thresh)-1)){
  presorted[(vals >= thresh[i]) & (vals < thresh[i+1])] <- i
}
print(min(presorted))
print(max(presorted))

col <- rev(viridis::magma(64))

fov <- 0
zoom <- 0.6
sz <- 10
phi <- 0
theta <- 0

rgl::open3d()
rgl::par3d(windowRect = c(50,50,600,400))
rgl::plot3d(bar, xlab = '', ylab = '', zlab = '', axes=FALSE, ann=FALSE,
            col = 'white', type = 'n', lwd = 4, radius = 1, size = 1, alpha=0)
rgl::aspect3d('iso')
rgl::points3d(coords, col=col[presorted], size=sz)
rgl::view3d(theta=0, phi=0, fov=0, zoom=0.6)
rgl::view3d(theta=0, phi=32, fov=0, zoom=0.6)
rgl::view3d(theta=0, phi=64, fov=0, zoom=0.6)
rgl::plot3d(bar, xlab = '', ylab = '', zlab = '', axes=FALSE, ann=FALSE,
            col = 'white', type = 'n', lwd = 4, radius = 1, size = 1, alpha=0)
rgl::aspect3d('iso')
rgl::bg3d("gray")

i <- 3L
for ( i in 1:length(col)){
  if(length(which(presorted <= i)) > 0){
    rgl::points3d(coords[which(presorted <= i),],  col=col[presorted[which(presorted <= i)]], size = sz)
    rgl::view3d(theta=theta, phi=phi+i, zoom =zoom, fov=fov)
  }
  filename <- paste(sname,'_',formatC(i, digits = 2, format='d', flag='0'),'.png', sep='')
  rgl::rgl.snapshot(filename, fmt='png')
}

um2 <- rgl::par3d()$userMatrix
um2 <-
[1,]  0.99751705 -0.02840403 0.06444369    0
[2,] -0.03780482  0.55607587 0.83027124    0
[3,] -0.05941862 -0.83064592 0.55362135    0
[4,]  0.00000000  0.00000000 0.00000000    1