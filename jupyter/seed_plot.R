library(rgl)
library(colorRamps)
library(viridis)
library(RColorBrewer)
library(dplyr)

setwd('/home/ejam/documents/barley_stacks/preproc/temp/seeds/S017/')

plot_coords <- function(src, sname, color, axis, type, rd, sz){
  foosrc <- paste(src, color, '/', sep='')
  B <- as.matrix(read.csv(paste(foosrc,sname,'_coords.csv',sep = ''), header=F))
  sigma <- unlist(as.vector(read.csv(paste(foosrc,sname,'_sigma.csv', sep=''), header=F)))
  maxvox <- unlist(as.vector(read.csv(paste(foosrc,sname,'_algn_max_vox.csv', sep=''), header=F)))
  #colrs <- unlist(as.vector(read.csv(paste(foosrc,sname,'_heights_',axis,'.csv', sep=''), header=F)))
  #col <- (viridis::plasma(32)) #rev
  
  open3d()
  rgl::par3d(windowRect = c(50,50,500,500))
  plot3d(B, xlab = 'x', ylab = 'y', zlab = 'z',
         col = 'peru', type = type, lwd = 4, radius = rd, size = sz)
  aspect3d('iso')
  axes3d()
  
  lines3d(c(sigma[1],-sigma[1]), c(0,0), c(0,0),
          lwd=10, col = 'firebrick1')
  lines3d(c(0,0), c(sigma[2],-sigma[2]), c(0,0),
          lwd=8, col = 'blue3')
  lines3d(c(0,0),c(0,0),c(sigma[3],-sigma[3]),
          lwd=6, col = 'forestgreen')
  rgl.viewpoint(-0,10,zoom = 0.3)
}

type <- 's'
rd <- 0.75
sz <- 5
src <- '/home/ejam/downloads/temp/'
scan <- 224
color <- 'Red'
sname <- 'seed_13_0'

plot_coords(paste(src,'S',formatC(scan,width = 3,flag='0'),'/',sep=''), sname, color, 'X', type, rd, sz)

src <- paste(src,'S',formatC(scan,width = 3,flag='0'),'/',sep='')

directions <- as.matrix(read.csv(paste(src,'dirs.csv',sep = ''), header=F))
directions <- 40*directions
#foo <- matrix(0, nrow = nrow(directions), ncol = ncol(directions))
points3d(directions, col='magenta', size=15)
lines3d(directions)
texts3d(directions, texts=0:(length(directions)-1), font=3, cex=2)

######################################################################################
######################################################################################
######################################################################################
sz <- 8
axis <- 'Z'
sname <- 'seed_10_0'
TT <- 32

B <- as.matrix(read.csv(paste(sname,'_coords.csv',sep = ''), header=F))
sigma <- unlist(as.vector(read.csv(paste(sname,'_sigma.csv', sep=''), header=F)))
colrs <- unlist(as.vector(read.csv(paste(sname,'_coords_',axis,'.csv', sep=''), header=F)))
col <- (viridis::magma(TT)) 

lout <- 50
MX <- cbind(seq(-sigma[1], sigma[1], length.out = lout), rep.int(0,lout), rep.int(0,lout))
MY <- cbind(rep.int(0,lout), seq(-sigma[1], sigma[1], length.out = lout), rep.int(0,lout))
MZ <- cbind(rep.int(0,lout), rep.int(0,lout), seq(-sigma[1], sigma[1], length.out = lout))

rgl::open3d()
rgl::par3d(windowRect = c(50,50,600,400))
bar <- rbind(apply(B, 2, max), apply(B, 2, min))
rgl::plot3d(bar, xlab = '', ylab = '', zlab = '',
            col = 'white', type = 'n', lwd = 4, radius = rd, size = 0.1, alpha=0,
            axes=FALSE, ann=FALSE)
rgl::aspect3d('iso')
rgl::points3d(B, col=col[colrs], size=8)

rgl::segments3d(MX,
             lwd=10, col = 'firebrick1')
rgl::segments3d(MY,
             lwd=8, col = 'blue3')
rgl::segments3d(c(0,0),c(0,0),c(sigma[3],-sigma[3]),
             lwd=6, col = 'forestgreen')

rgl::rgl.viewpoint(userMatrix = um2,zoom = 0.25, fov=30)

rgl::plot3d(bar, xlab = '', ylab = '', zlab = '',
            col = 'white', type = 'n', lwd = 4, radius = rd, size = 0.1, alpha=0,
            axes=FALSE, ann=FALSE)
rgl::aspect3d('iso')

i <- 0
for( i in 1:length(col)){
  rgl::points3d(B[which(colrs <= i),],
              col =col[colrs[which(colrs <= i)]], size = sz)
  
  rgl::segments3d(MX,
                  lwd=10, col = 'firebrick1')
  rgl::segments3d(MY,
                  lwd=8, col = 'blue3')
  rgl::segments3d(c(0,0),c(0,0),c(sigma[3],-sigma[3]),
                  lwd=6, col = 'forestgreen')
  rgl::rgl.viewpoint(userMatrix = um2,zoom = 0.25, fov=30)
  
  rgl::rgl.snapshot(paste(sname,'_full_',axis,'_',formatC(i, digits = 1, format='d', flag='0'),'.png', sep=''), fmt='png')
}

um2 <- par3d()$userMatrix
