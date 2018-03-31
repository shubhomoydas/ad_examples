#rm(list=ls())
library("MASS")

euclidean_dist <- function(x1, x2) {
  v <- x1 - x2
  return (sqrt(sum(v^2)))
}

if (F) {
  source("/Users/moy/work/git/bb_python/R/manifold_diffusion.R")
}

colrs = c("blue", "red", "green", "brown")

sample_type = "donut"
# sample_type = "face"

#============ Label diffusion ===============
set.seed(42)

if (F) {
  pN <- 40
  s1 <- mvrnorm(n=pN, mu=c(1, 1), Sigma=diag(2)*0.2)
  s2 <- mvrnorm(n=pN, mu=c(4, 4), Sigma=diag(2)*0.3)
  s3 <- mvrnorm(n=pN, mu=c(1, 3), Sigma=diag(2)*0.3)
  cls <- rep(c(1, 2, 3), each=pN)
  samples <- rbind(s1, s2, s3)
} else {
  pN <- 100
  if (sample_type == "donut") {
    samples_path = "/Users/moy/work/git/bb_python/datasets/donut-shape.csv"
    fdata <- read.csv(samples_path, header=T)
    samples <- fdata[, 1:2]
    samples <- as.matrix(samples)
    cls <- rep(1, nrow(samples))
    cls[nrow(samples)] = 2
    k2 = 10
  } else {
    samples_path = "/Users/moy/work/git/bb_python/datasets/face.csv"
    fdata <- read.csv(samples_path, header=T)
    samples <- fdata[, 2:3]
    samples <- as.matrix(samples)
    cls <- fdata[, 1]
    k2 = 1.0
  }
}

y <- c(sample(1:pN, 3)) + c(0, pN, pN*2)

n <- nrow(samples)
nn <- 5

dists <- matrix(0, nrow=n, ncol=n)
for (i in 1:n) {
  for (j in i:n) {
    dists[i, j] <- euclidean_dist(samples[i, ], samples[j, ])
    dists[j, i] <- dists[i, j]
  }
}

neighbors <- matrix(0, nrow=n, ncol=nn)
for (i in 1:n) {
  neighbors[i, ] <- order(dists[i, ])[1:nn]
}
# dists[i, neighbors[i, ]]

W <- matrix(0, nrow=n, ncol=n)
for (i in 1:n) {
  for (j in neighbors[i, ]) {
    W[i, j] <- exp(-euclidean_dist(samples[i, ], samples[j, ])^2/k2)
    W[j, i] <- W[i, j]
  }
}
diag(W) <- 0 # W has 0 on diagonal
# round(W[1:5, 1:5], 3)

D <- diag(apply(W, MAR=c(1), FUN=sum))
# round(D[1:5, 1:5], 9)

iDroot <- diag(sqrt(diag(D))^(-1))
# round(iDroot[1:10, 1:10], 3)

S <- iDroot %*% W %*% iDroot
# round(S[1:10, 1:10], 3)

alpha <- 0.99
Fl <- matrix(0, nrow=nrow(samples), ncol=3)
Y <- matrix(0, nrow=nrow(samples), ncol=3)
for (i in y) {
  Y[i, cls[i]] <- 1
}

if (F) {
  pdf(file.path("/Users/moy/work/git/bb_python/python/temp/diffusion", sprintf("%s_orig.pdf", sample_type)))
  plot(samples[, 1], samples[, 2], pch="*", cex=2.5, axes=F, xlab=NA, ylab=NA,
       col=colrs[cls])
  dev.off()
}
pltidxs <- c(1, 2, 3, 4, 10, 45)
pdf(file.path("/Users/moy/work/git/bb_python/python/temp/diffusion", sprintf("%s_iter_%0d.pdf", sample_type, 0)))
par(mfrow=c(2, 3), 
    oma = c(1,1,0,0) + 0.1,
    mar = c(2, 2, 3, 1) + 0.1 # bottom, left, top and right
    # mai=c(0.42, 0.42, 0.42, 0.42) # default is 1.02 0.82 0.82 0.42
    ) 
if (F) {
  plot(samples[, 1], samples[, 2], pch="*", cex=2.5, axes=F, xlab=NA, ylab=NA,
       col="grey50")
  points(samples[y, 1], samples[y, 2], pch="o", cex=2.5, 
         col=colrs[cls])
}
#dev.off()
for (i in 1:100) {
  Fl <- alpha * S %*% Fl + (1-alpha) * Y
  if (length(which(pltidxs==i)) == 0) {
    next
  } else {
    colvals <- apply(Fl, MAR=c(1), FUN=max)
    lblvals <- max.col(Fl)
    cols <- c()
    for (l in 1:nrow(Fl)) {
      #cols <- c(cols, rgb(Fl[l, 1], Fl[l, 2], Fl[l, 3], maxColorValue=max(Fl[l, ])))
      tmpcol <- c(0, 0, 0)
      tmpcol[lblvals[l]] <- 1
      aleph <- Fl[l, lblvals[l]] / ifelse(max(Fl[l, ])==0, 1, max(Fl[l, ]))
      cols <- c(cols, rgb(tmpcol[1], tmpcol[3], tmpcol[2], alpha=aleph))
    }
    #pdf(file.path("/Users/moy/work/git/bb_python/python/temp/diffusion", sprintf("iter_%0d.pdf", i)))
    plot(samples[, 1], samples[, 2], pch="*", cex=2.5, axes=T, xlab=NA, ylab=NA, 
         main=sprintf("Iter %d", i-1), cex.main=1.5,
         col="grey50")
    points(samples[, 1], samples[, 2], pch="*", cex=2.5, col=cols)
    points(samples[y, 1], samples[y, 2], pch="o", cex=2.5, 
           col=colrs[cls])
    #dev.off()
  }
}
dev.off()

