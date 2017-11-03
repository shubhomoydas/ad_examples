#rm(list=ls())
library("MASS")

euclidean_dist <- function(x1, x2) {
  v <- x1 - x2
  return (sqrt(sum(v^2)))
}

if (F) {
  source("/Users/moy/work/git/bb_python/R/manifold_learn.R")
  
  write.table(round(1-A, 3), file="/Users/moy/work/git/bb_python/python/temp/a.csv", quote=F, sep=',', row.names=F, col.names=F, append=F)
}

colrs = c("blue", "red", "green", "brown", "magenta")

sample_type = "donut"
# sample_type = "face"


if (sample_type == "donut") {
  samples_path = "/Users/moy/work/git/bb_python/datasets/donut-shape.csv"
  samples <- read.csv(samples_path, header=T)
  samples$label = rep(1, nrow(samples))
  samples$label[nrow(samples)] = 2
  k2 <- 1
  nn <- 10
  a = 0.99
} else {
  samples_path = "/Users/moy/work/git/bb_python/datasets/face.csv"
  samples <- read.csv(samples_path, header=T)
  nanom1 <- 10
  anom1 <- data.frame(cbind(rep(4, nanom1), mvrnorm(n=nanom1, mu=c(110, 100), Sigma=diag(2)*0.2)))
  colnames(anom1) <- colnames(samples)
  samples <- rbind(samples, anom1)
  if (F) {
    tmp = samples
    tmp$label = ifelse(samples$label == 4, 1, 0)
    write.table(tmp, file="/Users/moy/work/git/bb_python/datasets/face_with_anoms.csv", 
                append=F, quote=F, row.names=F, col.names=colnames(tmp), sep=",")
  }
  k2 <- 1
  nn <- 10
  a = 0.99
}

n <- nrow(samples)

center <- c(mean(samples$x), mean(samples$y))
ranges <- c(max(samples$x)-min(samples$x), max(samples$y)-min(samples$y))
normed_samples <- matrix(0, nrow=n, ncol=2)
for (i in 1:n) {
  normed_samples[i, ] <- as.matrix((samples[i, c('x', 'y')] - center) / ranges)
}
head(normed_samples)

dists <- matrix(0, nrow=n, ncol=n)
for (i in 1:n) {
  for (j in i:n) {
    dists[i, j] <- euclidean_dist(normed_samples[i, ], normed_samples[j, ])
    dists[j, i] <- dists[i, j]
  }
}

neighbors <- matrix(0, nrow=n, ncol=nn)
for (i in 1:n) {
  neighbors[i, ] <- order(dists[i, ])[1:nn]
}

W <- matrix(0, nrow=n, ncol=n)
for (i in 1:n) {
  for (j in neighbors[i, ]) {
    W[i, j] <- exp(-euclidean_dist(normed_samples[i, ], normed_samples[j, ])^2/k2)
    W[j, i] <- W[i, j]
  }
}
diag(W) <- 0 # W has 0 on diagonal
# round(W[1:5, 1:5], 3)

D <- diag(apply(W, MAR=c(1), FUN=sum))
# round(D[1:5, 1:5], 3)

iDroot <- diag(sqrt(diag(D))^(-1))
# round(iDroot[1:10, 1:10], 3)

S <- iDroot %*% W %*% iDroot
# round(S[1:10, 1:10], 3)

B <- diag(n) - a*S
# round(B[1:10, 1:10], 3)
A <- solve(diag(n) - a*S)
tdA <- diag(sqrt(diag(A))^(-1))
A <- tdA %*% A %*% tdA
# round(A[1:10, 1:10], 3)

# round(abs(1-A[1:10, 1:10]), 3)

if (T) {
  
  # plot the entire sample data with labeled info
  pdf(sprintf("/Users/moy/work/git/bb_python/python/temp/samples_%s.pdf", sample_type))
  plot(samples$x, samples$y, typ="p", 
       xlim=c(min(samples$x), max(samples$x)), 
       ylim=c(min(samples$y), max(samples$y)), #main="Samples", 
       axes=T, xlab="x", ylab="y",
       pch="*", cex=2.0, col=colrs[samples$label]
      )
  dev.off()
  
  #d <- as.dist(1 - S)
  d <- as.dist((1-A))
  # fit <- cmdscale(d,eig=TRUE, k=2) # k is the number of dim
  fit <- isoMDS(d, k=2) # k is the number of dim
  #fit # view results
  # plot solution
  x <- fit$points[,1]
  y <- fit$points[,2]
  
  if (F) {
    # write the computed coordinates to file so that it can be processed later
    tmp = data.frame(label=ifelse(samples$label==max(samples$label), 1, 0), x=x, y=y)
    # tmp$label = ifelse(samples$label == 4, 1, 0)
    write.table(tmp, file=sprintf("/Users/moy/work/git/bb_python/datasets/%s_with_labeldiffusion.csv", sample_type), 
                append=F, quote=F, row.names=F, col.names=colnames(tmp), sep=",")
  }
  
  pdf(sprintf("/Users/moy/work/git/bb_python/python/temp/samples_%s_labeldiffusion.pdf", sample_type))
  plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2",
       type="n") # , main="Nonmetric MDS")
  points(x, y, pch="*", cex=2.0, col=colrs[samples$label]
         )
  dev.off()
}
