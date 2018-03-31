#rm(list=ls())

# Sample usage draw_circle:
# n <- 100
# r <- 40
# center <- c(50, 100)
# width <- 10
# draw_circle(n, r, center, width)
draw_circle <- function(n, r, center, width) {
  # draw a circle with noise
  rads <- runif(n, min=0, max=2*pi)
  rr <- runif(n, r - (width/2), r + (width/2))
  samples <- matrix(0, nrow=n, ncol=2)
  for (i in 1:n) {
    samples[i, ] <- center + c(rr[i] * cos(rads[i]), rr[i] * sin(rads[i]))
  }
  return (samples)
}

# draw line with noise
draw_line <- function(n, bottomleft, width, height) {
  xs <- runif(n, min=0, max=width)
  ys <- runif(n, min=0, max=height)
  samples <- matrix(0, nrow=n, ncol=2)
  samples[, 1] <- bottomleft[1] + xs
  samples[, 2] <- bottomleft[2] + ys
  return (samples)
}

# generate a 'face' with two eyes and a mouth:
#   O O
#    -
# Each of the elements will be a different class. Add noise

set.seed(42)
n <- 100
e1_samples <- draw_circle(n, 40, c(50, 150), 10)
e2_samples <- draw_circle(n, 40, c(170, 150), 10)
m_samples <- draw_line(n, c(50, 50), 120, 10)

face <- rbind(e1_samples, e2_samples, m_samples)
label <- rep(c(1,2,3), each=n)
facedf <- data.frame(label=label, x=face[, 1], y=face[, 2])
write.table(facedf, file="/Users/moy/work/git/adnotes/data/face.csv", 
            append=F, quote=F, sep=",", 
            row.names=F, col.names=colnames(facedf))

if (F) {
  facedata <- read.csv("/Users/moy/work/git/adnotes/data/face.csv", header=T)
  plot(facedata$x, facedata$y, typ="p", xlim=c(0, 250), ylim=c(0, 250), main="Face", 
       pch=".", col=ifelse(facedata$label==1, "red", ifelse(facedata$label==2, "blue", "green")))
}

if (F) {
  plot(0, typ="n", xlim=c(0, 250), ylim=c(0, 250), main="Face")
  points (e1_samples[, 1], e1_samples[, 2], pch=".", col="red")
  points (e2_samples[, 1], e2_samples[, 2], pch=".", col="red")
  points (m_samples[, 1], m_samples[, 2], pch=".", col="red")
}
