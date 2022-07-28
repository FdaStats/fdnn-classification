##one-dimensional functional data example
library(keras)
library(tensorflow)
#generate data
n0.train=n1.train=100; n0.test=n1.test=500
#grid points
S=seq(0, 1, length.out=50)
#three terms of spectral decomposition
eigen0=c(3/5, 2/5, 1/5); eigen1=c(9/10, 1/2, 3/10)#variance
mu0=c(-1,2,-3); mu1=c(-1/2, 5/2, -5/2)#mean
#generate projection scroes
xi0.train=cbind(rnorm(n0.train, mu0[1], eigen0[1]), rnorm(n0.train, mu0[2], eigen0[2]), rnorm(n0.train, mu0[3], eigen0[3]))
xi1.train=cbind(rnorm(n1.train, mu1[1], eigen1[1]), rnorm(n1.train, mu1[2], eigen1[2]), rnorm(n1.train, mu1[3], eigen1[3]))
xi0.test=cbind(rnorm(n0.test, mu0[1], eigen0[1]), rnorm(n0.test, mu0[2], eigen0[2]), rnorm(n0.test, mu0[3], eigen0[3]))
xi1.test=cbind(rnorm(n1.test, mu1[1], eigen1[1]), rnorm(n1.test, mu1[2], eigen1[2]), rnorm(n1.test, mu1[3], eigen1[3]))
#basis functions
BB1=log10(S+2); BB2=S; BB3=S^3
BB=rbind(BB1, BB2, BB3)
#generate discretely observed curves
D0.train=as.matrix(xi0.train%*%BB)
D1.train=as.matrix(xi1.train%*%BB)
D0.test=as.matrix(xi0.test%*%BB)
D1.test=as.matrix(xi1.test%*%BB)


#Call M_dnn function
source("dnn_1d_par.R")
source("dnn_1d.R")
#setup candidates
J=c(10,20); L=c(2,3); p=c(100,200); B=c(3,5)
#selection for hyperparameters
r1.v=M_dnn.1d.par(D0.train, D1.train, J, M, S, L, p, B, epoch=100, batch=64)
#extract hyperparameters
optimal=which(r1.v$error == min(r1.v$error), arr.ind = TRUE)[1,]
J=J[optimal[1]]; L=L[optimal[2]]; p=p[optimal[3]]; B=B[optimal[4]]
#fit fdnn model
r1=M_dnn.1d(D0.train, D1.train ,D0.test, D1.test, J, S, L, p, B, epoch, batch)
#demonstrate error
r1$error

