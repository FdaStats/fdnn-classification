##two-dimensional functional data example
library(keras)
library(tensorflow)
#Generate data
#M is sampling freqeuncy
M=50; n0.train=n1.train=100; n0.test=n1.test=500


S=seq(0, 1, length.out=M)
#3 terms of spectral decomposition
NN=3
#
eigen0=c(3/5, 2/5, 1/5); eigen1=c(9/10, 1/2, 3/10)
mu0=c(-1,2,-3); mu1=c(-1/2, 5/2, -5/2)


xi0.train=cbind(rnorm(n0.train, mu0[1], eigen0[1]), rnorm(n0.train, mu0[2], eigen0[2]), rnorm(n0.train, mu0[3], eigen0[3]))
xi1.train=cbind(rnorm(n1.train, mu1[1], eigen1[1]), rnorm(n1.train, mu1[2], eigen1[2]), rnorm(n1.train, mu1[3], eigen1[3]))
xi0.test=cbind(rnorm(n0.test, mu0[1], eigen0[1]), rnorm(n0.test, mu0[2], eigen0[2]), rnorm(n0.test, mu0[3], eigen0[3]))
xi1.test=cbind(rnorm(n1.test, mu1[1], eigen1[1]), rnorm(n1.test, mu1[2], eigen1[2]), rnorm(n1.test, mu1[3], eigen1[3]))




BB1=log10(S+2); BB2=S; BB3=S^3


BB=rbind(BB1, BB2, BB3)

D0.train=as.matrix(xi0.train%*%BB)
D1.train=as.matrix(xi1.train%*%BB)
D0.test=as.matrix(xi0.test%*%BB)
D1.test=as.matrix(xi1.test%*%BB)


#Call M_dnn function
source("dnn_1d_par.R")
source("dnn_1d.R")
J=c(10,20); L=c(2,3); p=c(100,200); B=c(3,5)
epoch=100; batch=64
r1.cv=M_dnn.1d.par(D0.train, D1.train, n0.train, n1.train, J, M, S, L, p, B, epoch, batch)
optimal=which(r1.cv$error == min(r1.cv$error), arr.ind = TRUE)[1,]
J=optimal[1]; L=optimal[2]; p=optimal[3]; B=optimal[4] 
r1=M_dnn.1d(D0.train, D1.train ,D0.test, D1.test, n0.train, n1.train, J, M, n0.test, n1.test, S, L, p, B, epoch, batch)
r1$error

