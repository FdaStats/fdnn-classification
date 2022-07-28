##two-dimensional functional data example
library(keras)
library(tensorflow)
#generate data
n0.train=n1.train=100; n0.test=n1.test=500
#Grid points for two dimensions
S1=seq(0, 1, length.out=50)
S2=seq(0, 1, length.out=50)
#four terms of spectral decomposition
eigen0=c(8, 6, 4, 2); eigen1=c(4.5, 3.5, 2.5, 1.5)
mu0=c(8, -6, 4, -2); mu1=c(-3.5, -2.5, 1.5, -0.5)
#generate projection scores
xi0.train=cbind(rnorm(n0.train, mu0[1], eigen0[1]), rnorm(n0.train, mu0[2], eigen0[2]), rnorm(n0.train, mu0[3], eigen0[3]), rnorm(n0.train, mu0[4], eigen0[4]))
xi1.train=cbind(rnorm(n1.train, mu1[1], eigen1[1]), rnorm(n1.train, mu1[2], eigen1[2]), rnorm(n1.train, mu1[3], eigen1[3]), rnorm(n1.train, mu1[4], eigen1[4]))
xi0.test=cbind(rnorm(n0.test, mu0[1], eigen0[1]), rnorm(n0.test, mu0[2], eigen0[2]), rnorm(n0.test, mu0[3], eigen0[3]), rnorm(n0.test, mu0[4], eigen0[4]))
xi1.test=cbind(rnorm(n1.test, mu1[1], eigen1[1]), rnorm(n1.test, mu1[2], eigen1[2]), rnorm(n1.test, mu1[3], eigen1[3]), rnorm(n1.test, mu1[4], eigen1[4]))
#from tensor to vector
##all data should follow the same procedure to proceed 2d data!
SS1=rep(S1, M2)
SS2=rep(S2, each=M1)
#basis functions
BB1=(SS1)*(SS2); BB2=(SS1)^2*(SS2); BB3=(SS1)*(SS2)^2; BB4=(SS1)^2*(SS2)^2
BB=rbind(BB1, BB2, BB3, BB4)
#generate discretely observed curves
D0.train=as.matrix(xi0.train%*%BB)
D1.train=as.matrix(xi1.train%*%BB)
D0.test=as.matrix(xi0.test%*%BB)
D1.test=as.matrix(xi1.test%*%BB)

#Call M_dnn function
source("dnn_2d_par.R")
source("dnn_2d.R")
#setup candidates for hyperparameters
J1=J2=c(5,10); L=c(2,3); p=c(100,200); B=c(3,5)
#selection for hyperparameters
r1.v=M_dnn.2d.par(D0.train, D1.train, J1, J2, S1, S2, L, p, B, epoch=100, batch=64)
optimal=which(r1.v$error == min(r1.v$error), arr.ind = TRUE)[1,]
J1=J1[optimal[1]]; J2=J2[optimal[2]]; L=L[optimal[3]]; p=p[optimal[4]]; B=B[optimal[5]]
#fit fdnn model
r1=M_dnn.2d(D0.train, D1.train, D0.test, D1.test, J1, J2, S1, S2, L, p, B, epoch, batch)
r1$error

