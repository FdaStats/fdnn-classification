###two-dimensional FDNN classification: cross validation for hyperparameters selection
#################################################
##########Fourier basis function#################
#################################################
Fourier=function(s, M, j){
  k=j %/% 2
  
  if(j==1){
    return(1)
  }else if(j %% 2 == 0){
    return(sqrt(2/M)*cos(2*pi*k*s))
  }else if(j %% 2 != 0){
    return(sqrt(2/M)*sin(2*pi*k*s))
  }
}
##input
#n0.train: training sample size for group 0
#n1.train: training sample size for group 1
#M1: number of grid points for the 1st dimension
#M2: number of grid points for the 2nd dimension
#S1: a vector of all grid points with length M1 for the 1st dimension
#S2: a vector of all grid points with length M2 for the 2nd dimension
#J1: candidate number of truncated eigenvalues for the 1st dimension
#J2: candidate number of truncated eigenvalues for the 2nd dimension
#D0.train: training data matrix from group 0, n0.train by M1*M2 matrix
#D1.train: training data matrix from group 0, n1.train by M1*M2 matrix
#L: candidate length of the DNN
#p: candidate width of the DNN
#B: maximal norm for all nodes
#epoch: epoch number
#batch: batch size
##return
#error: misclassification rate of the testing set

M_dnn.2d.par=function(D0.train, D1.train, n0.train, n1.train, J1, J2, M1, M2, S1, S2, L, p, B, epoch, batch){
  
  n0.train.cv= floor(0.8*n0.train)
  n1.train.cv= floor(0.8*n1.train)
  n0.test.cv= n0.train-n0.train.cv
  n1.test.cv= n1.train-n1.train.cv
  
  ind0.train=sample(1:n0.train,n0.train.cv)
  ind1.train=sample(1:n1.train,n1.train.cv)
    
  D0.train.cv=D0.train[ind0.train,]
  D1.train.cv=D1.train[ind1.train,]
  D0.test.cv=D0.train[-ind0.train,]
  D1.test.cv=D1.train[-ind1.train,]
  
  y_train.cv=c(rep(0, n0.train.cv), rep(1, n1.train.cv))
  y_test.cv=c(rep(0, n0.test.cv), rep(1, n1.test.cv))
  
    
  l1.1=length(J1); l1.2=length(J2); l2=length(L); l3=length(p); l4=length(B) 
  
  error=array(NA, c(l1.1,l1.2,l2,l3,l4))
  

for(hh in 1:l1.1){  
  for(ii in 1:l1.2){
    for(jj in 1:l2){
      for(kk in 1:l3){
        for(ll in 1:l4){
          J1.cv=J1[hh]; J2.cv=J2[ii]; L.cv=L[jj]; p.cv=p[kk]; B.cv=B[ll]
          J.cv=J1.cv*J2.cv
          
          C0.train.cv=matrix(NA, n0.train.cv, J.cv);C1.train.cv=matrix(NA, n1.train.cv, J.cv)
          C0.test.cv=matrix(NA, n0.test.cv, J.cv);C1.test.cv=matrix(NA, n1.test.cv, J.cv)
          
          phi1.cv=matrix(NA, M1, J1.cv)
          for(m in 1:M1){
            for(j in 1:J1.cv){
              phi1.cv[m, j]=Fourier(S1[m], M1, j)
            }
          }
          phi2.cv=matrix(NA, M2, J2.cv)
          for(m in 1:M2){
            for(j in 1:J2.cv){
              phi2.cv[m, j]=Fourier(S2[m], M2, j)
            }
          }
          
          phi.cv=kronecker(t(phi2.cv), t(phi1.cv)) 
          
          for(i in 1:n0.train.cv){
            for(j in 1:J.cv){
              C0.train.cv[i, j] = mean(D0.train.cv[i,]*phi.cv[j,])
            }
          }
          for(i in 1:n1.train.cv){
            for(j in 1:J.cv){
              C1.train.cv[i, j] = mean(D1.train.cv[i,]*phi.cv[j,])
            }
          }
          for(i in 1:n0.test.cv){
            for(j in 1:J.cv){
              C0.test.cv[i, j] = mean(D0.test.cv[i,]*phi.cv[j,])
            }
          }
          for(i in 1:n1.test.cv){
            for(j in 1:J.cv){
              C1.test.cv[i, j] = mean(D1.test.cv[i,]*phi.cv[j,])
            }
          }
          
          
          x_train.cv=rbind(C0.train.cv, C1.train.cv); x_test.cv=rbind(C0.test.cv, C1.test.cv)
          
          
          model=keras::keras_model_sequential()
          model %>% keras::layer_dense(units=p.cv, activation = "relu", input_shape = c(J.cv), kernel_initializer = "normal", constraint_maxnorm(max_value = B.cv, axis = 0))%>% 
            layer_dropout(rate = 0.4)
          for(xx in 1:L.cv){
            model %>% keras::layer_dense(units=p.cv, activation = "relu", kernel_initializer = "normal", constraint_maxnorm(max_value = B.cv, axis = 0))%>% 
              layer_dropout(rate = 0.4)
          }
          model %>% keras::layer_dense(units=1,  activation = "relu")
          
          model %>% keras::compile(
            loss="hinge",
            optimizer = 'adam',
            metrics=c('accuracy')
          )
          
          
          history = model %>% keras::fit(
            x_train.cv, y_train.cv,
            epochs=epoch, batch_size=batch
          )
          
          y.pred.cv=model %>% predict(x_test.cv) %>% `>`(0.5) %>% k_cast("int32")
          
          
          
          
          E.cv=1-mean(y.pred.cv==y_test.cv)
          
          
          error[hh,ii,jj,kk,ll]=E.cv
         }
        }
      }
    }
  }
  
  list(error=error)
}
