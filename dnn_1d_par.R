###one-dimensional FDNN classification: cross validation for hyperparameters selection
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
#S: a vector of all grid points with length M
#J: candidate number of truncated eigenvalues
#D0.train: training data matrix from group 0, n0.train by M matrix
#D1.train: training data matrix from group 0, n1.train by M matrix
#L: candidate length of the DNN
#p: candidate width of the DNN
#B: maximal norm for all nodes
#epoch: epoch number
#batch: batch size
##return
#error: misclassification rate of the testing set

M_dnn.1d.par=function(D0.train, D1.train, J, M, S, L, p, B, epoch, batch){
  M=length(S)
  
  n0.train=dim(D0.train)[1];n1.train=dim(D1.train)[1]
  
  n0.train.cv= floor(0.8*n0.train); n1.train.cv= floor(0.8*n1.train)
  n0.test.cv= n0.train-n0.train.cv; n1.test.cv= n1.train-n1.train.cv
  
  ind0.train=sample(1:n0.train,n0.train.cv); ind1.train=sample(1:n1.train,n1.train.cv)
    
  D0.train.cv=D0.train[ind0.train,]; D1.train.cv=D1.train[ind1.train,]
  D0.test.cv=D0.train[-ind0.train,]; D1.test.cv=D1.train[-ind1.train,]
  
  y_train.cv=c(rep(0, n0.train.cv), rep(1, n1.train.cv))
  y_test.cv=c(rep(0, n0.test.cv), rep(1, n1.test.cv))
  
    
  l1=length(J); l2=length(L); l3=length(p); l4=length(B) 
  
  error=array(NA, c(l1,l2,l3,l4))
  

  for(ii in 1:l1){
    for(jj in 1:l2){
      for(kk in 1:l3){
        for(ll in 1:l4){
          J.cv=J[ii]; L.cv=L[jj]; p.cv=p[kk]; B.cv=B[ll]
          
          
          phi.cv=matrix(NA, M, J.cv)
          for(m in 1:M){
            for(j in 1:J.cv){
              phi.cv[m, j]=Fourier(S[m], M, j)
            }
          }
          
          C0.train.cv=lapply(D0.train.cv, FUN = function(x) (x/M) %*% phi.cv); C1.train.cv=lapply(D1.train.cv, FUN = function(x) (x/M) %*% phi.cv)
          C0.test.cv=lapply(D0.test.cv, FUN = function(x) (x/M) %*% phi.cv); C1.test.cv=lapply(D1.test.cv, FUN = function(x) (x/M) %*% phi.cv)
          
          
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
          
          y.pred.cv=as.numeric(model %>% predict(x_test.cv)>0.5)
          
          
          
          
          E.cv=1-mean(y.pred.cv==y_test.cv)
          
          
          error[ii,jj,kk,ll]=E.cv
         }
        }
      }
    }
  
  
  list(error=error)
}
