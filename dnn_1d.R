###one-dimensional FDNN classification
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
#n0.test: testing sample size for group 0
#n1.test: testing sample size for group 1
#M: number of grid points
#S: a vector of all grid points with length M
#J: number of truncated eigenvalues
#D0.train: training data matrix from group 0, n0.train by M matrix
#D1.train: training data matrix from group 0, n1.train by M matrix
#D0.test: testing data matrix from group 0, n0.test by M matrix
#D1.test: testing data matrix from group 0, n1.test by M matrix
#L: length of the DNN
#p: width of the DNN
#B: maximal norm for all nodes
#epoch: epoch number
#batch: batch size
##return
#error: misclassification rate of the testing set

M_dnn=function(D0.train, D1.train ,D0.test, D1.test, n0.train, n1.train, J, M, n0.test, n1.test, S, L, p, B, epoch, batch){
  
  C0.train=matrix(NA, n0.train, J);C1.train=matrix(NA, n1.train, J)
  C0.test=matrix(NA, n0.test, J);C1.test=matrix(NA, n1.test, J)
  
  phi=matrix(NA, M, J)
  for(m in 1:M){
    for(j in 1:J){
      phi[m, j]=Fourier(S[m], M, j)
    }
  }
  
  
  for(i in 1:n0.train){
    for(j in 1:J){
      C0.train[i, j] = mean(D0.train[i,]*phi[j,])
    }
  }
  for(i in 1:n1.train){
    for(j in 1:J){
      C1.train[i, j] = mean(D1.train[i,]*phi[j,])
    }
  }
  for(i in 1:n0.test){
    for(j in 1:J){
      C0.test[i, j] = mean(D0.test[i,]*phi[j,])
    }
  }
  for(i in 1:n1.test){
    for(j in 1:J){
      C1.test[i, j] = mean(D1.test[i,]*phi[j,])
    }
  }
  
  
  
  x_train=rbind(C0.train, C1.train); y_train=c(rep(0, n0.train), rep(1, n1.train))
  x_test=rbind(C0.test, C1.test); y_test=c(rep(0, n0.test), rep(1, n1.test))
  
 
  
  model=keras::keras_model_sequential()
  model %>% keras::layer_dense(units=p, activation = "relu", input_shape = c(J), kernel_initializer = "normal", constraint_maxnorm(max_value = B, axis = 0))%>% 
    layer_dropout(rate = 0.4)
  for(xx in 1:L){
    model %>% keras::layer_dense(units=p, activation = "relu", kernel_initializer = "normal", constraint_maxnorm(max_value = B, axis = 0))%>% 
      layer_dropout(rate = 0.4)
  }
  model %>% keras::layer_dense(units=1,  activation = "relu")
  
  model %>% keras::compile(
    loss="hinge",
    optimizer = 'adam',
    metrics=c('accuracy')
  )
  
  
  history = model %>% keras::fit(
    x_train, y_train,
    epochs=epoch, batch_size=batch
  )
  
  y.pred=model %>% keras::predict_classes(x_test)
  
  
  
  E=1-mean(y.pred==y_test)
  
  
  list(error=E)
}
