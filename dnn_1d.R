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

M_dnn.1d=function(D0.train, D1.train, D0.test, D1.test, J, S, L, p, B, epoch, batch){
  
  phi=matrix(NA, M, J)
  for(m in 1:M){
    for(j in 1:J){
      phi[m, j]=Fourier(S[m], M, j)
    }
  }
  
  n0.train=dim(D0.train)[1];n1.train=dim(D1.train)[1];n0.test=dim(D0.test)[1];n1.test=dim(D1.test)[1]
  
  C0.train=lapply(D0.train, FUN = function(x) (x/M) %*% phi)
  C1.train=lapply(D1.train, FUN = function(x) (x/M) %*% phi)
  C0.test=lapply(D0.test, FUN = function(x) (x/M) %*% phi)
  C1.test=lapply(D1.test, FUN = function(x) (x/M) %*% phi)
  
  
  
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
  
  y.pred=as.numeric(model %>% predict(x_test)>0.5)
  
  
  E=1-mean(y.pred==y_test)
  
  
  list(error=E)
}
