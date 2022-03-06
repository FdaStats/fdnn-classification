# Deep Neural Network Classifier for Multi-dimensional Functional Data
------------------------------------------------

# Projection scores
- Given one functional curve ![first equation](https://latex.codecogs.com/gif.latex?X%28t%29), using Fourier basis to extract projection scores ![second equation](https://latex.codecogs.com/gif.latex?%5Cxi_1%2C%20%5Cxi_2%2C%20%5Cldots).
-------------------------------------------------------

# Model input and output
## Input: Projection scores ![xi](https://latex.codecogs.com/gif.latex?%5Cxi_1%2C%20%5Cxi_2%2C%20%5Cldots%2C%20%5Cxi_J), ![J](https://latex.codecogs.com/gif.latex?J) is the hyperparameter to choose.
----------------------------------------------------------
## Output: Class label.
-------------------------------------------------------------
# Model selection
## Cross-validation 

### Model hyperparameters 
- L: number of layers
- p: neurons per layer (uniform for all layers)
- s: sparsity parameter (use dropout)
- B: max norm of the network weights
- Batch Size: data dependent
- Epoch number: data 
- Activation function: ReLU
- Optimizer: Adam 
