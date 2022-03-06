# Deep Neural Network Classifier for Multi-dimensional Functional Data
------------------------------------------------

# Functional curve pre-processing
- Given one functional curve ![first equation](https://latex.codecogs.com/gif.latex?X%28t%29), first use basis functions to extract projection scores ![second equation](https://latex.codecogs.com/gif.latex?%5Cxi_1%2C%20%5Cxi_2%2C%20%5Cldots) by integration.
-------------------------------------------------------

# Model input and output
- Input: Projection scores ![xi](https://latex.codecogs.com/gif.latex?%5Cxi_1%2C%20%5Cxi_2%2C%20%5Cldots%2C%20%5Cxi_J).
- Output: Class label.
-------------------------------------------------------------

# Model selection
## Cross-validation 
### Model hyperparameters 
- J: number of projection scores
- L: number of layers
- p: neurons per layer (uniform for all layers)
- B: maximal norm of weights
-------------------------------------------------------------

# Other parameters
- Dropout rate: data dependent
- Batch Size: data dependent
- Epoch number: data dependent
- Activation function: ReLU
- Optimizer: Adam 
-------------------------------------------------------------

# Examples
- 
