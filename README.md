# Deep Neural Network Classifier for Multi-dimensional Functional Data
------------------------------------------------

# Fourier basis
- Given one functional curve ![first equation](https://latex.codecogs.com/gif.latex?X%28t%29), using Fourier basis to extract projection scores ![second equation](https://latex.codecogs.com/gif.latex?%5Cxi_1%2C%20%5Cxi_2%2C%20%5Cldots%2C).
-------------------------------------------------------

# Model input and output
## Input: a set of coordinates e.g. (1/95, 1/79)

### X:
- 
----------------------------------------------------------
## Output: Class label.

### Y:
- 
-------------------------------------------------------------
# Model
## Deep Neural Network with 2 layers
### Model Architecture 
<img src="models/model.png"></img>

### Current Model Hyperparameters 
- Layers: 2
- Neurons per layer: 1000
- Loss Function: Huber
- Number of Epochs: 1000
- Batch Size: 10
- Dropout rate: 0.25
- Activation layer 1: ReLU
- Activation layer 2: ReLU
- Optimizer: SGD - gradient descent

### Desired Model Hyperparameters - ADNI PET Analysis
- Layers: 3
- Neurons per layer: 1000
- Loss Function: Euclidean
- Number of Epochs: 300 or 500
- Batch Size: 32 or 64
- Sparsity: L_1
- Activation layer 1: ReLU
- Activation layer 2: ReLU
- Activation layer 2: ReLU
- Optimizer: Adam
- Image Dimensions: 79x95 pixels
