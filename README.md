# Deep Neural Network Classifier for Multi-dimensional Functional Data
------------------------------------------------

# Functional data pre-processing
- Given functional data ![first equation](https://latex.codecogs.com/gif.latex?X%28s_1%2C%20%5Cldots%2C%20s_d%29), first use basis functions to extract projection scores ![second equation](https://latex.codecogs.com/gif.latex?%5Cxi_1%2C%20%5Cxi_2%2C%20%5Cldots) by integration.
-------------------------------------------------------

# Model input and output
- Input: Projection scores ![xi](https://latex.codecogs.com/gif.latex?%5Cxi_1%2C%20%5Cxi_2%2C%20%5Cldots%2C%20%5Cxi_J).
- Output: Class label.
-------------------------------------------------------------

# Model selection
## Model hyperparameters 
- J: number of projection scores
- L: number of layers
- p: neurons per layer (uniform for all layers)
- B: maximal norm of weights
-------------------------------------------------------------

# Other parameters
- Loss: hinge
- Dropout rate: data dependent
- Batch size: data dependent
- Epoch number: data dependent
- Activation function: ReLU
- Optimizer: Adam 
-------------------------------------------------------------

# Function descriptions
## One dimensional functional data
- "dnn_1d_par.R": parameter selection using the training data. More details can be found in comments
- "dnn_1d.R": functional deep neural netowrks method. More details can be found in comments 
## Two dimensional functional data
- "dnn_2d_par.R": parameter selection using the training data. More details can be found in comments
- "dnn_2d.R": functional deep neural netowrks method. More details can be found in comments 
-------------------------------------------------------------

# Examples
- "example_2d.R": ![f](https://latex.codecogs.com/gif.latex?X%28s_1%2Cs_2%29%3D%20%5Csum_%7Bj%3D1%7D%5E%7B4%7D%20%5Cxi_%7Bj%7D%5Cpsi_j%28s_1%2Cs_2%29), ![range](https://latex.codecogs.com/gif.latex?0%5Cle%20s_1%2Cs_2%5Cle1), where ![psi1](https://latex.codecogs.com/gif.latex?%5Cpsi_1%28s_1%2C%20s_2%29%3Ds_1s_2), ![psi2](https://latex.codecogs.com/gif.latex?%5Cpsi_2%28s_1%2C%20s_2%29%3Ds_1s_2%5E2), ![psi3](https://latex.codecogs.com/gif.latex?%5Cpsi_3%28s_1%2C%20s_2%29%3Ds_1%5E2s_2), ![psi4](https://latex.codecogs.com/gif.latex?%5Cpsi_4%28s_1%2C%20s_2%29%3Ds_1%5E2s_2%5E2). Under class k, generate independently ![dis](https://latex.codecogs.com/gif.latex?%28%5Cxi_1%2C%5Cxi_2%2C%5Cxi_3%2C%5Cxi_4%29%5E%7B%5Ctop%7D%5Csim%20N%28%5Cpmb%7B%5Cmu%7D_k%2C%5Cpmb%7B%5CSigma%7D_k%29),  
where ![mu1](https://latex.codecogs.com/gif.latex?%5Cpmb%5Cmu_1%3D%288%2C-6%2C4%2C-2%29%5E%5Ctop), ![sigma1](https://latex.codecogs.com/gif.latex?%5Cpmb%5CSigma_1%3D%20%5Ctext%7Bdiag%7D%5Cleft%28%208%2C%206%2C%204%2C%202%5Cright%29),  ![mu2](https://latex.codecogs.com/gif.latex?%5Cpmb%5Cmu_%7B-1%7D%3D%20%5Cleft%28-%5Cfrac%7B7%7D%7B2%7D%2C%20-%5Cfrac%7B5%7D%7B2%7D%2C%20%5Cfrac%7B3%7D%7B2%7D%2C%20-%5Cfrac%7B1%7D%7B2%7D%5Cright%29%5E%5Ctop),  ![sigma2](https://latex.codecogs.com/gif.latex?%5Cpmb%5CSigma_%7B-1%7D%3D%5Ctext%7Bdiag%7D%5Cleft%28%20%5Cfrac%7B9%7D%7B2%7D%2C%20%5Cfrac%7B7%7D%7B2%7D%2C%20%5Cfrac%7B5%7D%7B2%7D%2C%20%5Cfrac%7B3%7D%7B2%7D%5Cright%29). 
