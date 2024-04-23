# Deep Neural Network Classifier for Multi-dimensional Functional Data
------------------------------------------------

# Functional data pre-processing
- Given functional data $X (s_{1}, \dots, s_{d})$, first use basis functions to extract projection scores $\xi_{1}, \xi_{2}, \dots$ by integration.
-------------------------------------------------------

# Model input and output
- Input: Projection scores $\xi_{1}, \xi_{2}, \dots, \xi_{J}$.
- Output: Binary class label k={-1, 1}.
-------------------------------------------------------------

# Model selection
## Neural network hyperparameters 
- J: number of projection scores for network inputs
- L: number of layers 
- p: neurons per layer (uniform for all layers)
- B: maximal norm of weights
-------------------------------------------------------------

# Other hyperparameters
- Loss function: hinge loss
- Dropout rate: data dependent
- Batch size: data dependent
- Epoch number: data dependent
- Activation function: ReLU
- Optimizer: Adam 
-------------------------------------------------------------

# Function descriptions
## One dimensional functional data
- "dnn_1d_par.R": hyperparameter selection with training data. More details can be found in comments
- "dnn_1d.R": functional deep neural netowrks. More details can be found in comments 
## Two dimensional functional data
- "dnn_2d_par.R": hyperparameter selection with training data. More details can be found in comments
- "dnn_2d.R": functional deep neural netowrks. More details can be found in comments 
-------------------------------------------------------------

# Examples
- "example_1d.R": $X(s) = \sum_{j = 1}^{3} \xi_{j} \psi_{j}(s), \ 0 \le s \le 1$, where $\psi_{1}(s) = \log(s + 2)$, $\psi_{1}(s) = s$, $\psi_{1}(s) = s ^ 3$. Under class k, generate independently $(\xi_{1}, \xi_{2}, \xi_{3}) ^ \mathrm{T} \sim N(\mathbf{\mu}_{k}, \mathbf{\Sigma}_{k})$,  
where $\mathbf{\mu}_{1} = (-1, 2, -3) ^ \mathrm{T}$, $\mathbf{\Sigma}_{1} = \mathrm{diag}(\frac{3}{5}, \frac{2}{5}, \frac{1}{5})$ , $\mathbf{\mu}_{-1} = (-\frac{1}{2}, \frac{5}{2}, -\frac{5}{2}) ^ \mathrm{T}$, $\mathbf{\Sigma}_{-1} = \mathrm{diag}(\frac{9}{10}, \frac{1}{2}, \frac{3}{10})$. 


- "example_2d.R": $X(s_{1}, s_{2}) = \sum_{j = 1}^{4} \xi_{j} \psi_{j}(s_{1}, s_{2}), \ 0 \le s_{1}, s_{2} \le 1$, where $\psi_{1}(s_{1}, s_{2}) = s_{1}s_{2}$, $\psi_{2}(s_{1}, s_{2}) = s_{1}s_{2} ^ 2$, $\psi_{3}(s_{1}, s_{2}) = s_{1} ^ {2} s_{2}$, $\psi_{4}(s_{1}, s_{2}) = s_{1} ^ {2} s_{2} ^ {2}$. Under class k, generate independently $(\xi_{1}, \xi_{2}, \xi_{3}, \xi_{4}) ^ \mathrm{T} \sim N(\mathbf{\mu}_{k}, \mathbf{\Sigma}_{k})$,  
where $\mathbf{\mu}_{1} = (8, -6, 4, -2) ^ \mathbf{T}$, $\mathbf{\Sigma}_{1} = \mathrm{diag}(8, 6, 4, 2)$, $\mathbf{\mu}_{-1} = (-\frac{7}{2}, -\frac{5}{2}, \frac{3}{2}, -\frac{1}{2}) ^ \mathbf{T}$, $\mathbf{\Sigma}_{-1} = \mathrm{diag}(\frac{9}{2}, \frac{7}{2}, \frac{5}{2}, \frac{3}{2})$. 
