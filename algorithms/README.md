# Algorithms

104 from-scratch implementations covering deep learning, classical ML, linear algebra, and attention mechanisms. Most sourced from [Deep-ML](https://www.deep-ml.com/) problem sets.

## attention/

| File | Description |
|------|-------------|
| `layer_norm.py` | Layer normalization with learnable gamma and beta |
| `masked_attention.py` | Scaled dot-product attention with causal masking |
| `mha.py` | Multi-head attention with per-head self-attention |
| `pe.py` | Sinusoidal positional encoding |
| `self_attention.py` | Self-attention mechanism with softmax |

## deeplearning/

| File | Description |
|------|-------------|
| `activation_derivative.py` | Derivatives of sigmoid, tanh, and ReLU |
| `adam.py` | Adam optimizer with bias correction |
| `adam_algo.py` | Adam optimizer with momentum and RMSprop |
| `autograd.py` | Automatic differentiation engine (Value class with computational graph) |
| `batch_normalization.py` | Batch normalization on 4D tensors (BCHW) |
| `bce.py` | Binary cross-entropy loss |
| `cnn.py` | CNN with forward pass and backpropagation |
| `conv2d.py` | 2D convolution layer |
| `cross_entropy.py` | Cross-entropy loss for classification |
| `cross_entropy_derivative.py` | Derivative of cross-entropy w.r.t. logits |
| `dense_layer.py` | Fully connected layer |
| `dense_net_block.py` | DenseNet block with dense connections |
| `dropout.py` | Dropout regularization |
| `global_avg_pool.py` | Global average pooling |
| `gradient_descent.py` | Gradient descent (batch and SGD) |
| `log_softmax.py` | Log-softmax for numerical stability |
| `lstm.py` | Long Short-Term Memory network |
| `mnist.py` | MNIST digit classification (PyTorch) |
| `momentum_update.py` | Momentum optimizer |
| `numerical_gradient.py` | Numerical gradient checking via finite differences |
| `perceptron.py` | Perceptron for binary classification |
| `residual_block.py` | Residual block with skip connection |
| `rnn.py` | Recurrent Neural Network forward pass |
| `sigmoid.py` | Sigmoid activation function |
| `single_neuron.py` | Single neuron with sigmoid activation |
| `single_neuron_with_backprop.py` | Single neuron with backpropagation |
| `softmax.py` | Softmax activation |
| `softmax_derivative.py` | Jacobian matrix of softmax |
| `xor_neural_network.py` | Neural network solving XOR |

## machinelearning/

| File | Description |
|------|-------------|
| `accuracy_score.py` | Accuracy metric |
| `batch_iterator.py` | Mini-batch iterator |
| `calculate_covariance_matrix.py` | Covariance matrix computation |
| `chain_rule.py` | Chain rule for composite derivatives |
| `confusion_matrix.py` | Confusion matrix for binary classification |
| `correlation_matrix.py` | Correlation coefficients between variables |
| `decision_tree.py` | Decision tree using entropy and information gain |
| `derivative.py` | Polynomial derivative |
| `divide_dataset.py` | Dataset splitting by feature threshold |
| `feature_scaling.py` | Feature normalization |
| `fscore.py` | F-score (harmonic mean of precision and recall) |
| `generate_subset.py` | Random subset generation |
| `gradient_direction.py` | Gradient vector magnitude and direction |
| `hessian.py` | Hessian matrix (second-order partial derivatives) |
| `jacobian.py` | Jacobian matrix via numerical differentiation |
| `k_fold_cross_validation.py` | K-fold cross-validation |
| `k_means.py` | K-means clustering |
| `kl_divergence_normal.py` | KL divergence between normal distributions |
| `knn.py` | K-Nearest Neighbors classifier |
| `linear_kernel.py` | Linear kernel (dot product) |
| `linear_regression.py` | Linear regression (normal equation + gradient descent) |
| `logistic_regression.py` | Logistic regression for binary classification |
| `naivebayes.py` | Naive Bayes with Laplace smoothing |
| `partial_derivative.py` | Partial derivatives of multivariable functions |
| `pca.py` | Principal Component Analysis |
| `precision.py` | Precision metric |
| `product_rule.py` | Product rule for polynomial derivatives |
| `quotient_rule.py` | Quotient rule for polynomial derivatives |
| `recall.py` | Recall metric |
| `ridge_regression.py` | Ridge regression with L2 regularization |
| `rmse.py` | Root Mean Squared Error |
| `shuffle_data.py` | Data shuffling |
| `svd.py` | Singular Value Decomposition |
| `svm.py` | Support Vector Machine (Pegasos) |
| `taylor_series.py` | Taylor/Maclaurin series |
| `to_categorical.py` | One-hot encoding |

## math/

| File | Description |
|------|-------------|
| `2dtranslation.py` | 2D point translation |
| `calculate_matrix_mean.py` | Row/column-wise matrix mean |
| `captain_redbeard.py` | Optimization on wavy function |
| `criticalpt.py` | Critical point classification via Hessian |
| `cross_product.py` | 3D vector cross product |
| `eigenval.py` | Eigenvalues for 2x2 matrices |
| `image_matrix.py` | Column space via row echelon form |
| `inverse.py` | 2x2 matrix inverse |
| `inversegeneral.py` | General matrix inverse (cofactor method) |
| `jacobi_method.py` | Jacobi iterative method for linear systems |
| `lagrange_optimise.py` | Lagrange multiplier optimization |
| `laplace.py` | Determinant via Laplace expansion |
| `linear_projection.py` | Orthogonal projection onto line |
| `make_diagonal.py` | Vector to diagonal matrix |
| `matrix_dot_vector.py` | Matrix-vector multiplication |
| `matrixmul.py` | Matrix multiplication |
| `matrixmul_conv.py` | Matrix multiplication as 1x1 convolution |
| `newton_method.py` | Newton's method for optimization |
| `norm.py` | Vector norms (L1, L2, infinity) |
| `ref.py` | Reduced row echelon form |
| `reshape_matrix.py` | Matrix reshape |
| `scalar_multriplication.py` | Scalar-matrix multiplication |
| `svd.py` | Singular Value Decomposition |
| `svd_2x2.py` | SVD for 2x2 matrices |
| `transform_basis.py` | Basis transformation |
| `transpose.py` | Similarity transformation |
| `transpose_matrix.py` | Matrix transpose |
| `vector_sum.py` | Element-wise vector addition |

## labs/

| File | Description |
|------|-------------|
| `loss.py` | Cross-entropy loss with logits |
| `network.py` | TinyNet - CNN with conv and dense layers |
| `optimizer.py` | Mini-SGD with momentum |
| `training_loop.py` | Training loop with validation |
| `training_step.py` | Single training step |
| `transforms.py` | Data augmentation for MNIST |
