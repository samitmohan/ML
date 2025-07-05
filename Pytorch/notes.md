# pytorch tutorial cuz why not

[article](https://sebastianraschka.com/teaching/pytorch-1h/)


Core libraries:
Tensor: array lib for efficient computing
- extends numpy (adds support for accelerated compute on gpus)

Autograd: diffrentiate computation automatically (backprop)
- gradients for tensor operations

Deep Learning: makes use of tensor and autograd
- loss fns, optimizers, pretrained models


LLMs are also a type of deep neural network, and PyTorch is a deep learning library. 

# Tensors
- scalar is 0d tensor, vector is 1d tensor, matrix is 2d tensor
- they hold multi-dimensional data, where each dimension represents a different feature

Why float32 and int64? A 32-bit floating point number offers sufficient precision for most deep learning tasks, while consuming less memory and computational resources than a 64-bit floating point number.
- Readily change precision using tensort1d.to(torch.float32)

# Autograd
-By default, PyTorch destroys the computation graph after calculating the gradients to free memory. However, since we are going to reuse this computation graph shortly, we set retain_graph=True so that it stays in memory.

- For instance, we can call .backward on the loss, and PyTorch will compute the gradients of all the leaf nodes in the graph, which will be stored via the tensors’ .grad attributes: instead of calling for looping grad and setting retain_graph=True for all tensors

# Create neural net.
- When implementing a neural network in PyTorch, we typically subclass the torch.nn.Module class to define our own custom network architecture. This Module base class provides a lot of functionality, making it easier to build and train models.
- We define the network layers in the __init__ constructor and specify how they interact in the forward method. The forward method describes how the input data passes through the network and comes together as a computation graph.

### Model
NeuralNetwork(
  (layers): Sequential(
    (0): Linear(in_features=50, out_features=30, bias=True)
    (1): ReLU()
    (2): Linear(in_features=30, out_features=20, bias=True)
    (3): ReLU()
    (4): Linear(in_features=20, out_features=3, bias=True)
  )
)

- A linear layer multiplies the inputs with a weight matrix and adds a bias vector. This is sometimes also referred to as a feedforward or fully connected layer.

- we want to keep using small random numbers as initial values for our layer weights, we can make the random number initialization reproducible by seeding PyTorch’s random number generator via manual_seed:

- grad_fn=<AddmmBackward0> means that the tensor we are inspecting was created via a matrix multiplication and addition operation
- Addmm stands for matrix multiplication (mm) followed by an addition (Add).
- If we just want to use a network without training or backpropagation, for example, if we use it for prediction after training, constructing this computational graph for backpropagation can be wasteful as it performs unnecessary computations and consumes additional memory. So, when we use a model for inference (for instance, making predictions) rather than training, it is a best practice to use the torch.no_grad()

- In PyTorch, it’s common practice to code models such that they return the outputs of the last layer (logits) without passing them to a nonlinear activation function. (So for inference/computation class membership: use softmax fn)

# Setting up efficient data loaders

- What we iterate over while training a the model.

Custom dataset class: defines how individual data records are loaded. -> instantiate -> training dataset -> DataLoader class -> instantiate -> training dataloader & test dataset -> test dataloader
Each dataset object is fed to a data loader -> each dataloader object handles dataset shuffling, assembling data records into batches etc...


### num_workers

- This parameter in PyTorch’s DataLoader function is crucial for parallelizing data loading and preprocessing. When num_workers is set to 0, the data loading will be done in the main process and not in separate worker processes. This might seem unproblematic, but it can lead to significant slowdowns during model training when we train larger networks on a GPU.
-  In contrast, when num_workers is set to a number greater than zero, multiple worker processes are launched to load data in parallel, freeing the main process to focus on training your model and better utilizing your system’s resources


- Loading data without multiple workers (setting `num_workers=0`) will create a data loading bottleneck where the model sits idle until the next batch is loaded as illustrated in the left subpanel. If multiple workers are enabled, the data loader can already queue up the next batch in the background as shown in the right subpanel.

- However, if we are working with very small datasets, setting num_workers to 1 or larger may not be necessary since the total training time takes only fractions of a second anyway.



### validation set
- we often use a third dataset, a so-called validation dataset, to find the optimal hyperparameter settings. A validation dataset is similar to a test set. However, while we only want to use a test set precisely once to avoid biasing the evaluation, we usually use the validation set multiple times to tweak the model settings.

- We also introduced new settings called model.train() and model.eval(). As these names imply, these settings are used to put the model into a training and an evaluation mode. This is necessary for components that behave differently during training and inference, such as dropout or batch normalization layers.

-  It is important to include an optimizer.zero_grad() call in each update round to reset the gradients to zero. Otherwise, the gradients will accumulate, which may be undesired.