import torch

# data OR gate (4 options: 00, 01, 10, 11)

x = torch.tensor(
    [[0.,0.],[0,1.],[1,0.],[1,1.]] # outputs (y) = 0,1,1,1 (shape: x: (4, 2))
    )


y = torch.tensor([0., 1., 1., 1.,]).unsqueeze(1) # add new dim (shape: y: (1, 4, 1)

# here nn.Sequential is usef since we are using multiple .nn functions, we can also use just nn.Linear and then nn.Sigmoid seperately (works the same)
model = torch.nn.Sequential(
    torch.nn.Linear(2, 1), # in_features = 2 (like 01), out_features = 1 (like 1)
    torch.nn.Sigmoid(),
)

# using stochastic gradient descent
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
loss_function = torch.nn.MSELoss() # better to use BCELoss() for binary classification

epochs = 1000
for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_function(y_pred, y) # calc mse 
    optimiser.zero_grad() # zero out gradients just to be safe and make sure no residuals form from before and also after first / nth iteration
    loss.backward() # magic!! simply backward pass through the layers to find all gradients wrt weights and biases
    optimiser.step() # this basically performs weight = weight - learning_rate * gradients (updatation)

print("Prediction:", (model(x) > 0.5).int().squeeze().tolist()) 


'''
Running through the output:
> uv run single_neuron.py
Input: [0, 0], [0, 1], [1, 0], [1, 1] 
Prediction: [0, 1, 1, 1] # CORRECT!! 

Model predicts jargon -> backprop to find gradients of weights and biases, update them by taking 0.1 step in the negative direction of the gradient each iteration until loss is minimised.
'''







