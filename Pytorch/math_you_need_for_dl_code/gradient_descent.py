import torch

# Create parameter θ
theta = torch.tensor(5.0, requires_grad=True) # Enable gradient tracking

# Define hyperparameters
eta = 0.1 # Learning rate
total_steps = 15 # No. of iterations

print("Step 0 (initial θ):", theta.item())
print("Initial loss:", (theta ** 2).item())
print()

# Iterate for total_steps 
for step in range(1, total_steps + 1):
    # Forward pass
    loss = theta ** 2
    
    # Backward pass (calculates d(θ²)/dθ = 2θ and stores in theta.grad)
    loss.backward() 
    
    # Store gradient and loss to print later
    gradient = theta.grad.item()
    loss_value = loss.item()

    # Gradient descent update: θ = θ - η·∇J(θ)
    # Use no_grad() to prevent PyTorch from tracking this parameter update
    with torch.no_grad():
        theta -= eta * theta.grad

    print(f"Step {step}:")
    print(f"loss = {round(loss_value, 3)}")
    print(f"θ = {round(theta.item(), 3)}")
    print(f"gradient = {round(gradient, 3)}")
    print(f"update = {round(eta * gradient, 3)}")

    # Reset gradients for next iteration
    theta.grad.zero_()

print("Final θ:", round(theta.item(), 3))
print("Final loss:", round((theta ** 2).item(), 3))

'''
Output
Step 0 (initial θ): 5.0
Initial loss: 25.0

Step 1:
loss = 25.0
θ = 4.0
gradient = 10.0
update = 1.0

Step 2:
loss = 16.0
θ = 3.2
gradient = 8.0
update = 0.8

Step 3:
loss = 10.24
θ = 2.56
gradient = 6.4
update = 0.64

Step 4:
loss = 6.554
θ = 2.048
gradient = 5.12
update = 0.512

Step 5:
loss = 4.194
θ = 1.638
gradient = 4.096
update = 0.41

Step 6:
loss = 2.684
θ = 1.311
gradient = 3.277
update = 0.328

Step 7:
loss = 1.718
θ = 1.049
gradient = 2.621
update = 0.262

Step 8:
loss = 1.1
θ = 0.839
gradient = 2.097
update = 0.21

Step 9:
loss = 0.704
θ = 0.671
gradient = 1.678
update = 0.168

Step 10:
loss = 0.45
θ = 0.537
gradient = 1.342
update = 0.134

Step 11:
loss = 0.288
θ = 0.429
gradient = 1.074
update = 0.107

Step 12:
loss = 0.184
θ = 0.344
gradient = 0.859
update = 0.086

Step 13:
loss = 0.118
θ = 0.275
gradient = 0.687
update = 0.069

Step 14:
loss = 0.076
θ = 0.22
gradient = 0.55
update = 0.055

Step 15:
loss = 0.048
θ = 0.176
gradient = 0.44
update = 0.044

Final θ: 0.176
Final loss: 0.031
'''
