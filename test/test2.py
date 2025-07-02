import tensor

# Create a dummy input tensor (e.g., 3x1 vector)
x = tensor.Tensor([3, 1], True)
x[0] = 1.0
x[1] = 2.0
x[2] = 3.0
print("Input tensor:")
x.print()

# Create a Linear layer: 2 output features, 3 input features
layer = tensor.Linear(3, 2)

# Forward pass
output = layer.forward(x)

print("Linear layer output:")
output.print()

# Test MSELoss
print("\n--- Testing MSELoss ---")

# Create target tensor
target = tensor.Tensor([2, 1], False)
target[0] = 1.5
target[1] = 0.8
print("Target tensor:")
target.print()

# Create MSE loss function
mse_loss = tensor.MSELoss()

# Calculate loss
loss = mse_loss.forward(output, target)
print("MSE Loss:")
loss.print()

# Test tensor operations used in MSE
print("\n--- Testing tensor operations ---")
diff = output.add(target * -1.0)
print("Difference (output - target):")
diff.print()

squared_diff = diff ^ 2
print("Squared difference:")
squared_diff.print()

sum_val = squared_diff.sum()
print("Sum of squared differences:", sum_val)

mean_val = sum_val / output.size()
print("Mean squared error:", mean_val)
