import torch
import inferno_relu

x = torch.tensor([-1.0, 2.5, -0.3, 4.2, 0.0, -2.2, 1.1], device='cuda')
y = torch.empty_like(x)

inferno_relu.relu(x, y)

print("Input: ", x)
print("Output:", y)