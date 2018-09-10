import torch

x = torch.rand(10, 10)
print(x)
print(x.new(x.shape))
