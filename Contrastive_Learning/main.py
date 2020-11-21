import numpy as np
import torch
import torch.distributions as distributions
import matplotlib.pyplot as plt

mu1 = torch.tensor([0.0, 0.0])
sigma1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
normal_dist_1 = distributions.MultivariateNormal(mu1, sigma1)

mu2 = torch.tensor([1.0, 1.0])
sigma2 = torch.tensor([[1.5, 1.0], [1.0, 1.5]])
normal_dist_2 = distributions.MultivariateNormal(mu2, sigma2)

mu3 = torch.tensor([-1.0, -2.0])
sigma3 = torch.tensor([[1.5, -1.0], [-1.0, 1.5]])
normal_dist_3 = distributions.MultivariateNormal(mu3, sigma3)

p1, p2, p3 = 0.5, 0.3, 0.2
x = torch.linspace(-3., 3., steps = 100)
y = torch.linspace(-3., 3., steps = 100)

grid_x, grid_y = torch.meshgrid(x, y)
grid_xy = torch.stack([grid_x, grid_y], dim = -1)



