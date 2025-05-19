#%% Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#%% Define a vanilla MLP for PINN
class MLP(nn.Module):
    """
    Simple feedforward network: [insize] -> hidden layers -> [outsize]
    """
    def __init__(self, insize, outsize, hsizes, bias=True, nonlin=nn.Tanh):
        super(MLP, self).__init__()
        layer_sizes = [insize] + hsizes + [outsize]
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias))
            if i < len(hsizes):
                self.activations.append(nonlin())
            else:
                self.activations.append(nn.Identity())

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x

#%% Trainer for vanilla PINN
class VanillaPINN:
    def __init__(self, model, optimizer, gamma=0.01):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma

    def physics_loss(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        inp = torch.cat([x, t], dim=1)
        u = self.model(inp)
        # PDE residual: u_t - gamma * u_xx = 0
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f = u_t - self.gamma * u_xx
        return torch.mean(f**2)

    def boundary_loss(self, x_b, t_b, u_b):
        inp_b = torch.cat([x_b, t_b], dim=1)
        u_pred = self.model(inp_b)
        return torch.mean((u_pred - u_b)**2)

    def train(self, x_coll, t_coll, x_b, t_b, u_b, epochs=2000):
        losses = []
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()

            loss_pde = self.physics_loss(x_coll, t_coll)
            loss_bc = self.boundary_loss(x_b, t_b, u_b)
            loss = loss_pde + loss_bc

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Total={loss:.2e}, PDE={loss_pde:.2e}, BC={loss_bc:.2e}")

        # Plot loss
        plt.figure(figsize=(8, 5))
        plt.semilogy(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Vanilla PINN Training Loss')
        plt.grid(True)
        plt.show()

#%% Analytical solution and error metric
def analytical_solution(x, t, gamma):
    return torch.exp(-np.pi**2 * gamma * t) * torch.sin(np.pi * x)


def l2_relative_error(u_pred, u_true):
    return torch.norm(u_pred - u_true) / torch.norm(u_true)

#%% Main execution
if __name__ == "__main__":
    # Problem parameters
    gamma = 0.01  # Diffusion Coefficient
    insize = 2   # (x, t)
    outsize = 1  # u
    hsizes = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40]  # hidden layers

    # Instantiate model and trainer
    model = MLP(insize, outsize, hsizes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = VanillaPINN(model, optimizer, gamma)

    # Collocation points (PDE residual)
    N_coll = 1000
    x_coll = torch.rand(N_coll, 1)
    t_coll = torch.rand(N_coll, 1)

    # Boundary conditions (Dirichlet at x=0,1 and initial at t=0)
    N_b = 100
    x_b = torch.cat([torch.zeros(N_b, 1), torch.ones(N_b, 1)], dim=0)
    t_b = torch.rand(2 * N_b, 1)
    u_b = torch.zeros(2 * N_b, 1)
    # initial condition u(x,0) = sin(pi x)
    x_init = torch.rand(N_b, 1)
    t_init = torch.zeros(N_b, 1)
    u_init = torch.sin(np.pi * x_init)

    x_b = torch.cat([x_b, x_init], dim=0)
    t_b = torch.cat([t_b, t_init], dim=0)
    u_b = torch.cat([u_b, u_init], dim=0)

    # Train the PINN
    trainer.train(x_coll, t_coll, x_b, t_b, u_b, epochs=2000)

    # Test performance
    N_test = 200
    x_test = torch.rand(N_test, 1)
    t_test = torch.rand(N_test, 1)
    u_true = analytical_solution(x_test, t_test, gamma)
    model.eval()
    with torch.no_grad():
        u_pred = model(torch.cat([x_test, t_test], dim=1))
    error = l2_relative_error(u_pred, u_true).item()
    print(f"L2 Relative Error (vanilla PINN): {error:.5f}")

    # Plot error distribution
    abs_err = torch.abs(u_pred - u_true).numpy().flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(abs_err, bins=30, edgecolor='k')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    plt.show()

# %%
# Plot error distribution: boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot(abs_err, vert=True)
    plt.ylabel('Absolute Error')
    plt.title('Error Boxplot')
    plt.grid(True)
    plt.show()

# %%
