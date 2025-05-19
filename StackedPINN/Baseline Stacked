#%% Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#%% Define the base Multi-Layer Perceptron (MLP) class
class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) class with regularization handling.

    --------------
    insize: input size (int)
    outsize: output size (int)
    hsize: A list specifying the number of neurons in each hidden layer (list of ints).
    bias: Boolean indicating whether to include a bias term in the linear layers (default: True)
    nonlin: The activation function to be used in the hidden layers (default: nn.Tanh).
    linear_map:  The type of linear transformation to be used (default: nn.Linear)
    """
    def __init__(self, insize, outsize, hsizes, bias=True, nonlin=nn.Tanh, linear_map=nn.Linear):
        super(MLP, self).__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        self.layers = nn.ModuleList()
        self.nonlin = nn.ModuleList()

        # Create linear layers and activation functions
        layer_sizes = [insize] + hsizes + [outsize]
        for k in range(len(layer_sizes) - 1):
            self.layers.append(linear_map(layer_sizes[k], layer_sizes[k+1], bias=bias))
            if k < self.nhidden:
                self.nonlin.append(nonlin())
            else:
                self.nonlin.append(nn.Identity())

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.nonlin[i](self.layers[i](x))  # Apply the linear layer and activation
        return x

    def reg_error(self):
        """
        Compute the regularization error for the linear layers that support it.
        """
        reg_loss = 0.0
        for layer in self.layers:
            if hasattr(layer, "reg_error"):
                reg_loss += layer.reg_error()
        return reg_loss


# Define the Stacked MLP class for Stacked PINN
class StackedMLP(nn.Module):
    """
    Stacked Multi-Layer Perceptron designed for multi-fidelity learning where multiple layers are
    stacked to refine the prediction progressively. Each layer is a blend of linear and nonlinear transformations
    controlled by an adaptive parameter alpha.
    ---------------------------------------------------------------------
    Attributes:
    insize (int): The dimension of the input features.
    outsize (int): The dimension of the output features.
    h_singlef_size (list of int): Specifies the sizes of hidden layers in the single-fidelity MLP.
    n_stacked_mf_layers (int): Defines the number of multi-fidelity layers that are stacked.
    h_linear_sizes (list of int): Sizes of the hidden layers within each linear sub-network in the multi-fidelity layers.
    bias (bool): Indicates whether bias terms are included in linear transformations.
    nonlin (callable): The nonlinear activation function applied following linear transformations.
    h_nonlinear_sizes (list of int): Sizes of the hidden layers within each nonlinear sub-network in the multi-fidelity layers.
    alpha_init (float): The initial alpha value, which regulates the mix of linear and nonlinear transformations.


    """
    def __init__(
        self,
        insize,
        outsize,
        h_sf_sizes=[20, 20],
        n_stacked_mf_layers=3,
        h_linear_sizes=[10],
        h_nonlinear_sizes=[20, 20],
        bias=True,
        nonlin=nn.Tanh,
        alpha_init=0.1
    ):
        super(StackedMLP, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.h_sf_sizes = h_sf_sizes
        self.n_stacked_mf_layers = n_stacked_mf_layers
        self.h_linear_sizes = h_linear_sizes
        self.h_nonlinear_sizes = h_nonlinear_sizes
        self.bias = bias
        self.nonlin = nonlin
        self.alpha_init = alpha_init

        # Initial single-fidelity MLP
        self.first_layer = MLP(insize, outsize, hsizes=h_sf_sizes, bias=bias, nonlin=nonlin)

        # Alpha parameters
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(alpha_init), requires_grad=True) for _ in range(n_stacked_mf_layers)
        ])

        # Layers
        self.layers = nn.ModuleList()
        for i in range(n_stacked_mf_layers):
            linear_net = MLP(outsize, outsize, hsizes=h_linear_sizes, bias=bias, nonlin=nn.Identity)
            nonlinear_net = MLP(insize + outsize, outsize, hsizes=h_nonlinear_sizes, bias=bias, nonlin=nonlin)
            self.layers.append(nn.ModuleDict({
                'linear': linear_net,
                'nonlinear': nonlinear_net
            }))

    def forward(self, x, i=None):
        if i is None:
            i = self.n_stacked_mf_layers
        i = min(i, self.n_stacked_mf_layers)
        out = self.first_layer(x)
        for j in range(i):
            alpha = self.alpha[j]
            layer = self.layers[j]
            linear_out = layer['linear'](out)
            nonlinear_input = torch.cat([x, out], dim=1)
            nonlinear_out = layer['nonlinear'](nonlinear_input)
            out = torch.abs(alpha) * nonlinear_out + (1 - torch.abs(alpha)) * linear_out
        return out

    def get_alpha_loss(self, i=None):
        """
        Retrieve the accumulated loss from alpha parameters used for regularization purposes.

        :return: Alpha loss as a torch scalar.
        """
        return torch.pow(self.alpha[i], 4)


# Analytical solution for the 1D heat equation
def analytical_solution(x, t, gamma):
    """
    Compute the analytical solution for the 1D heat equation.

    Parameters:
    x (Tensor): Spatial points
    t (Tensor): Temporal points
    gamma (float): Diffusion coefficient

    Returns:
    Tensor: Analytical solution u(x, t)
    """
    return torch.exp(-np.pi**2 * gamma * t) * torch.sin(np.pi * x)

# Define the L2 relative error function
def l2_relative_error(u_pred, u_true):
    """
    Compute L2 relative error between predicted values and true values.

    Parameters:
    u_pred (Tensor): Predicted values (model output)
    u_true (Tensor): True values (ground truth)

    Returns:
    float: L2 relative error
    """
    error = torch.norm(u_pred - u_true, p=2) / torch.norm(u_true, p=2)
    return error.item()

# Define the Stacked PINN class
class StackedPINN:
    def __init__(self, model, optimizer, gamma=0.01, lr_scheduler=None, patience=100, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.logger = logger
        self.n_stacked_layers = model.n_stacked_mf_layers  # Number of stacked layers (Using for averaging loss)

        # Early stopping attributes
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def loss_function(self, x, t):
        x.requires_grad = True
        t.requires_grad = True

        losses = []
        for i in range(self.n_stacked_layers):
            u = self.model(torch.cat([x, t], dim=1), i=i)
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
            f = u_t - self.gamma * u_xx
            losses.append(torch.mean(f ** 2) + self.model.get_alpha_loss(i=i))

        return losses

    def train(self, x_collocation, t_collocation, x_boundary, t_boundary, u_boundary, epochs=1000, x_test=None, t_test=None, u_test=None):
        train_losses = []
        test_losses = []
        l2_errors = []

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Compute physics loss
            physics_loss = self.loss_function(x_collocation, t_collocation)

            # Compute boundary loss
            boundary_loss = []
            for i in range(self.n_stacked_layers):
                u_pred = self.model(torch.cat([x_boundary, t_boundary], dim=1), i=i)
                boundary_loss.append(torch.mean((u_pred - u_boundary) ** 2))

            # Total loss
            loss = (sum(physics_loss) + sum(boundary_loss)) / self.n_stacked_layers

            loss.backward()
            self.optimizer.step()

            # Learning rate scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step(loss)

            # Logging
            if self.logger:
                self.logger.log_metrics({"loss": loss.item()}, step=epoch)

            # Early stopping check
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1

            if self.bad_epochs >= self.patience:
                print("Early stopping triggered.")
                break

            train_losses.append(loss.item())

            # Calculate test loss if test data is provided
            if x_test is not None and t_test is not None and u_test is not None:
                self.model.eval()
                with torch.no_grad():
                    u_pred_test = [self.model(torch.cat([x_test, t_test], dim=1), i) for i in range(self.model.n_stacked_mf_layers)]
                    test_loss = [torch.mean((u_pred_test_i - u_test) ** 2).item() for u_pred_test_i in u_pred_test]
                    test_losses.append(test_loss)

                    # Compute L2 relative error
                    l2_error = [l2_relative_error(u_pred_test_i, u_test) for u_pred_test_i in u_pred_test]
                    l2_errors.append(l2_error)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}, L2 Relative_error : {l2_error}")

        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        if test_losses:
            plt.plot(test_losses, label="Test Loss", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Test Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

         # Plot L2 relative error over epochs
        if l2_errors:  # Check if l2_errors is not empty
            plt.figure()
            plt.plot(range(len(l2_errors)), l2_errors, label=[f"L2 Relative Error{i}" for i in range (3)])
            plt.xlabel("Epochs")
            plt.ylabel("L2 Relative Error")
            plt.title("L2 Relative Error over Training Epochs")
            plt.legend()
            plt.grid(True)
            plt.show()

#%% Main execution to solve the 1D heat equation using the Stacked PINN
if __name__ == "__main__":
    # Parameters
    alpha = 0.01  # Diffusion coefficient
    insize = 2  # Input size (x, t)
    outsize = 1  # Output size (u)
    h_sf_sizes = [40, 40, 40]
    h_linear_sizes = [10]
    h_nonlinear_sizes = [40, 40, 40]
    n_stacked_mf_layers = 3

    # Create model
    model = StackedMLP(insize, outsize, h_sf_sizes=h_sf_sizes, h_linear_sizes=h_linear_sizes,
                       h_nonlinear_sizes=h_nonlinear_sizes, n_stacked_mf_layers=n_stacked_mf_layers)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create collocation points
    N_collocation = 1000
    x_collocation = torch.FloatTensor(N_collocation, 1).uniform_(0, 1)
    t_collocation = torch.FloatTensor(N_collocation, 1).uniform_(0, 1)

    # Boundary conditions (Dirichlet conditions at x=0 and x=1)
    N_boundary = 100
    x_boundary = torch.cat([torch.zeros(N_boundary, 1), torch.ones(N_boundary, 1)], dim=0)
    t_boundary = torch.FloatTensor(2 * N_boundary, 1).uniform_(0, 1)
    u_boundary = torch.zeros(2 * N_boundary, 1)  # Assuming zero Dirichlet boundary conditions

    # Initial condition (u(x, 0) = sin(pi * x))
    N_initial = 100
    x_initial = torch.FloatTensor(N_initial, 1).uniform_(0, 1)
    t_initial = torch.zeros(N_initial, 1)
    u_initial = torch.sin(np.pi * x_initial)  # Initial condition u(x, 0) = sin(pi x)

    # Combine boundary and initial conditions
    x_boundary = torch.cat([x_boundary, x_initial], dim=0)
    t_boundary = torch.cat([t_boundary, t_initial], dim=0)
    u_boundary = torch.cat([u_boundary, u_initial], dim=0)

    # Test points (optional for validation/testing)
    N_test = 200
    x_test = torch.FloatTensor(N_test, 1).uniform_(0, 1)
    t_test = torch.FloatTensor(N_test, 1).uniform_(0, 1)
    u_test = analytical_solution(x_test, t_test, alpha)  # Analytical solution for testing

    # Create instance of StackedPINN
    pinn = StackedPINN(model, optimizer, gamma=0.01)

    # Train the model with test data for loss visualization
    pinn.train(x_collocation, t_collocation, x_boundary, t_boundary, u_boundary, epochs=2000,
               x_test=x_test, t_test=t_test, u_test=u_test)

    #%% Verify that final parameter alpha is small (see Eq. 11 of Howard, Amanda A. et al. 2024).
    for idx,alpha in enumerate(model.alpha):
        print(f"alpha_{idx} = {alpha}")

#%% Compute absolute error
u_pred_test = model(torch.cat([x_test, t_test], dim=1)).detach()
u_true_test = analytical_solution(x_test, t_test, alpha).detach()
absolute_error = torch.abs(u_pred_test - u_true_test).numpy()

plt.figure(figsize=(8, 6))
plt.hist(absolute_error, bins=30, alpha=0.75, edgecolor='black')
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title("Distribution of Absolute Errors")
plt.grid(True)
plt.show()

# %%
