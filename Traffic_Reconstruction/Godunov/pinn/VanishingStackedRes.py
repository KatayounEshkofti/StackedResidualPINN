import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..config import device
from ..models.stacked_res import StackedRes
from ..utils.resampling import adaptive_resample
from ..utils.l2error import l2_relative_error

############################################################################
#       STACKED PINN 
############################################################################
class StackedPINN:
    '''
    Vanishing stacked-residual PINN for solving partial differential equations.

    This class implements a vanishing stacked residual network architecture 
    for solving hyperbolic PDEs of the form:
        u_t + F(u)*u_x = 0 (using vanishing viscosity mechanism by considering viscosity coefficient: - γ_i*u_xx )


    Parameters:
    -----------
    model_density : StackedRes                      Stacked residual network model
    optimizer : torch.optim.Optimizer               Optimization algorithm (e.g., Adam)
    F_func : callable                               Flux function F(u) in the PDE
    lr_scheduler : torch.optim.lr_scheduler,        Learning rate scheduler
    patience : int, optional (default=1000)         Early stopping patience
    logger : Logger, optional                       Logging utility
    t_min, t_max : float                            Temporal domain boundaries
    x_min, x_max : float                            Spatial domain boundaries
    gamma_init : float                              Initial diffusion coefficient value

    Methods:
    --------
    get_gamma_for_layer(i, p=2): Compute diffusion coefficient γ_i for layer i
    normalized(x, t): Normalize inputs to [0,1] range
    physics_loss(x_colloc, t_colloc): Compute physics-based loss
    data_loss(x_data, t_data, u_data): Compute data mismatch loss

'''

    def __init__(
        self,
        model_density,
        optimizer,
        F_func,
        lr_scheduler=None,
        patience=1000,
        logger=None,
        t_min=0.0,
        t_max=2.0,
        x_min=0.0,
        x_max=5.0, 
        gamma_init = 0.1,
    ):
        self.model_density = model_density
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.F_func = F_func
        self.n_stacked_layers = model_density.n_stacked_rs_layers
        self.logger = logger
        self.patience = patience
        self.best_loss = float('inf')
        self.bad_epochs = 0

        self.t_min, self.t_max = t_min, t_max
        self.x_min, self.x_max = x_min, x_max
        self.gamma_init = gamma_init

        # For counting training iterations
        self.iter_count = 0
        # Print frequency
        self.print_every = 100

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_gamma_for_layer(self, i, p = 2):
        if self.n_stacked_layers == 1:
            return 0.0
        return self.gamma_init * (1 - (i/ (self.n_stacked_layers - 1)**p))  #   Polynomial function for vanishing viscosity

    def normalized(self, x, t): 
        x_norm = (x - self.x_min)/(self.x_max - self.x_min)
        t_norm = (t - self.t_min)/(self.t_max - self.t_min)
        return x_norm, t_norm
    
    # ----------------------------------------------------------------------
    # Method for computing the absolute PDE residual at arbitrary points
    # ----------------------------------------------------------------------
    def compute_residuals(self, x_eval, t_eval, layer_index=None):
        """
        Compute the absolute PDE residual f = u_t + F(u)*u_x - gamma*u_xx
        at the points (x_eval, t_eval).
        
        layer_index: which stacked layer's PDE do we measure? 
                     If None, use the final stacked layer: i = self.n_stacked_layers.
        Returns: |f|, the absolute value of the PDE residual as a 1D tensor.
        """
        if layer_index is None:
            layer_index = self.n_stacked_layers  # final layer
        x_eval = x_eval.to(device)
        t_eval = t_eval.to(device)
        x_eval.requires_grad = True
        t_eval.requires_grad = True

        # normalize inputs
        x_norm, t_norm = self.normalized(x_eval, t_eval)
        inputs = torch.cat([x_norm, t_norm], dim=1)

        # forward pass up to layer_index
        u = self.model_density(inputs, i=layer_index)

        # derivatives
        u_t = torch.autograd.grad(
            u, t_eval, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x_eval, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x_eval, grad_outputs=torch.ones_like(u_x),
            retain_graph=True, create_graph=True
        )[0]

        # PDE: f = u_t + F(u)*u_x - gamma_i*u_xx
        gamma_i = self.get_gamma_for_layer(layer_index - 1)  # index offset
        f = u_t + self.F_func(u)*u_x - gamma_i*u_xx

        # return absolute value 
        return torch.abs(f).detach()

    ########################################################################
    # PDE RESIDUAL
    ########################################################################
    def physics_loss(self, x_colloc, t_colloc):
        """
        PDE: u_t + F(u)*u_x - gamma*u_xx = 0
        We will compute for each layer i. Then apply RBA if use_rba=True.
        """
        x_colloc = x_colloc.to(device)
        t_colloc = t_colloc.to(device)
        x_colloc = x_colloc.detach().clone().requires_grad_(True)
        t_colloc = t_colloc.detach().clone().requires_grad_(True)
        x_norm, t_norm = self.normalized(x_colloc, t_colloc)
        inputs = torch.cat([x_norm, t_norm], dim=1)

        losses_pde = []
        for i in range(self.n_stacked_layers):
            # Compute the PDE residual f_i
            u = self.model_density(inputs, i=i)
            u_t = torch.autograd.grad(u, t_colloc,
                                      grad_outputs=torch.ones_like(u),
                                      retain_graph=True, create_graph=True)[0]
            u_x = torch.autograd.grad(u, x_colloc,
                                      grad_outputs=torch.ones_like(u),
                                      retain_graph=True, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_colloc,
                                       grad_outputs=torch.ones_like(u_x),
                                       retain_graph=True, create_graph=True)[0]
            gamma_i = self.get_gamma_for_layer(i)
            f_i = u_t + self.F_func(u)*u_x - gamma_i*u_xx  # PDE residual
            pde_i = torch.mean(f_i**2) + self.model_density.get_alpha_loss(i=i)
            losses_pde.append(pde_i)

        return losses_pde

    ########################################################################
    # Data MISMATCH
    ########################################################################
    def data_loss(self, x_data, t_data, u_data):
        x_data = x_data.to(device)
        t_data = t_data.to(device)
        u_data = u_data.to(device)

        losses_data = []
        for i in range(self.n_stacked_layers):
            u_pred = self.model_density(torch.cat([x_data, t_data], dim=1), i=i)
            data_i = torch.mean((u_pred - u_data)**2)
            losses_data.append(data_i)
        return losses_data

    ########################################################################
    # TRAIN
    ########################################################################
    def train(self, 
              x_data, t_data, u_data,
              x_colloc, t_colloc,
              x_test=None, t_test=None, u_test=None,
              epochs_adam=5000,
              adaptive_every=500,   # how often to do adaptive resample
              n_new_points=0,
    ):
        """
        x_data, t_data, u_data: measurement data
        x_colloc, t_colloc: initial collocation points
        x_test, t_test, u_test: optional test set for monitoring
        epochs_adam: number of Adam iterations
        adaptive_every: every 'adaptive_every' epochs, call adaptive_resample (Default: without adaptive residual sampling)
        n_new_points: number of new points to add each time (Default: without adaptive residual sampling)
        """
        
        train_losses = []
        test_losses = []
        l2_errors = []

        # Move data to device if not already
        x_data, t_data, u_data = x_data.to(device), t_data.to(device), u_data.to(device)
        x_colloc, t_colloc = x_colloc.to(device), t_colloc.to(device)
        if x_test is not None:
            x_test, t_test, u_test = x_test.to(device), t_test.to(device), u_test.to(device)

        print("\n========== Adam Optimization  ==========\n")
        for epoch in range(epochs_adam):
            self.model_density.train()
            self.optimizer.zero_grad()

            # PDE residual
            pde_losses = self.physics_loss(x_colloc, t_colloc)
            
            # Data mismatch
            data_losses = self.data_loss(x_data, t_data, u_data)
            pde_loss = sum(pde_losses)/self.n_stacked_layers
            data_loss = sum(data_losses)/self.n_stacked_layers
            loss = pde_loss + data_loss

            loss.backward()
            self.optimizer.step()

            # LR scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step(loss)

            # Early stopping
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                print("Early stopping triggered.")
                break

            self.iter_count += 1
            train_losses.append(loss.item())
            
             # Adaptive Resampling step
            if (epoch > 0) and (epoch % adaptive_every == 0):
                x_colloc, t_colloc = self.adaptive_resample(x_colloc, t_colloc, n_new=n_new_points)


            # Evaluate on test data
            if (x_test is not None) and (t_test is not None) and (u_test is not None):
                # if epoch % 100 == 0 or epoch == epochs_adam - 1:
                self.model_density.eval()
                with torch.no_grad():
                    u_pred_test = [
                        self.model_density(torch.cat([x_test, t_test], dim=1), i=i)
                        for i in range(self.n_stacked_layers)
                        ]
                    test_loss_layer = [
                        torch.mean((u_pred_test[i] - u_test)**2).item()
                        for i in range(self.n_stacked_layers)
                        ]
                    test_losses.append(test_loss_layer)
                        # L2 error
                    l2_layer = [
                        l2_relative_error(u_pred_test[i], u_test)
                        for i in range(self.n_stacked_layers)
                        ]
                    l2_errors.append(l2_layer)
            
            # Logging
            if epoch % self.print_every == 0:
                if (l2_errors):
                    # last one
                    l2_layer_current = l2_errors[-1]
                else:
                    l2_layer_current = [0.0]*self.n_stacked_layers
                    
                print(f"Epoch: {epoch} | Loss: {loss.item():.6f} "
                      f"| PDE: {pde_loss.item():.6f} | Data: {data_loss.item():.6f} "
                      f"| L2 Relative_error : {l2_layer_current}")
