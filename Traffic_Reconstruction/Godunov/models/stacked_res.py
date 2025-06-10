import torch
from .resblock import ResNetBlock 
import torch.nn as nn
from .mlp import MLP


class StackedRes(nn.Module):
    
    '''
    Stacked Residual Network (StackedRes) for iterative output refinement.

    This module implements a neural network architecture that combines:
    1. An initial feature transformation (MLP)
    2. Multiple residual refinement blocks (ResNetBlock)
    3. Learnable scaling parameters (alpha) for each refinement step

    Parameters:
    -----------
    insize : int                Input feature dimension
    outsize : int               Output dimension
    h_sf_sizes : list[int]      Hidden layer sizes for the initial MLP (state function)
    n_stacked_rs_layers : int   Number of residual refinement blocks
    h_res_sizes : list[int]     Hidden layer sizes for each residual MLP
    bias : bool                 Whether to include bias parameters
    nonlin : (default=nn.Tanh)  Nonlinear activation function constructor
    alpha_init : (default=0.1)  Initial value for learnable scaling parameters

    Forward Pass:
    -------------
    Inputs:
        x : torch.Tensor, shape (batch_size, insize)       Input features
        i : int or None, optional                          Number of residual blocks to execute (default: all blocks)

    Output:
        out : torch.Tensor, shape (batch_size, outsize)    Refined output after i refinement steps

    Architecture:
    1. Initial transformation: out0 = MLP(x)
    2. For k in [0, i-1]:
    out_{k+1} = out_k + |alpha_k| * ResNetBlock(x, out_k)

    Methods:
    --------
    get_alpha_loss(i: int) ->       Returns regularization loss for i-th alpha parameter (alpha[i]^4)

    Example:
    >>> model = StackedRes(insize=10, outsize=2, n_stacked_rs_layers=4)
    >>> x = torch.randn(32, 10)
    >>> # Full forward pass through all blocks
    >>> y_full = model(x)
    >>> # Partial forward pass (first 2 blocks only)
    >>> y_partial = model(x, i=2)
    >>> # Get regularization loss for first alpha
    >>> alpha_loss = model.get_alpha_loss(0)
    '''
    def __init__(self,
                 insize,
                 outsize,
                 h_sf_sizes=[40, 40],
                 n_stacked_rs_layers=3,
                 h_res_sizes=[40, 40],
                 bias=True,
                 nonlin=nn.Tanh,
                 alpha_init=0.1):
        super(StackedRes, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.n_stacked_mf_layers = n_stacked_rs_layers
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.tensor(alpha_init), requires_grad=True)
             for _ in range(n_stacked_rs_layers)]
        )

        self.first_layer = MLP(insize, outsize, hsizes=h_sf_sizes, bias=bias, nonlin=nonlin)

        self.layers = nn.ModuleList()
        for _ in range(n_stacked_rs_layers):
            block = ResNetBlock(insize, outsize, hres_sizes=h_res_sizes, bias=bias, nonlin=nonlin)
            self.layers.append(block)

    def forward(self, x, i=None):
        if i is None:
            i = self.n_stacked_mf_layers
        i = min(i, self.n_stacked_mf_layers)
        out = self.first_layer(x)
        for j in range(i):
            out = self.layers[j](x, out, self.alpha[j])
        return out

    def get_alpha_loss(self, i=None):
        return torch.pow(self.alpha[i], 4)

def l2_relative_error(u_pred, u_true):
    err = torch.norm(u_pred - u_true, p=2)
    denom = torch.norm(u_true, p=2)
    if denom.item() < 1e-12:
        return err.item()
    return (err/denom).item()