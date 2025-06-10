import torch
import torch.nn as nn
from .mlp import MLP

'''
    Residual Network Block (ResNetBlock) for iterative refinement of outputs.

    This module implements a residual block that refines an existing output by adding
    a learned residual term. The residual is computed by an MLP that takes both
    the original input and the previous output as inputs.

    Parameters:
    -----------
    in_dim_x : int              Dimension of the input feature vector x
    in_dim_out : int            Dimension of the output vector (and previous state out_old)
    hres_sizes : list[int]      Hidden layer sizes for the residual MLP
    bias : bool                 Whether to use bias parameters in linear layers
    nonlin : (default=nn.Tanh)  Nonlinear activation function constructor

    Forward Pass:
    -------------
    Inputs:
        x : torch.Tensor, shape (batch_size, in_dim_x)           Input features
        out_old : torch.Tensor, shape (batch_size, in_dim_out)   Previous output state to be refined
        alpha : float or torch.Tensor                            Scaling factor for the residual update (absolute value used)

    Output:
        out_new : torch.Tensor, shape (batch_size, in_dim_out)   Refined output state

    Architecture:
        residual_input = concatenate(x, out_old)
        residual = MLP(residual_input)
        out_new = out_old + |alpha| * residual

    Example:
    >>> block = ResNetBlock(in_dim_x=10, in_dim_out=5)
    >>> x = torch.randn(32, 10)
    >>> out_old = torch.zeros(32, 5)
    >>> out_new = block(x, out_old, alpha=0.1)
'''

class ResNetBlock(nn.Module):
    def __init__(self, in_dim_x, in_dim_out, hres_sizes=[20, 20], bias=True, nonlin=nn.Tanh):
        super(ResNetBlock, self).__init__()
        self.res_mlp = MLP(in_dim_x + in_dim_out,
                           in_dim_out,
                           hsizes=hres_sizes,
                           bias=bias,
                           nonlin=nonlin)

    def forward(self, x, out_old, alpha):
        residual_input = torch.cat([x, out_old], dim=1)
        residual = self.res_mlp(residual_input)
        # Out_new = out_old + alpha * F([x, out_old])
        out_new = out_old + torch.abs(alpha)*residual
        return out_new
