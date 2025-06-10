#%%###########################################################################
#            DEFINE MLP
############################################################################
import torch.nn as nn

class MLP(nn.Module):
    '''
    Multi-Layer Perceptron (MLP) neural network module with flexible architecture.

    This module implements a fully-connected neural network with configurable:
    - Number of hidden layers
    - Hidden layer sizes
    - Nonlinear activation functions
    - Linear mapping layers 
    Parameters:
    -----------
    insize : int                    Number of input features
    outsize : int                   Number of output units
    hsizes : list[int]              List of hidden layer sizes (determines number of hidden layers)
    bias : bool,                    Whether to include bias parameters
    nonlin : (default=nn.Tanh)      Nonlinear activation function constructor (e.g. nn.ReLU, nn.Sigmoid)
    linear_map :                    Linear layer constructor 

    Forward Pass:
    -------------
    Input: tensor of shape (batch_size, insize)
    Output: tensor of shape (batch_size, outsize)

    Architecture:
    input -> [Linear -> Nonlin] * (len(hsizes)) -> Linear -> Identity -> output

'''

    def __init__(self, insize, outsize, hsizes,
                 bias=True, nonlin=nn.Tanh, linear_map=nn.Linear):
        super(MLP, self).__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        self.layers = nn.ModuleList()
        self.nonlin_list = nn.ModuleList()

        layer_sizes = [insize] + hsizes + [outsize]
        for k in range(len(layer_sizes) - 1):
            self.layers.append(linear_map(layer_sizes[k], layer_sizes[k+1], bias=bias))
            if k < self.nhidden:
                self.nonlin_list.append(nonlin())
            else:
                self.nonlin_list.append(nn.Identity())

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.nonlin_list[i](self.layers[i](x))
        return x

    def reg_error(self):
        return 0.0
