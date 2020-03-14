import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import numpy as np
"""
    Represents a sketched linear layer, which can be used in the same way
    as a torch.nn.Linear layer

    parameters:
        - linear_layer: A torch.nn.Linear layer with trained weights
        - m: The sketching dimension

"""
class SketchedLinear(nn.Module):
    def __init__(self, linear_layer, m = 64, method_id = 'rademacher'):
        super(SketchedLinear, self).__init__()
        self.b = linear_layer.bias
        self.out_dim, self.in_dim = linear_layer.weight.shape
        self.m = m
        if method_id =='rademacher':
            self.method = self.rademacher_mat
        elif method_id == 'gaussian':
            self.method = self.gaussian_mat
        else:
            self.method = self.fast_jl_mat
        self.W_s, self.U = self.sketch_weight(linear_layer.weight, self.m, self.in_dim, self.method)
        self.W_s = nn.Parameter(self.W_s)


    # Expect input size of (batch_size, dim)
    def forward(self, x):
        #W_s, U = self.sketch_weight(self.W, self.m, self.in_dim, self.rademacher_mat)
        sketch_input = torch.matmul(x, self.U.transpose(-2,-1))
        result = torch.matmul(sketch_input, self.W_s.transpose(-2,-1))
        return result  + self.b

    def extra_repr(self):
        return 'in_features={}, out_features={}, m={}, bias={}'.format(self.in_dim, self.out_dim, self.m, True)

    # Construct a random sampling matrix (Used in the fast JL transform)
    def sampling_mat(self,m,n):
        inds = torch.zeros(m, dtype = int).random_(n).unsqueeze(1)
        P = torch.zeros((m,n))
        P = P.scatter(1, inds, 1)
        return P

    def fast_jl_mat(self,m,n):
        bern = Bernoulli(probs = 0.5)
        D = torch.diag(bern.sample([n]) * 2 - 1)
        H = torch.tensor(hadamard(n)).float()
        P = self.sampling_mat(m, n)
        U = P.matmul(H.matmul(D))  / np.sqrt(m)

        return U

    def rademacher_mat(self,m,n):
        bern = Bernoulli(probs = 0.5)
        U = (bern.sample([m,n]) * 2 - 1) / np.sqrt(m)
        return U

    def gaussian_mat(self,m,n):
        # Scaled sketch matrix
        U = torch.randn( (m,n) ) / np.sqrt(m)
        return U
    
    # Output sketched weights
    def sketch_weight(self, W, m, n, method ):
        U = method(m,n).cuda()
        W_s = W.cuda().matmul(U.transpose(-2,-1))
        return (W_s, U)

class SvdLinear(nn.Module):
    def __init__(self, linear_layer, m = 64):
        super(SvdLinear, self).__init__()
        self.b = linear_layer.bias
        self.out_dim, self.in_dim = linear_layer.weight.shape
        self.m = m
        self.W_s, self.U = self.svd_weight(linear_layer.weight, self.m)
        self.W_s = nn.Parameter(self.W_s)
        self.U = nn.Parameter(self.U)
        self.W_s.requires_grad = True
        self.U.requires_grad = False


    # Expect input size of (batch_size, dim)
    def forward(self, x):
        #W_s, U = self.sketch_weight(self.W, self.m, self.in_dim, self.rademacher_mat)
        sketch_input = torch.matmul(x, self.U.transpose(-2,-1))
        result = torch.matmul(sketch_input, self.W_s.transpose(-2,-1))
        return result  + self.b

    def extra_repr(self):
        return 'in_features={}, out_features={}, m={}, bias={}'.format(self.in_dim, self.out_dim, self.m, True)

    # Output sketched weights
    def sketch_weight(self, W, m, n, method ):
        U = method(m,n).cuda()
        W_s = W.cuda().matmul(U.transpose(-2,-1))
        return (W_s, U)

    # Produce low rank approximation with SVD
    def svd_weight(self, W, m):
        U, Sigma, V = torch.svd(W)
        V = V[:,:m].cuda()
        Sigma = Sigma[:m].cuda()
        U = U[:,:m].cuda()
        return (U.matmul(torch.diag_embed(Sigma)), V.transpose(-2,-1))

# Sketch the convolutional network
def sketch_network_conv(model, n_layers= 3, m = 256):
    layers = list(list(model.children())[0].children())

    new_layers = []
    sketched_layers = 0
    for layer in reversed(layers):
        if isinstance(layer, nn.Conv2d) and sketched_layers < n_layers:
            sketched_layers += 1
            new_layers.append(SketchedConv(layer.cpu(), m))
        else:
            new_layers.append(layer.cpu())

    new_layers = list(reversed(new_layers))
    new_model = nn.Sequential(*new_layers)
    return new_model

class SketchedConv(nn.Module):
    def __init__(self, conv_layer, m = 1024):
        super(SketchedConv, self).__init__()
        self.W = conv_layer.weight
        self.bias = conv_layer.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).cuda()
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        # This is a tuple
        self.kernel_size = conv_layer.kernel_size
        self.padding = conv_layer.padding
        self.m = m
        self.method = self.rademacher_mat
        # Flatten the weight for matmul implementation
        self.W_flat = self.W.view(self.out_channels, -1)
        self.W_s, self.U = self.sketch_weight(self.W_flat, m, self.W_flat.shape[-1], self.method)
        self.W_s = nn.Parameter(self.W_s)
        self.U = nn.Parameter(self.U)
        self.W_s.requires_grad = True
        self.U.requires_grad = False

    # Expect input size of (batch_size, dim)
    def forward(self, x):
        batch_size, _, height, width = x.shape
        x_unf = F.unfold(x, kernel_size = self.kernel_size[0], padding = self.padding)

        sketch_input = torch.matmul(self.U, x_unf)
        result = torch.matmul(self.W_s, sketch_input)

        # Reshape the result
        result = result.view(batch_size, self.out_channels, height, width)
        result = result + self.bias
        return result

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, m={}, bias={}'.format(self.in_channels, self.out_channels, self.kernel_size, self.m, True)
        
    # Output sketched weights
    def sketch_weight(self, W, m, n, method):
        U = method(m,n)
        W_s = W.matmul(U.transpose(-2,-1))
        return (W_s, U)


    # Construct a random sampling matrix
    def sampling_mat(self, m, n):
        inds = torch.zeros(m, dtype = int).random_(n).unsqueeze(1)
        P = torch.zeros((m,n))
        P = P.scatter(1, inds, 1)
        return P

    def fast_jl_mat(self,m,n):
        bern = Bernoulli(probs = 0.5)
        D = torch.diag(bern.sample([n]) * 2 - 1)
        H = torch.tensor(hadamard(n)).float()
        P = self.sampling_mat(m, n)
        U = P.matmul(H.matmul(D))  / np.sqrt(m)

        return U

    def rademacher_mat(self,m,n):
        bern = Bernoulli(probs = 0.5)
        U = (bern.sample([m,n]) * 2 - 1) / np.sqrt(m)
        return U

    # Output sketched weights
    def sketch_weight(self, W, m, n, method):
        U = method(m,n)
        W_s = W.matmul(U.transpose(-2,-1))
        return (W_s, U)


# Sketch the linear network
def sketch_network_linear(model, n_layers= 2,  m = 64):
    layers = list(list(model.children())[0].children())

    new_layers = []
    sketched_layers = 0
    for layer in reversed(layers):
        if isinstance(layer, nn.Linear) and sketched_layers < n_layers:
            sketched_layers += 1
            new_layers.append(SketchedLinear(layer.cpu(), m))
        else:
            new_layers.append(layer.cpu())

    new_layers = list(reversed(new_layers))
    new_model = nn.Sequential(*new_layers)
    return new_model

# Sketch the linear network
def svd_network_linear(model, n_layers= 2,  m = 64):
    layers = list(list(model.children())[0].children())

    new_layers = []
    sketched_layers = 0
    for layer in reversed(layers):
        if isinstance(layer, nn.Linear) and sketched_layers < n_layers:
            sketched_layers += 1
            new_layers.append(SvdLinear(layer.cpu(), m))
        else:
            new_layers.append(layer.cpu())

    new_layers = list(reversed(new_layers))
    new_model = nn.Sequential(*new_layers)
    return new_model
