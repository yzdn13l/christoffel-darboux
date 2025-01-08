import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from itertools import combinations_with_replacement
from utils import cal_loss

def generate_exponents(d, degree):
    """
    Generate all multi-indices with total degree up to 'degree' for d-dimensional points.
    
    Parameters:
        d (int): The dimension of the points.
        degree (int): The maximum degree of the monomials.
    
    Returns:
        ndarray: The multi-indices of shape (num_poly, d).
    """
    num_poly = math.comb(degree + d, d)
    exponents = torch.zeros(num_poly, d, dtype=int)
    i = 0
    for total_degree in range(degree + 1):
        for exps in combinations_with_replacement(range(d), total_degree):
            for var in exps:
                exponents[i, var] += 1
            i += 1
            
    return exponents[1:]

def generate_monomials_sequences_batch(X, exponents):
    """
    Generate monomials given a point cloud and multi-indices.

    Parameters:
        X (ndarray): An array of shape (B, N, d) representing the point cloud.
        exponents (ndarray): The multi-indices of shape (M, d).

    Returns:
        ndarray: Monomial sequences of shape (B, M).
    """
    B, N, d = X.shape
    device = X.device
    exponents = exponents.to(device)
    M = len(exponents)
    # print(f'Number of monomials: {M}') # Number of polynomials: n1 + n2 + ... + n_d = degree; degree + d choose d; d number of dividers for an array in space R^d.
    # monomials = torch.ones(B, N, M, device=device)
    # for i, exp in enumerate(exponents):
    #     monomials[:, :, i] = torch.prod(X ** exp, axis=2) # x1^exp1 * x2^exp2 * ... * xd^expd. e.g. x1^2 * x2^3 * x3^1 \in R^3
    monomials = X.unsqueeze(2).repeat(1, 1, M, 1) ** exponents.unsqueeze(0).unsqueeze(0) # (B, N, M, d) ** (1, 1, M, d) -> (B, N, M, d)
    monomials = monomials.prod(dim=-1) # (B, N, M)
    return monomials.sum(dim=1) / N # (B, N, M) -> (B, M)

def generate_chebyshev_polynomials_sequence_batch(X, exponents):
    """
    Generate Chebyshev polynomials given a point cloud and multi-indices.

    Parameters:
        X (ndarray): An array of shape (B, N, d) representing the d-dimensional point cloud.
        exponents (ndarray): The multi-indices of shape (M, d).

    Returns:
        ndarray: Chebyshev polynomial sequences of shape (B, M).
    """
    B, N, d = X.shape
    device = X.device
    exponents = exponents.to(device)
    cheby_polynomials = torch.cos(exponents.unsqueeze(0).unsqueeze(0) * torch.acos(X).unsqueeze(2)) # (B, N, M)
    cheby_polynomials = cheby_polynomials.prod(dim=-1) # (B, N)
    
    return cheby_polynomials.sum(dim=1) / N # (B, N, M) -> (B, M)

def poly_seq_batch(X, exponents, poly_type='monomial'):
    if poly_type == 'monomial':
        return generate_monomials_sequences_batch(X, exponents)
    elif poly_type == 'chebyshev':
        return generate_chebyshev_polynomials_sequence_batch(X, exponents)
    else:
        raise ValueError('Unknown polynomial type')

class MLPClaasifier(nn.Module):
    def __init__(self, dim_in, layer_dims, dim_out, dropout=0.5):
        super(MLPClaasifier, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(
            nn.Linear(dim_in, layer_dims[0]),
            nn.BatchNorm1d(layer_dims[0]),
            nn.ReLU(),
        ))
        for i in range(1, len(layer_dims)):
            self.convs.append(nn.Sequential(
                nn.Linear(layer_dims[i-1], layer_dims[i]),
                nn.BatchNorm1d(layer_dims[i]),
                nn.Dropout(dropout),
                nn.ReLU(),
            ))
        self.convs.append(nn.Linear(layer_dims[-1], dim_out))

    def forward(self, x):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
        x = self.convs[-1](x)
        return x
    
class MomentNet(nn.Module):
    def __init__(self, dim_in=3, dim_out=40, degree=15, poly_type='chebyshev', cls_type='mlp', layer_dims=[512, 256, 128]):
        super(MomentNet, self).__init__()
        self.poly_type = poly_type
        self.exponts = generate_exponents(dim_in, degree)
        self.num_poly = len(self.exponts) # number of polynomials. First polynomial is 1, so we exclude it.
        if cls_type == 'linear':
            self.cls = nn.Linear(self.num_poly, dim_out)
        elif cls_type == 'mlp':
            self.cls = MLPClaasifier(self.num_poly, layer_dims, dim_out)
        
    def forward(self, x):
        # x: (B, d, N)
        x = x.permute(0, 2, 1)
        x = poly_seq_batch(x, self.exponts, self.poly_type)
        x = self.cls(x)
        return x
    
    def get_loss(self, x, y, loss_type='cross_entropy'):
        logits = self(x)
        if loss_type == 'cross_entropy':
            return cal_loss(logits, y), logits
        else:
            raise ValueError('Unknown loss type')