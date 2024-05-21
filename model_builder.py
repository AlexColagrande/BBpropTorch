"""
Contains PyTorch models
"""

import torch
from torch import nn

import layer_builder
from layer_builder import Linear_BP, Linear_FA, Linear_DFA

########### Linear models with 2 linear Layers ############

# Device-agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearModel_NT(nn.Module):
  """An classic MLP with 2 linear layers"""
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int,
               use_bias: bool=False):
    super().__init__()

    self.W1 = nn.Parameter(torch.randn(size=(hidden_units, input_shape)))
    self.W2 = nn.Parameter(torch.randn(size=(output_shape, hidden_units)))

    if use_bias:
      self.b1 = nn.Parameter(torch.zeros(hidden_units))
      self.b2 = nn.Parameter(torch.zeros(output_shape))
    else:
      self.b1 = self.b2 = None

  def forward(self, x):
    x = torch.mm(x, self.W1.T) + (self.b1 if self.b1 is not None else 0)     # matmul because we use BC
    x = torch.mm(x, self.W2.T) + (self.b2 if self.b2 is not None else 0)
    return x

class LinearModel_BP(nn.Module):
  """An MLP with 2 linear layers ready to be trained with an hand made Backpropagation that works exactly like the classic one"""
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int,
               use_bias: bool=False):
    super().__init__()

    self.W1 = nn.Parameter(torch.randn(size=(hidden_units, input_shape)))
    self.W2 = nn.Parameter(torch.randn(size=(output_shape, hidden_units)))

    if use_bias:
      self.b1 = nn.Parameter(torch.zeros(hidden_units))
      self.b2 = nn.Parameter(torch.zeros(output_shape))
    else:
      self.b1 = self.b2 = None

  def forward(self, x):
    x = layer_builder.Linear_BP.apply(x, self.W1, self.b1)
    x = layer_builder.Linear_BP.apply(x, self.W2, self.b2)
    return x

class LinearModel_FA(nn.Module):
  """An MLP with 2 linear layers ready to be trained with Feedback Alignement"""
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int,
               use_bias: bool=False):
    super().__init__()

    self.W1 = nn.Parameter(torch.randn(size=(hidden_units, input_shape)))
    self.W2 = nn.Parameter(torch.randn(size=(output_shape, hidden_units)))

    if use_bias:
      self.b1 = nn.Parameter(torch.zeros(hidden_units))
      self.b2 = nn.Parameter(torch.zeros(output_shape))
    else:
      self.b1 = self.b2 = None

    # Initializing FA matrices
    self.B1 = torch.randn(size=(input_shape, hidden_units)).to(device)
    self.B2 = torch.randn(size=(hidden_units, output_shape)).to(device)

  def forward(self, x):
    x = Linear_FA.apply(x, self.W1, self.B1, self.b1)
    x = Linear_FA.apply(x, self.W2, self.B2, self.b2)
    return x

class LinearModel_DFA(nn.Module):
  """An MLP with 2 linear layers ready to be trained with Direct Feedback Alignement"""
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int,
               use_bias: bool=False):
    super().__init__()

    self.W1 = nn.Parameter(torch.randn(size=(hidden_units, input_shape)))
    self.W2 = nn.Parameter(torch.randn(size=(output_shape, hidden_units)))

    if use_bias:
      self.b1 = nn.Parameter(torch.zeros(hidden_units))
      self.b2 = nn.Parameter(torch.zeros(output_shape))
    else:
      self.b1 = self.b2 = None

    # Initializing DFA matrices
    self.B1 = torch.randn(size=(input_shape, output_shape)).to(device)
    self.B2 = torch.randn(size=(hidden_units, output_shape)).to(device)
    self.e = []

  def forward(self, x):
    x = Linear_DFA.apply(x, self.W1, self.B1, self.b1, self.e) # apply takes no keyword arguments
    x = Linear_DFA.apply(x, self.W2, self.B2, self.b2, self.e)
    return x
