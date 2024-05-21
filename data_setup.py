"""
Contains functionality for creating PyTorch Data
"""

import torch

############# LINEAR DATA ###############

def Vandermonde_matrix(x: torch.Tensor,
                       n: int):
  """
  Returns a Vandermond matrix with n columns
  i.e. a matrix s.t. the i-th column is (x_0^i, ..., x_m^i)
  """
  exponents = torch.arange(end=n, dtype=torch.float32)
  return x.unsqueeze(1) ** exponents.unsqueeze(0) # the magic of broadcasting

def create_linear_data(num_samples: int,
                       M: torch.Tensor,
                       b: torch.Tensor=0):
  """
  Create a batch of num_samples data (x,y) where x is
  sampled from a standard normal distribution and y = Mx
  """
  n = M.shape[1]
  X = torch.randn(size=(num_samples, n))
  y = X @ M.T + b
  return X, y

def data_012Vsplit(num_samples: int,
                               n: int,
                               m: int,
                               b: torch.Tensor=0):
  """
  Create a batch of data from create_linear_data() with M
  a Vandermonde matrix generated by the vector torch.arange(m)
  """
  x = torch.arange(end=m, dtype=torch.float32)
  M = Vandermonde_matrix(x=x, n=n)

  # Inizialize the data
  X, y = create_linear_data(num_samples=100, M=M)

  # Create a train/test split
  train_split = int(0.8 * len(X))
  X_train, y_train = X[:train_split], y[:train_split]
  X_test, y_test = X[train_split:], y[train_split:]

  return X_train, y_train, X_test, y_test