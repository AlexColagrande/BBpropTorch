"""
Contains PyTorch layer code to define models
trainable with alternatives of backpropagation
"""
import torch
from torch import autograd

##################### LINEAR LAYERS ######################

class Linear_BP(autograd.Function):
  """
  A modification of a linear layer that is exactly the same of the default one :)
  """
  @staticmethod
  def forward(h, W, b):
    # a_l = W_l h_{l-1} + b_l and
    # Notations: a -> a_{l}, h -> h_{l-1}, W -> W_l, b -> b_l
    a = torch.mm(h, W.T)
    if b is not None:
      a += b.unsqueeze(0)
    return a

  @staticmethod
  def setup_context(ctx, inputs, a):
    """
    inputs: The tuple of all the inputs passed forward
    a: The output of the forward
    """
    h, W, b = inputs
    ctx.save_for_backward(h, W, b)

  @staticmethod
  def backward(ctx, delta_a):
    h, W, b = ctx.saved_tensors
    delta_h = delta_W = delta_b = None

    if ctx.needs_input_grad[0]:
      # \delta h_{l} = W_{l+1}^t \delta a_{l+1}
      delta_h = torch.mm(delta_a, W)
    if ctx.needs_input_grad[1]:
      # \delta W_l = \delta_a_l h_{l-1}^t
      delta_W = torch.mm(delta_a.T, h)
    if b is not None and ctx.needs_input_grad[2]:
      # \delta b = \delta a_l
      delta_b = delta_a.sum(0)

    return delta_h, delta_W, delta_b


class Linear_FA(autograd.Function):
  """
  A modification of a linear layer that performs Feedback Alignement instead of BP in the backward pass
  """
  @staticmethod
  def forward(h, W, B, b):
    # a_l = W_l h_{l-1} + b_l and
    # Notations: a -> a_{l}, h -> h_{l-1}, W -> W_l, b -> b_l
    a = torch.mm(h, W.T)
    if b is not None:
      a += b.unsqueeze(0)
    return a

  @staticmethod
  def setup_context(ctx, inputs, a):
    """
    inputs: The tuple of all the inputs passed forward
    a: The output of the forward
    """
    h, W, B, b = inputs
    ctx.save_for_backward(h, W, B, b)

  @staticmethod
  def backward(ctx, delta_a):
    h, W, B, b = ctx.saved_tensors
    delta_h = delta_W = delta_B = delta_b = None

    if ctx.needs_input_grad[0]:
      # \delta h_{l} = B_l \delta a_{l+1}
      delta_h = torch.mm(delta_a, B.T)
    if ctx.needs_input_grad[1]:
      # \delta W_l = \delta_a_l h_{l-1}^t
      delta_W = torch.mm(delta_a.T, h)
    if b is not None and ctx.needs_input_grad[2]:
      # \delta b = \delta a_l
      delta_b = delta_a.sum(0)

    return delta_h, delta_W, None, delta_b


class Linear_DFA(autograd.Function):
  """
  A modification of a linear layer that performs Direct Feedback Alignement instead of BP in the backward pass
  """
  @staticmethod
  def forward(ctx, h, W, B, b, e):
    # a_l = W_l h_{l-1} + b_l and
    # Notations: a -> a_{l}, h -> h_{l-1}, W -> W_l, b -> b_l
    a = torch.mm(h, W.T)
    if b is not None:
      a += b.unsqueeze(0)

    ctx.save_for_backward(h, W, B, b)

    ctx.e = e
    return a

  @staticmethod
  def backward(ctx, delta_a):
    #input()
    h, W, B, b = ctx.saved_tensors
    delta_h = delta_W = delta_B = delta_b = None

    err = ctx.e
    if ctx.needs_input_grad[0]:
      # \delta h_{l} = B_l e = B_l \delta a_L
      delta_h = torch.mm(err[0], B.T)
    if ctx.needs_input_grad[1]:
      # \delta W_l = \delta_a_l h_{l-1}^t
      delta_W = torch.mm(delta_a.T, h)
    if b is not None and ctx.needs_input_grad[2]:
      # \delta b = \delta a_l
      delta_b = delta_a.sum(0).squeeze(0)

    return delta_h, delta_W, None, delta_b, None
