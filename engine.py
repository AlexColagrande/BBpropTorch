"""
Contains functions for training and testing PyTorch models.
"""

# TODO TO UNDERSTAND IF IT'S USEFUL -> from typing import Dict, List, Tuple
import os
import torch
from tqdm.auto import tqdm
import pickle
from utils import save_dictionary

############# REGRESSION TRAIN & TEST #############
# TODO: AT THE MOMENT STRONGLY TIED WITH THE LINEAR MODELS

def train_step(model: torch.nn.Module,
               data,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               epochs: int,
               model_name: str="",
               printing: bool=True,
               print_step: int=10,
               random_seed: int=42):

  torch.manual_seed(random_seed)
  torch.cuda.manual_seed(random_seed)

  ## Initialize the Dstate_dict
  Dstate_dict = {"W1":[], "W2":[]}
  Dstate_dict["W1"].append(model.state_dict()["W1"].clone().detach())
  Dstate_dict["W2"].append(model.state_dict()["W2"].clone().detach())

  X_train, y_train = data
  ## Training loop
  for epoch in range(epochs):
    model.train()

    # 1. Forward pass
    y_pred = model(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    if model_name in ['Linear_DFA']:
      error_upd = y_pred - y_train
      if len(model.e)==0:
        model.e.append(error_upd)
      else:
        model.e[0] = error_upd

    # 3. Optimizer and zero grad
    optimizer.zero_grad()

    # 4. Backward pass
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # 6. Add W1 and W2 to Dstate_dic
    Dstate_dict["W1"].append(model.state_dict()["W1"].clone().detach())
    Dstate_dict["W2"].append(model.state_dict()["W2"].clone().detach())

    # Print out what's happening
    if printing == True:
      if epoch % print_step == 0:
        print(f"Epoch: {epoch} | Train loss: {loss}")

  model_name = (model.__class__.__name__)[-3:]
  if model_name[0] != "_":
    model_name = "_" + model_name
  Dstate_dict_name = f'Dstate_dict{model_name}.pkl'
  pickles_path = '/Users/alexcolagrande/Desktop/Python/BGradTorch/pickles'
  file_path = os.path.join(pickles_path, Dstate_dict_name)
  save_dictionary(Dstate_dict, file_path)

def test_step(model: torch.nn.Module,
              data,
              loss_fn: torch.nn.Module,
              epochs: int,
              print_step: int=10,
              printing: bool=True,
              random_seed: int=42):
  X_test, y_test = data
  ### Testing
  for epoch in range(epochs):
    model.eval()
    with torch.inference_mode():
      test_pred = model(X_test)
      test_loss = loss_fn(test_pred, y_test)

    # Print out what's happening
    if printing:
      if epoch % print_step == 0:
        print(f"Epoch: {epoch} | Test loss: {test_loss}")
# TODO: def train: PUT TOGHETER TRAIN AND TEST TO HAVE BOTH AT THE SAME TIME
