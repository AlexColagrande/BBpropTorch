"""
Trains PyTorch model using device-agnostic code
"""
import argparse

import sys
import os
import torch
from torch import nn

from timeit import default_timer as Timer

# Append the directory of your script to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/alexcolagrande/Desktop/Python/BGradTorch')))

import data_setup, model_builder, utils, engine

# Parser
parser = argparse.ArgumentParser(description="Train a PyTorch model.")
parser.add_argument("model_name", choices=['Linear_NT', 'Linear_BP', 'Linear_FA', 'Linear_DFA'], help="Model type to train")
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = 300
#BATCH_SIZE = 32 TODO: to add for bigger datasets
HIDDEN_UNITS = 10
LEARNING_RATE = 0.003
random_seed = 42


# Setup data
  # Data parameters
n, m = 3, 2
num_samples = 100

printing = True
print_step = 15

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

# Retrieve from parse and instantiate the model
if args.model_name == 'Linear_NT':
    model = model_builder.LinearModel_NT(input_shape=n, hidden_units=HIDDEN_UNITS, output_shape=m, use_bias=True)
elif args.model_name == 'Linear_BP':
    model = model_builder.LinearModel_BP(input_shape=n, hidden_units=HIDDEN_UNITS, output_shape=m, use_bias=True)
elif args.model_name == 'Linear_FA':
    model = model_builder.LinearModel_FA(input_shape=n, hidden_units=HIDDEN_UNITS, output_shape=m, use_bias=True)
elif args.model_name == 'Linear_DFA':
    model = model_builder.LinearModel_DFA(input_shape=n, hidden_units=HIDDEN_UNITS, output_shape=m, use_bias=True)
else:
    raise ValueError("Invalid model name provided.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Manual seeds for reproducibility (I PUT OR NO???)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Finally we get the data
X_train, y_train, X_test, y_test = data_setup.data_012Vsplit(num_samples = num_samples,
                                                  n = n,
                                                  m = m)

X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)

# Set loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params = model.parameters(),
                            lr = LEARNING_RATE)

engine.train_step(model = model,
                  data = (X_train, y_train),
                  loss_fn = loss_fn,
                  optimizer = optimizer,
                  epochs = NUM_EPOCHS,
                  model_name = args.model_name,
                  printing = printing,
                  print_step = print_step)
