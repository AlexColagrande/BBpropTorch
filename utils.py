"""
File containing various utility functions for PyTorch model training and testing
"""

import torch
from pathlib import Path
import pickle
"""
Contains utils functions to do different things:
1) Save model and important datas
2) Do an "Angle analysis"
"""

########################## 1) Let's save things ##########################
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

def save_dictionary(dict_obj, file_path):
    """Save the dictionary object to a file using pickle."""
    with open(file_path, 'wb') as file:
        pickle.dump(dict_obj, file)

def load_dictionary(file_path):
    """Load a dictionary object from a pickle file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
########################## 1) "Angles analysis" ##########################
def dot_m(A: torch.Tensor,
        B: torch.Tensor):
    return torch.dot(A.flatten(), B.flatten())

def angle_m(A, B):
    return dot_m(A, B) / (torch.norm(A,p="fro")*torch.norm(B,p="fro"))

def rad_to_deg(radians: float):
    return radians * 180 / torch.pi