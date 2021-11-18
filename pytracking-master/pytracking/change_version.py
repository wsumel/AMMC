import os
import sys
import torch 

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

state_dict = torch.load('/home/user-njf87/hl/workspace1/pytracking-master/pytracking/networks/ATOM_aug.pth.tar')

torch.save(state_dict,'/home/user-njf87/hl/workspace1/pytracking-master/pytracking/networks/ATOM_aug.pth',_use_new_zipfile_serialization=False)