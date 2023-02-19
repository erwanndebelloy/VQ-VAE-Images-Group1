# Fichier de test

from evaluator import Evaluator
import torch
from auto_encoder import AutoEncoder
PATH = 'results\shuffle\model.pth'
model = AutoEncoder()
model.load_state_dict(torch.load(PATH))
model.eval()