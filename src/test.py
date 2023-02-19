# Fichier de test

from evaluator import Evaluator
import torch
from auto_encoder import AutoEncoder
from cifar10_dataset import Cifar10Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Cifar10Dataset(configuration.batch_size, dataset_path, configuration.shuffle_dataset) # Create an instance of CIFAR10 dataset
PATH = 'results\shuffle\model.pth'
model = AutoEncoder()
model.load_state_dict(torch.load(PATH))
model.eval()
Evaluator('cpu', model, dataset)