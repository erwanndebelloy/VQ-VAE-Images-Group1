# Fichier de test

from evaluator import Evaluator
import torch
from auto_encoder import AutoEncoder
from cifar10_dataset import Cifar10Dataset
from configuration import Configuration
import os
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


PATH =r'C:\Users\baudo\Documents\ML-DATA\VQ-VAE-Images-Group1\results\shuffled\model.pth'



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', nargs='?', default=Configuration.default_batch_size, type=int, help='The size of the batch during training')
parser.add_argument('--num_training_updates', nargs='?', default=Configuration.default_num_training_updates, type=int, help='The number of updates during training')
parser.add_argument('--num_hiddens', nargs='?', default=Configuration.default_num_hiddens, type=int, help='The number of hidden neurons in each layer')
parser.add_argument('--num_residual_hiddens', nargs='?', default=Configuration.default_num_residual_hiddens, type=int, help='The number of hidden neurons in each layer within a residual block')
parser.add_argument('--num_residual_layers', nargs='?', default=Configuration.default_num_residual_layers, type=int, help='The number of residual layers in a residual stack')
parser.add_argument('--embedding_dim', nargs='?', default=Configuration.default_embedding_dim, type=int, help='Representing the dimensionality of the tensors in the quantized space')
parser.add_argument('--num_embeddings', nargs='?', default=Configuration.default_num_embeddings, type=int, help='The number of vectors in the quantized space')
parser.add_argument('--commitment_cost', nargs='?', default=Configuration.default_commitment_cost, type=float, help='Controls the weighting of the loss terms')
parser.add_argument('--decay', nargs='?', default=Configuration.default_decay, type=float, help='Decay for the moving averages (set to 0.0 to not use EMA)')
parser.add_argument('--learning_rate', nargs='?', default=Configuration.default_learning_rate, type=float, help='The learning rate of the optimizer during training updates')
parser.add_argument('--use_kaiming_normal', nargs='?', default=Configuration.default_use_kaiming_normal, type=bool, help='Use the weight normalization proposed in [He, K et al., 2015]')
parser.add_argument('--unshuffle_dataset', default=not Configuration.default_shuffle_dataset, action='store_true', help='Do not shuffle the dataset before training')
parser.add_argument('--data_path', nargs='?', default='data', type=str, help='The path of the data directory')
parser.add_argument('--results_path', nargs='?', default='results', type=str, help='The path of the results directory')
parser.add_argument('--loss_plot_name', nargs='?', default='loss.png', type=str, help='The file name of the training loss plot')
parser.add_argument('--model_name', nargs='?', default='model.pth', type=str, help='The file name of trained model')
parser.add_argument('--original_images_name', nargs='?', default='original_images.png', type=str, help='The file name of the original images used in evaluation')
parser.add_argument('--validation_images_name', nargs='?', default='validation_images.png', type=str, help='The file name of the reconstructed images used in evaluation')
args = parser.parse_args()

dataset_path = '..' + os.sep + args.data_path


# Dataset and model hyperparameters
configuration = Configuration.build_from_args(args)

dataset = Cifar10Dataset(configuration.batch_size, dataset_path, configuration.shuffle_dataset) # Create an instance of CIFAR10 dataset

auto_encoder = AutoEncoder(device, configuration).to(device)
auto_encoder.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
auto_encoder.eval()
ev =  Evaluator(device, auto_encoder, dataset).to(device)
ev.save_validation_reconstructions_plot(r'..\results\test')