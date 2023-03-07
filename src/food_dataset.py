import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os


class FoodDataset(object):

    def __init__(self, batch_size, path, shuffle_dataset=True):
        if not os.path.isdir(path):
            os.mkdir(path)

        self._training_data = datasets.Food101(
            root=path,
            split='train',
            download=True,
            target_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                transforms.Resize((64,64))
            ])
        )

        self._validation_data = datasets.Food101(
            root=path,
            split='test',
            download=True,
            target_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                transforms.Resize((64,64))
            ])
        )

        self._training_loader = DataLoader(
            self._training_data, 
            batch_size=batch_size, 
            num_workers=2,
            shuffle=shuffle_dataset,
            pin_memory=True
        )

        self._validation_loader = DataLoader(
            self._validation_data,
            batch_size=batch_size,
            num_workers=2,
            shuffle=shuffle_dataset,
            pin_memory=True
        )
        self._big_batch=DataLoader(
            self.training_data,
            batch_size=1000
            )
        
        images,labels= next(iter(self._big_batch))
        
        images = images.numpy()

        self._train_data_variance = np.var(images)/ 255.0

    @property
    def training_data(self):
        return self._training_data

    @property
    def validation_data(self):
        return self._validation_data

    @property
    def training_loader(self):
        return self._training_loader

    @property
    def validation_loader(self):
        return self._validation_loader

    @property
    def train_data_variance(self):
        return self._train_data_variance
