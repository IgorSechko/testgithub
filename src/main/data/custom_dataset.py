"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import skimage
from skimage import io, color
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import random

""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""


class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset

        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']

        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)

        return sample


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""


class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform

        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform

        dataset.transform = None
        self.dataset = dataset
        self.indices = indices  # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors + 1]
        assert (self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)

        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image']
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']

        return output


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None, listdir=None):

        self.image_dir = image_dir
        self.transform = transform
        self.classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

        listdir = torch.load("footwear_names_listdir.pt")
        image_names = []
        if listdir is not None:
            image_names = listdir
        else:
            image_names = os.listdir(image_dir)

        #random.shuffle(image_names)
        #image_names = image_names[:10000]
        self.image_names = [os.path.join(image_dir, image_name) for image_name in image_names]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img_np = self.load_image(idx)
        img = Image.fromarray(img_np)
        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'target': 0}
        return sample

    def get_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img  # .astype(np.float32) / 255.0

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)
