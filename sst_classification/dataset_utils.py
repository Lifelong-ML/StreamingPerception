import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class PseudoDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.sample_list = self.make_sample_list()
        self.transform=transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.sample_list[idx][0]
        target =  self.sample_list[idx][1]
        image = Image.open(img_path)

        if(self.transform):
          #print(np.shape(image))
          image=self.transform(image)
          #image=np.transpose(image, (0, 3, 2, 1))
          #print(np.shape(image))

        return (image, target)

    def __len__(self):
        return len(self.sample_list)

    def make_sample_list(self):
        sample_list = []
        f = open(self.file_path, "r")
        for line in f:
            strings = line.replace("(", " ")
            strings = strings.replace(")", " ")
            strings = strings.split(" ")
            sample_list.append((strings[0], int(strings[2])))
        return sample_list

'''
class PseudoDataset2(datasets.DatasetFolder):
    def __init__(self, file_path, transform=None, target_transform=None, loader=default_loader):
        super.__init__(root=None, loader=loader, transform=transform, target_transform=target_transform)
        self.file_path = file_path
        self.samples = self.make_dataset()

    def make_dataset(self):
        sample_list = []
        f = open(self.file_path, "r")
        for line in f:
            strings = line.replace("(", " ")
            strings = strings.replace(")", " ")
            strings = strings.split(" ")
            sample_list.append((strings[0], int(strings[2])))
        return sample_list

    def find_classes(self, directory):
        print("skipping class identification")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self) -> int:
        return len(self.samples)

def load_func(line):
    # a line in 'list.txt"
    strings = line.replace("(", " ")        
    strings = strings.replace(")", " ")
    strings = strings.split(" ")
    sample_list.append((strings[0], int(strings[2])))
    return {'src': strings[0], 'target': strings[2]} 

def batchify(batch):
    # batch will contain a list of {'src', 'target'}, or how you return it in load_func.

    # Implement method to batch the list above into Tensor here

    # assuming you already have two tensor containing batched Tensor for src and target
    return {'src': batch_src, 'target': batch_target} # you can return a tuple or whatever you want it to

def get_pseudo_dataset(file_path):
    sample_list = []
    f = open(file_path, "r")
    for line in f:
        strings = line.replace("(", " ")
        strings = strings.replace(")", " ")
        strings = strings.split(" ")
        sample_list.append((strings[0], int(strings[2])))
    return sample_list

'''
