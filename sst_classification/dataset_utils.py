import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import albumentations as alb
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.augmentations.transforms import ColorJitter, Superpixels, ToGray, GaussNoise
from albumentations.augmentations.geometric.resize import Resize
from albumentations.pytorch import ToTensorV2

import random
import cv2


class StreamDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.sample_list = self.make_sample_list()
        self.transform=transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            print("converting tensor to list", flush=True)
            print(idx)
            exit()
        img_path = self.sample_list[idx].strip('\n')
        image = Image.open(img_path)

        if(self.transform):
          if(len(np.shape(image)) < 3 or np.shape(image)[2] != 3):
            #print("found weird image", flush=True)
            image = image.convert(mode='RGB')
          image=self.transform(image)

        #target = 0
        return (image, img_path)

    def __len__(self):
        return len(self.sample_list)

    def make_sample_list(self):
        sample_list = []
        f = open(self.file_path, "r")
        for line in f:
            sample_list.append(line)
        return sample_list


class PseudoDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.sample_list = self.make_sample_list()
        self.transform=transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.sample_list[idx][0]
        image = Image.open(img_path)
        target =  self.sample_list[idx][1]

        if(self.transform):
          if(len(np.shape(image)) < 3 or np.shape(image)[2] != 3):
            #print("found weird image", flush=True)
            image = image.convert(mode='RGB')
          image=self.transform(image)

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


class AugmentedStreamDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.base_transform=transform
        self.sample_list = self.make_sample_list()
        self.aug_list = self.make_aug_list()

    def __getitem__(self, idx):
        base_idx, aug_id = divmod(idx, 32)
        img_path = self.sample_list[base_idx].strip('\n')
        image = cv2.imread(img_path)
        # no target because we are pseudo-labelling

        #create a random seed based on idx to get consistent transforms across stages
        random.seed(idx)

        #handle transforms
        if(len(np.shape(image)) < 3 or np.shape(image)[2] != 3):
          #print("found weird image", flush=True)
          image = image.convert(mode='RGB')
        image = self.aug_list[aug_id//8](image=image)['image']
        ''' if (self.base_transform):
            image=self.base_transform(image)
        '''

        return (image, img_path, aug_id)

    def __len__(self):
        return len(self.sample_list * 32)

    def make_sample_list(self):
        sample_list = []
        f = open(self.file_path, "r")
        for line in f:
            sample_list.append(line)
        return sample_list

    def make_aug_list(self):
        # define transforms
        rotate = ShiftScaleRotate(p=1)
        jitter = ColorJitter(p=1)
        supPix = Superpixels(p=1)
        toGray = ToGray(p=1)
        gaussNoise = GaussNoise(p=1)

        resize = Resize(224, 224, p=1)
        normalize = alb.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        toTensor = ToTensorV2()
        base = alb.Compose([resize, normalize, toTensor])

        return [alb.Compose([rotate, base]),
                alb.Compose([rotate, gaussNoise, base]),
                alb.Compose([rotate, jitter, base]),
                alb.Compose([rotate, toGray, base]),
                alb.Compose([rotate, supPix, base]),
                alb.Compose([rotate, gaussNoise, jitter, base]),
                alb.Compose([rotate, gaussNoise, toGray, base]),
                alb.Compose([rotate, gaussNoise, supPix, base])]




class AugmentedPseudoDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.sample_list = self.make_sample_list()
        self.transform=transform
        self.aug_list = self.make_aug_list()


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        base_idx, aug_id = divmod(idx, 32)
        img_path = self.sample_list[idx][0]
        image = cv2.imread(img_path)
        aug_id = self.sample_list[idx][1]
        target =  self.sample_list[idx][2]


        #create a random seed based on idx to get consistent transforms across stages
        random.seed(idx)

        #handle transforms
        if(len(np.shape(image)) < 3 or np.shape(image)[2] != 3):
          #print("found weird image", flush=True)
          image = image.convert(mode='RGB')
        image = self.aug_list[aug_id//8](image=image)['image']
        ''' if (self.base_transform):
            image=self.base_transform(image)
        '''
        return (image, target)

    def __len__(self):
        return len(self.sample_list)

    def make_sample_list(self):
        sample_list = []
        f = open(self.file_path, "r")
        for line in f:
            strings = line.split(' ')
            sample_list.append((strings[0], int(strings[1]), int(strings[2])))
        return sample_list

    def make_aug_list(self):
        # define transforms
        rotate = ShiftScaleRotate(p=1)
        jitter = ColorJitter(p=1)
        supPix = Superpixels(p=1)
        toGray = ToGray(p=1)
        gaussNoise = GaussNoise(p=1)

        resize = Resize(224, 224, p=1)
        normalize = alb.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        toTensor = ToTensorV2()
        base = alb.Compose([resize, normalize, toTensor])

        return [alb.Compose([rotate, base]),
                alb.Compose([rotate, gaussNoise, base]),
                alb.Compose([rotate, jitter, base]),
                alb.Compose([rotate, toGray, base]),
                alb.Compose([rotate, supPix, base]),
                alb.Compose([rotate, gaussNoise, jitter, base]),
                alb.Compose([rotate, gaussNoise, toGray, base]),
                alb.Compose([rotate, gaussNoise, supPix, base])]








    '''
    def set_seed(self, seed):
        # This might not work unless in the DataLoader(), num_workers = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    '''

