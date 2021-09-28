print("starting imports", flush=True)
import torch
from utils import get_train_transform
from dataset_utils import PseudoDataset
import os
import numpy as np
import torch.nn as nn
print("finished imports", flush=True)


data_path = "/scratch/ssolit/StreamingPerception/sst_classification/small_gc_dataset.txt"
assert(os.path.isfile(data_path))

print("creating dataset", flush=True)
train_dataset = PseudoDataset(data_path, transform = get_train_transform())
print("finished dataset", flush=True)


import torchvision.models as models 
model = models.__dict__["resnet18"]()
print("model created", flush=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128,
    num_workers=4)

print("train_loader created", flush=True)
model.train()

criterion = nn.CrossEntropyLoss()

for i, (images, target) in enumerate(train_loader):
#    print(np.shape(images))
 #   print(np.shape(np.transpose(images, (0, 3, 2, 1))))
#    images = np.transpose(images, (0, 3, 2, 1))
    output = model(images)
    
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("complete", flush=True)
    exit()




