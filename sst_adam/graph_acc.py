print("first line print", flush=True)


import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import sys
from argparse import Namespace

from utils import get_finetuned_folder_name, get_train_transform, get_test_transform, load_from_checkpoint, save_checkpoint, model_names, ProgressMeter, AverageMeter, adjust_learning_rate, accuracy


best_acc1 = 0


args = Namespace(gpu = None, print_freq = 10, batch_size = 128, workers=4, arch="resnet18", lr = 0.1, momentum = 0.9)
args.classes = 102
models_dir = "/mnt/Data/Streaming_Data/flowers2/finetune/resnet18_finetuned/"
args.data = "/mnt/Data/Streaming_Data/102flowers/"


def main():
    print("main() called", flush=True)

    ngpus_per_node = torch.cuda.device_count()
    sys.stdout.flush()


    x_vals = []
    y_vals = []

    for filename in os.listdir(models_dir):
      model_dict = torch.load(models_dir + filename)
      args.finetuned_model = models_dir + filename
      x_vals.append(model_dict["epoch"])
      y_vals.append(main_worker(None, ngpus_per_node, args).item())

      print(x_vals)
      print(y_vals)
      exit()




def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu


    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    if model.fc.weight.shape[0] != args.classes:
        print("changing the size of last layer")
        model.fc = torch.nn.Linear(model.fc.weight.shape[1], args.classes)
    sys.stdout.flush()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    cudnn.benchmark = True

    # optionally resume from a checkpoint for model and optimizer
    _, model, _ = load_from_checkpoint(args.finetuned_model, model, None)

    # Data loading code
    print("Beginning Data Loading",flush=True)
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        get_train_transform()
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, get_test_transform()),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return validate(val_loader, model, criterion, args)




def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg



if __name__ == '__main__':
    main()
