print('first line print', flush=True)
import argparse
import os
import random
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models

import self_supervised

from utils import get_scratch_folder_name, get_train_transform, get_test_transform, load_from_checkpoint, just_save_checkpoint, model_names, ProgressMeter, AverageMeter, adjust_learning_rate, accuracy

from dataset_utils import PseudoDataset
from dataset_utils import AugmentedPseudoDataset
print(torch.__version__)
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data_txt', default=None, type=str)
parser.add_argument('--optimizer', default=None, type=str)
parser.add_argument('--aug', default=False, type=bool)


parser.add_argument('data', metavar='DIR',
                    help='path to (unlabeled) dataset')
parser.add_argument('--ckpt_dir', default=None, type=str, metavar='PATH',
                    help='saving directory (default: none).')
parser.add_argument('--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--classes', default=1000, type=int, metavar='N',
                    help='number of total classes for the experiment')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--step', default=25, type=int, metavar='N',
                    help='number of epochs till we lower the lr by 0.1')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--selfsupervised', default=None, choices=['moco_v2', 'byol', 'rot', 'deepcluster', 'relativeloc'],
                    help='name of self supervised model')
parser.add_argument('--samemodel', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def main():
    print("in main", flush=-True)
    args = parser.parse_args()
    if (args.aug):
        print('args.aug = True')
    else:
        print('args.aug = False')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.') 

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    print("main worker started", flush=True)
    ckpt_dir = get_scratch_folder_name(args)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # print("=> creating model '{}'".format(args.arch))
    # model = models.__dict__[args.arch]()

    if args.pretrained or args.selfsupervised:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        if args.selfsupervised:
            if args.selfsupervised == "moco_v2":
                model = self_supervised.moco_v2(model)
            elif args.selfsupervised == "byol":
                model = self_supervised.byol(model)
            elif args.selfsupervised == "rot":
                model = self_supervised.rot(model)
            elif args.selfsupervised == "deepcluster":
                model = self_supervised.deepcluster(model)
            elif args.selfsupervised == "relativeloc":
                model = self_supervised.relativeloc(model)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()


    if model.fc.weight.shape[0] != args.classes:
        print("changing the size of last layer")
        model.fc = torch.nn.Linear(model.fc.weight.shape[1], args.classes)

    if args.samemodel:
        print("!!!!!!!!!!!!!!!using model trained on S")
        _, model, _ = load_from_checkpoint(args.samemodel, model, None)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if (args.optimizer == 'svg'):
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif (args.optimizer == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError('unrecognized/unimplimented optimizer')

    # optionally resume from a checkpoint
    start_epoch = args.start_epoch
    if args.resume:
        start_epoch, model, optimizer = load_from_checkpoint(args.resume, model, optimizer)

    cudnn.benchmark = True

    # Data loading code
    print('starting data_loading', flush=True)
    if(args.data_txt==None):
        print("Loading train data from directory", flush=True)
        traindir = os.path.join(args.data, 'train')
        train_dataset = datasets.ImageFolder(
            traindir,
            get_train_transform()
        )
    else:
        if (args.aug):
            print("Loading augmented train data from txt", flush=True)
            train_dataset = AugmentedPseudoDataset(args.data_txt, transform = get_train_transform())
        else:
            print("Loading train data from txt", flush=True)
            train_dataset = PseudoDataset(args.data_txt, transform = get_train_transform())
    print('Train Dataset Size: ' + str(len(train_dataset)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    log_path = os.path.join(Path(args.ckpt_dir).parent.parent.absolute(), 'compiled_tb_logs', args.ckpt_dir.split('/')[-2], 'pseudo_train_log')
    log_writer = SummaryWriter(log_path)

    print('starting training')
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if (args.optimizer == 'svg'):
            adjust_learning_rate(optimizer, epoch, args.lr, args.step)

        # train for one epoch
        loss, top1, fc_weight, fc_bias, fc_grad = train(train_loader, model, criterion, optimizer, epoch, args, log_writer)

        #log to tensorboard
        log_writer.add_scalar('loss', loss, epoch)
        #log_writer.add_scalar('acc1', top1, epoch)
        log_writer.add_histogram('fc.weight', fc_weight, epoch)
        log_writer.add_histogram('fc.bias', fc_bias, epoch)
        log_writer.add_histogram('fc.weight.grad', fc_grad, epoch)
        log_writer.flush()


        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            just_save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, folder=ckpt_dir)

    print('ckpt:', os.path.join(ckpt_dir, 'checkpoint.state'))

def train(train_loader, model, criterion, optimizer, epoch, args, log_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    time_count = []

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        end = time.time()
        data_time.update(time.time() - end)

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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        time_count += [time.time() - end]
        print(time.time() - end)
        print(f"AVG: {sum(time_count)/len(time_count):.2f}", flush=True)

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg, model.module.fc.weight, model.module.fc.bias, model.module.fc.weight.grad
    '''
    #log to tensorboard
    log_writer.add_scalar('loss', losses.avg)
    log_writer.add_scalar('acc1', top1.avg)
    log_writer.add_histogram('fc.weight', model.module.fc.weight, epoch)
    log_writer.add_histogram('fc.bias', model.module.fc.bias, epoch)
    log_writer.add_histogram('fc.weight.grad', model.module.fc.weight.grad, epoch)
    log_writer.flush()
    '''


if __name__ == '__main__':
    main()


