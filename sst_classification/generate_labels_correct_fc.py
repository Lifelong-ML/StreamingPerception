import os
import shutil
import argparse
from tqdm import tqdm
import torch
import torchvision.datasets as datasets
import torchvision.models as models

from utils import model_names, load_from_checkpoint, get_test_transform

from dataset_utils import StreamDataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data_txt', default=None, type=str)

parser.add_argument('data', metavar='DIR',
                    help='path to (unlabeled) dataset')
parser.add_argument('--data_save_dir', default=None, type=str, metavar='PATH',
                    help='new directory (default: none) to save the pseudo label dataset.')
parser.add_argument('--classes', default=1000, type=int, metavar='N',
                    help='number of total classes for the experiment')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')


def main():
    print("in main", flush=-True)
    args = parser.parse_args()

    # create model
    model = models.__dict__[args.arch]()
    if model.fc.weight.shape[0] != args.classes:
        print("changing the size of last layer")
        model.fc = torch.nn.Linear(model.fc.weight.shape[1], args.classes)

    model = torch.nn.DataParallel(model).cuda()

    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir)

    # optionally resume from a checkpoint
    if os.path.isfile(args.resume):
        _, model, _ = load_from_checkpoint(args.resume, model, None)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        exit(0)

    # Data loading code
    # testdir = os.path.join(args.data, 'train')

    if (args.data_txt == None):
        testdir = args.data
        testfilename = os.path.join(args.data_save_dir, f'{args.arch}_scratch.txt')
        test_dataset = datasets.ImageFolder(
            testdir,
            get_test_transform()
        )
        validate_txt(test_dataset, testfilename, model)
        copy_images(args.data_save_dir, testfilename)
        return
    else:
        testfilename = os.path.join(args.data_save_dir, f'{args.arch}_scratch.txt')
        test_dataset = StreamDataset(args.data_txt, get_test_transform())
        print("Dataset length:", len(test_dataset), flush=True)
        validate_txt(test_dataset, testfilename, model)
        return

def validate_txt(test_dataset, testfilename, model, batch_size=64):
    # switch to evaluate mode
    model.eval()
    print(f"Writing to {testfilename}")
    if os.path.exists(testfilename):
        lines = open(testfilename, "r").readlines()
        start = len(lines)
        # assert start % batch_size == 0
        f = open(testfilename, "a+")
    else:
        f = open(testfilename, "w+")
        start = 0

    # loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False,
    #     num_workers=4, pin_memory=True)

    with torch.no_grad():
        print("in validate", flush=True)
        for i in tqdm(range(start, len(test_dataset))):
            image, im_name = test_dataset[i]
            image = image.unsqueeze_(0) # makes tensor 4D to match expected input shape
            #im_name = test_dataset.imgs[i][0]
            image = image.cuda()
            # compute output
            output = model(image)
            index_ = torch.argmax(output).cpu().numpy()
            write_str = im_name + ' '+ repr(index_) + '\n'
            f.write(write_str)
        # for i, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=80):
        #     images = images.cuda()
        #     outputs = model(images)
        #     index_ = torch.argmax(outputs, dim=1).tolist()
        #     pseudo_labels = pseudo_labels + index_
        #     write_str = write_str + [str(idx) + '\n' for idx in index_]
    f.close()

    # write_str = []

    # with open(label_file, "w+") as f:
    #     with torch.no_grad():
    #         # for i in tqdm(range(len(train_dataset)), ncols=80)
    #     for s in write_str:
    #         f.write(s)

def copy_images(data_save_dir, testfilename):
    save_dir = os.path.join(data_save_dir, 'train')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    print(f"Saving to {save_dir}")

    f = open(testfilename, "r")
    for x in f:
        output = x.split(' ')
        x1 = output[0]
        x2 = output[1].replace("array(", "")
        x2 = x2.replace(")\n", "")
        new_dir = os.path.join(save_dir, x2)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        shutil.copy(x1, new_dir, follow_symlinks=False)
    f.close()


if __name__ == '__main__':
    main()

