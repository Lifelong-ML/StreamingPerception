import os
import torch
import shutil
import torchvision.transforms as transforms
import torchvision.models as models
#import torchvision.datasets.DatasetFolder as DatasetFolder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def just_save_checkpoint(state, folder, filename='checkpoint.state'):
    torch.save(state, os.path.join(folder, filename))
    shutil.copyfile(os.path.join(folder, filename), os.path.join(folder, 'model_best.state'))

def save_checkpoint(state, folder, filename='checkpoint.state'):
    torch.save(state, os.path.join(folder, filename))
    best_ckpt = os.path.join(folder, 'model_best.state')

    if os.path.exists(best_ckpt):
        best = torch.load(best_ckpt)['best_acc1']
        is_best = best < state['best_acc1']
    else:
        is_best = True
    if is_best:
        print(f"Better than currently best model. Save at {os.path.join(folder, 'model_best.state')}")
        shutil.copyfile(os.path.join(folder, filename), os.path.join(folder, 'model_best.state'))
    else:
         print("old model is better")

def save_all(state, folder, filename='checkpoint.state'):
    recent_name = state["arch"] + "_checkpoint_e" + str(state["epoch"]) + ".state"
    torch.save(state, os.path.join(folder, recent_name))

    torch.save(state, os.path.join(folder, filename))
    best_ckpt = os.path.join(folder, 'model_best.state')

    if os.path.exists(best_ckpt):
        best = torch.load(best_ckpt)['best_acc1']
        is_best = best < state['best_acc1']
    else:
        is_best = True
    if is_best:
        print(f"Better than currently best model. Save at {os.path.join(folder, 'model_best.state')}")
        shutil.copyfile(os.path.join(folder, filename), os.path.join(folder, 'model_best.state'))
    else:
        print("old model is better")

def get_scratch_folder_name(args):
    return os.path.join(args.ckpt_dir, args.arch+"_scratch")

def get_selfsupervised_folder_name(args):
    return os.path.join(args.ckpt_dir, args.arch+"_self_"+args.selfsupervised)

def get_imgpretrained_folder_name(args):
    return os.path.join(args.ckpt_dir, args.arch+"_imgpretrained")


def get_finetuned_folder_name(args):
    return os.path.join(args.ckpt_dir, args.arch+"_finetuned")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

def load_from_checkpoint(ckpt_loc, model, optimizer):
    if os.path.isfile(ckpt_loc):
        print("=> loading checkpoint '{}'".format(ckpt_loc))
        checkpoint = torch.load(ckpt_loc)
        
        # Remove modules from keys
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        # TODO: Refactor
        if 'eval_model' in checkpoint:
            checkpoint['state_dict'] = checkpoint['eval_model']
            checkpoint['epoch'] = 0
            optimizer = None
        for k, v in checkpoint['state_dict'].items():
            if "module." in k and "module" not in model.__dict__["_modules"]:
                name = k[7:] # remove 'module.' of dataparallel if model is not itself dataparallel
            elif "module." not in k and "module" in model.__dict__["_modules"]:
                name = 'module.'+k
            else: 
                name = k
            new_state_dict[name]=v

            checkpoint['state_dict'] = new_state_dict
        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])
        if optimizer: 
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("No optimizer is loading from checkpoint.")
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(ckpt_loc, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_loc))
        exit(0)
    return start_epoch, model, optimizer

class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, pseudo_labels):
        """
            Args:
                original_dataset (torch.utils.data.Dataset) : The original dataset (the ground truth label will be discarded)
                pseudo_labels (list[int]) : The new labels
        """
        self.original_dataset = original_dataset
        self.pseudo_labels = pseudo_labels

        assert len(self.original_dataset) == len(self.pseudo_labels)

    def __getitem__(self, index):
        return self.original_dataset[index][0], self.pseudo_labels[index]
    
    def __len__(self):
        return len(self.original_dataset)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, lr, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_stream_dataset(file_path):
    sample_list = []
    f = open(file_path, "r")
    for line in f:
        sample_list.append((line, 0))
    return sample_list

#def save_pseudolabels_txt(save_path):
'''
class PseudoDataset(DatasetFolder):
    def __init__(self, file_path):
        super.__init__()
        self.file_path = file_path

    def make_dataset(self):
        sample_list = []
        f = open(self.file_path, "r")
        for line in f:
            strings = line.replace("(", " ")
            strings = strings.replace(")", " ")
            strings = strings.split(" ")
            sample_list.append((strings[0], int(strings[2])))
        return sample_list
'''

def get_pseudo_dataset(file_path):
    sample_list = []
    f = open(file_path, "r")
    for line in f:
        strings = line.replace("(", " ")
        strings = strings.replace(")", " ")
        strings = strings.split(" ")
        sample_list.append((strings[0], int(strings[2])))
    return sample_list
