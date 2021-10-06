import argparse
import os


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--mpath', help='path to old model')
parser.add_argument('--savedir', help='optionally save to specific dir')

args = parser.parse_args()
model_path = args.mpath
if (model_path is None or not os.path.isfile(model_path)):
    print('Error: ' + str(model_path) + ' is not a valid path')
    exit()



print('importing torch', flush=True)
from torch import load
from utils import just_save_checkpoint

old_file_name = model_path.split('/')[-1]
new_file_name = 'ereset_' + old_file_name

if (args.savedir == None):
    args.savedir = model_path[0:(-1 * len(old_file_name))]
    


'''
print(old_file_name)
print(new_file_name)
print(og_folder)
'''

model = load(model_path, map_location='cpu')
just_save_checkpoint({'epoch': 0, 'arch': model['arch'], 'state_dict': model['state_dict'], 'best_val_acc1': model['best_val_acc1'], 'optimizer': model['optimizer']}, folder=args.savedir, filename=new_file_name)
print('new model saved to ' + args.savedir + '/' + new_file_name)
