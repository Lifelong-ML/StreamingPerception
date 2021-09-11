import argparse
import os


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--mpath', help='path to old model')


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
og_folder = model_path[0:(-1 * len(old_file_name))]

'''
print(old_file_name)
print(new_file_name)
print(og_folder)
'''

model = load(model_path, map_location='cpu')
just_save_checkpoint({'epoch': 0, 'arch': model['arch'], 'state_dict': model['state_dict'], 'best_acc1': model['best_acc1'], 'optimizer': model['optimizer']}, folder=og_folder, filename=new_file_name)
print('new model saved to ' + og_folder + new_file_name)
