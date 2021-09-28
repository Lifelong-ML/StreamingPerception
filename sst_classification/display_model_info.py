import argparse
import os


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--mpath', help='path to model')
args = parser.parse_args()
model_path = args.mpath
if (model_path is None or not os.path.isfile(model_path)):
    print('Error: ' + str(model_path) + ' is not a valid path')
    exit()


print('importing torch', flush=True)
from torch import load
model = load(model_path, map_location='cpu')

print('model path:', model_path)
print('model epoch:', model['epoch'])
print('best val acc1:  ', model['best_val_acc1'])

try:
    print('best test acc1:  ', model['best_test_acc1'])
except:
    print('No test acc1 recorded')
