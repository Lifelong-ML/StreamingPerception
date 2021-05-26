# Example: To train on crop 20 shot dataset

# First you will need to prepare a PyTorch image folder at say location "/scratch/zhiqiu/crop_20_shot"
# which means that "/scratch/zhiqiu/crop_20_shot" will have two subfolder "train" and "val", and train and val folders should have 
# same numbers of subfolders (number of classes)
    
# The main_correct_size.py will train your first network from random initialization on this few shot dataset. 
# make sure you specificy the correct number of classes in your dataset with "--classes" arg. 
# The '--ckpt_dir' arg is the location to store your model checkpoints and you could name it arbitrarily.
python main_correct_size.py /scratch/zhiqiu/crop_20_shot --classes 38 --epoch 4000 --step 1500 --a resnet18 --ckpt_dir /project_data/ramanan/zhiqiu/Self-Improving/crop_20_shot ; 

# Then, you can generate the pseudolabels with the model you just trained (specify the location in {--resume} arg). Here the model checkpoint is at "{ckpt_dir}/resnet18_scratch/model_best.pth.tar"
# "/scratch/zhiqiu/imagenet12/train" is where all your unlabeled images are stored. It is also a PyTorch Imagefolder with subfolders indicating the actual classes (though this class information is not used).
# The '--data_save_dir' arg will save unlabeled samples as an image folder.
python generate_labels_correct_fc.py /scratch/zhiqiu/imagenet12/train --classes 38 --a resnet18 --resume /project_data/ramanan/zhiqiu/Self-Improving/crop_20_shot/resnet18_scratch/model_best.pth.tar --data_save_dir /scratch/zhiqiu/imagenet12_crop_20_shot ;

# Now, train another model from scratch on the unlabeled samples + pseudo labels
# The model checkpoint will be saved at "--ckpt_dir"
python main_g_correct_fc.py /scratch/zhiqiu/imagenet12_crop_20_shot --classes 38 --a resnet18 --epochs 30 --step 25 --ckpt_dir /project_data/ramanan/zhiqiu/crop_20_shot/st01 ;

# Finally, finetune this model on the original few shot dataset
# Make sure you specified the correct model to finetune at "--finetuned_model"
python main_gft_correct_fc.py /scratch/zhiqiu/crop_20_shot --classes 38 --epoch 4000 --step 1500 --a resnet18 --ckpt_dir /project_data/ramanan/zhiqiu/crop_20_shot/st01 --finetuned_model /project_data/ramanan/zhiqiu/crop_20_shot/st01/resnet18_scratch/checkpoint.pth.tar;
# Evaluate its final performance
python main_gft_correct_fc.py /scratch/zhiqiu/crop_20_shot --classes 38 --epoch 4000 --step 1500 --a resnet18 --ckpt_dir /project_data/ramanan/zhiqiu/crop_20_shot/st01 --evaluate --finetuned_model /project_data/ramanan/zhiqiu/crop_20_shot/st01/resnet18_finetuned/model_best.pth.tar;

# You can (and should) repeat this process iteratively, just make sure you changed all the paths accordingly so they don't overwrite your earlier checkpoints
# e.g., "--ckpt_dir", '--resume', '--finetuned_model'

