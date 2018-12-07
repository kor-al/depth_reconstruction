from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()


config.TRAIN.model_size = [320,480]##[400,  560] [480,640]#synthia

## Adam
config.TRAIN.batch_size = 4
config.TRAIN.lr_init = 1e-4

## training
config.TRAIN.n_epoch = 32

config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = 10#int(config.TRAIN.n_epoch / 2)

config.TRAIN.hr_img_path = 'D:/depth_dev/depth_datasets/SYNTHIA/'#SYNTHIA-SEQS-06-SPRING/'
#config.TRAIN.hr_img_path ='D:/Alice/Documents/depth_datasets/SYNTHIA/SYNTHIA-SEQS-06-SPRING/'

config.TRAIN.mask_type = 'rand'# rand, unif, grad, grad_unif, grid
#rand = random mask type

config.TRAIN.vgg_path = 'vgg/vgg16.npy'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'D:/depth_dev/depth_datasets/SYNTHIA/SYNTHIA-SEQS-05-SPRING/'
#config.VALID.hr_img_path = 'D:/Alice/Documents/depth_datasets/SYNTHIA/SYNTHIA-SEQS-05-SPRING/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
