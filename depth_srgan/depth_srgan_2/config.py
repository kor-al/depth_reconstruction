from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

#config.TRAIN.small_size =[76,128] #[187, 225] #synthia
#config.TRAIN.big_size = [608,1024]#[1496, 1800] #SR synthia

config.TRAIN.small_size =[38,64] #[187, 225] #synthia
config.TRAIN.big_size = [304,512]#[1496, 1800] #SR synthia

## Adam
config.TRAIN.batch_size = 4
config.TRAIN.lr_init =1e-5# 2 round =  1e-5 # 1 round =  1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 16
# config.TRAIN.lr_decay_init = 0.1
# config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 32
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = 10#int(config.TRAIN.n_epoch / 2)

config.TRAIN.hr_img_path = 'D:/depth_dev/depth_datasets/SYNTHIA/'#SYNTHIA-SEQS-06-SPRING/'
config.TRAIN.lr_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'D:/depth_dev/depth_datasets/SYNTHIA/SYNTHIA-SEQS-05-SPRING/'
config.VALID.lr_img_path = 'data2017/DIV2K_valid_LR_bicubic/X4/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
