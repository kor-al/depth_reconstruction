import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
import os

import cv2
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

def open_depth_synthia(fname, debug=False):
    depth = cv2.imread(fname, flags=cv2.IMREAD_ANYDEPTH)
    if debug == True:
        print(depth.shape)
        print(depth.dtype)
    depth = depth.astype('float32')
    return depth[:, :, None] / 100.

def open_rgb(fname):
    import cv2
    return cv2.imread(fname).astype('float32')

def get_imgs_fn(file):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    #return scipy.misc.imread(path + file_name, mode='L')
    image = np.asarray(open_depth_synthia(file))

    return image

def convert_float_img_to_uint8(img):
    #img is [-1,1]
    #return tf.image.convert_image_dtype(img, dtype=tf.uint8)
    img = tf.cast((tf.cast(img , tf.float32) + tf.constant(1.))*tf.constant(255.)/tf.constant(2.), tf.uint8)
    return tf.image.convert_image_dtype(img, dtype=tf.uint8)




def augment_imgs(xb, yb): #batch numpy
    aug = np.random.choice([0, 1, 2, 3])

    if aug == 1:
        return xb[:, :, ::-1, :],yb[:, :, ::-1, :]
    elif aug == 2:
        return xb[:, ::-1, :, :], yb[:,::-1,:,:]
    elif aug == 3:
        return xb[:, ::-1, ::-1, :], yb[:,::-1,::-1,:]
    else:
        return xb, yb



def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def preprocess_fn(x, is_small = False, normalize = True):
    if normalize:
        x_norm = (x - np.min(x)) / (np.max(x)- np.min(x))
        #x_norm = tf.concat([i for i in x])
        #print(x.shape)
        #print(x)
        #x_norm = x
        x_norm = x_norm*(2.)-(1.)#[-1,1]
        #x_norm = x_norm + tf.constant(1.)
    else:
        x_norm = x
    if is_small:
        x_res = imresize(x_norm, [96,96], interp='bicubic')
    else:
        x_res = imresize(x_norm, [384,384], interp='bicubic')
    return x_res

##################

def get_file_list(path_to_seq='.'):
    all_files = []
    for parent, subdir, fname in os.walk(path_to_seq):
        if len(subdir) == 0:
            for f in fname:
                all_files.append(os.path.join(parent, f))
    return all_files


def filter_files(flist, keywords):
    return [f for f in flist if all(s in f for s in keywords)]

def get_synthia_imgs_list(synthia_val, is_train = True, synthia_dataset = None, return_rgb = False):
    synthia_val_list = get_file_list(synthia_val)
    if is_train:
        synthia_list = get_file_list(synthia_dataset)
        synthia_list = [f for f in synthia_list if f not in synthia_val_list]
        synthia_depth = filter_files(synthia_list, ['Stereo_Left', 'Depth', 'png'])
        if return_rgb:
            synthia_rgb =  filter_files(synthia_list, ['Stereo_Left','RGB','png'])
            return synthia_depth, synthia_rgb
        return synthia_depth
    else:
        synthia_depth_val = filter_files(synthia_val_list, ['Stereo_Left', 'Depth', 'png'])
        if return_rgb:
            synthia_rgb_val =  filter_files(synthia_val_list, ['Stereo_Left','RGB','png'])
            return synthia_depth_val, synthia_rgb_val
        return synthia_depth_val