import numpy as np
import cv2

from skimage.filters import sobel_h, sobel_v
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.filters import threshold_adaptive
from skimage import feature

from utils.ops import *
from config import *

batch_size = config.TRAIN.batch_size

eps = tf.constant(1e-2)
maxi = tf.constant(1e4)

def threshold_mask(img, sz, t):
    img_rescaled = ((img - img.min(axis=(0,1)))/(img.max(axis=(0,1)) - img.min(axis=(0,1))) - 0.5)*2
    grey = rgb2grey(img_rescaled)
    grey_resized = resize(grey, sz, preserve_range=True)
    grad = np.sqrt(sobel_h(grey_resized)**2 + sobel_v(grey_resized)**2)
    thr = threshold_adaptive(grad, t, method='mean')
    return thr.astype(int)[None, :,:, None]

def combined_mask(img, sz, combined=True, block_size = 5, constant = 11 ):
    import cv2
    resized = cv2.resize(img.astype(np.uint8), tuple(sz[::-1]))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0.2)
    grey = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(grey, cv2.CV_16S, 1, 0).astype(float)
    grad_y = cv2.Sobel(grey, cv2.CV_16S, 0, 1).astype(float)
    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
    grad = np.round((grad - grad.min())/(grad.max() - grad.min())*255).astype(np.uint8)
    grad_thr = cv2.adaptiveThreshold(grad,np.max(grad),cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    grad_thr = (grad_thr - grad_thr.min())/(grad_thr.max() - grad_thr.min())
    grad_thr = np.ones(grad_thr.shape, dtype=np.float32) - grad_thr.astype(np.float32)
    random = np.random.choice([0, 1],size=sz, p=[1 - 0.05, 0.05])
    if combined:
        grad_thr = np.clip(grad_thr + random, 0, 1)
    else:
        grad_thr = np.clip(grad_thr, 0, 1)
    return grad_thr.astype(int)[None, :, :, None]


def get_grad_mask(img, sz,combined=True, block_size = 5, constant = 11 ):
    bs = img.shape[0]
    return np.concatenate([combined_mask(img[i,:,:,:], sz, combined,block_size, constant) for i in range(bs)])

def get_unif_mask(sz, p = 0.1):
    mask = np.random.choice([0,1],size=sz, p=[1 - p, p])[None,:,:,None]
    return mask

def get_regular_grid_mask(sz, step=7):
    z = np.zeros((1, *sz, 1))
    z[:, ::step, ::step, :] = 1
    return z

def get_mask(img, sz, mask_type, mask_params = None):
    if mask_type == 'rand':
        mask_type = np.random.choice(['unif', 'grad', 'grad_unif'])
    return get_typed_mask(img, sz, mask_type, mask_params)

def get_typed_mask(img, sz, mask_type, mask_params = None):
    if mask_params == None:
        mask_params = {}

    if mask_type == 'unif':
        if 'p' in mask_params:
            p = mask_params['p']
        else:
            p = 0.1
        return get_unif_mask(sz, p)
    elif mask_type == 'grad' or mask_type == 'grad_unif':
        if 'block size' in mask_params:
            b_sz = mask_params['block_size']
        else:
            b_sz = 5
        if  'constant' in mask_params:
            const = mask_params['constant']
        else:
            const = 11
        if mask_type == 'grad':
            return get_grad_mask(img, sz,combined=False, block_size = b_sz, constant = const)
        else:
            return get_grad_mask(img, sz, combined= True, block_size=b_sz, constant=const)
    elif mask_type == 'grid':
        if 'step' in mask_params:
            step = mask_params['step']
        else:
            step = 7
        return get_regular_grid_mask(sz, step)


def detect_edges(greyscale, sz):
    resized = cv2.resize(greyscale.astype(np.uint8), tuple(sz[::-1]))
    edges = feature.canny(resized, low_threshold=0, high_threshold=110)
    #edges = feature.canny(greyscale, sigma=25)
    return edges[None,...,None]

def get_edges(greyscale, sz):
    bs = greyscale.shape[0]
    return np.concatenate([detect_edges(greyscale[i, :, :, :],sz) for i in range(bs)])

def greyscale_from_rgb(rgb):
    greyscale =  rgb[:,:,:,0] * 0.2989 + rgb[:,:,:,1] * 0.5870 + rgb[:,:,:,2] * 0.1140
    return greyscale[:, :, :, None]

def get_edges_from_rgb(rgb,sz):
    greyscale = greyscale_from_rgb(rgb)
    return get_edges(greyscale,sz)




def preprocess(t_input_d, t_mask,t_edges, size):

    d_resized = tf.image.resize_images(t_input_d, size)
    d_clipped = tf.clip_by_value(d_resized, eps, maxi)
    d_log = tf.log(d_clipped + 1.)
    d_norm = d_log # try normalization?
    ground_truth = d_norm

    d_masked = ground_truth*t_mask[:,:]

    nonzero = tf.cast(tf.count_nonzero(d_masked, axis = (1,2), keep_dims= True), tf.float32)
    masked_sum = tf.reduce_sum(d_masked, axis = (1,2), keep_dims= True)
    mean_d = masked_sum/nonzero

    #mean, std = tf.nn.moments(ground_truth*mask_tensor[:,:], axes=(1,2), keep_dims=True)
    d_masked_centered = d_masked - mean_d

    edges_float = tf.cast(t_edges, tf.float32)

    return d_masked_centered, mean_d, ground_truth, edges_float

def return2original_scale(t_depth, mean_d):
    return t_depth + mean_d
