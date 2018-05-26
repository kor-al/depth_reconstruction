import numpy as np
import os


def get_file_list(path_to_seq='.'):
    all_files = []
    for parent, subdir, fname in os.walk(path_to_seq):
        if len(subdir) == 0:
            for f in fname:
                all_files.append(os.path.join(parent, f))
    return all_files


def filter_files(flist, keywords):
    return [f for f in flist if all(s in f for s in keywords)]


def open_depth(fname):
    import cv2
    return cv2.imread(fname, flags=cv2.IMREAD_ANYDEPTH).astype('float32')

def open_rgb(fname):
    import cv2
    return cv2.imread(fname).astype('float32')


def open_depth_freiburg(fname):
    from utils.pfm import load_pfm
    with open(fname, errors='ignore') as f:
        depth, _ = load_pfm(f)
    #focal lengths are 35 and 15mm, baseline is ~54cm
    if "15" in fname:
        depth = (540. * 450.)/depth[::-1, :, None].astype('float32')
    else:
        depth = (540. * 1050.)/depth[::-1, :, None].astype('float32')
    #return depth in meters
    return depth / 1000.

def open_depth_synthia(fname, debug=False):
    import cv2
    depth = cv2.imread(fname, flags=cv2.IMREAD_ANYDEPTH)
    if debug == True:
        print(depth.shape)
        print(depth.dtype)
    depth = depth.astype('float32')
    return depth[:, :, None] / 100.

def open_depth_mp(fname, debug=False):
    import cv2
    depth = cv2.imread(fname, flags=cv2.IMREAD_ANYDEPTH)
    if debug == True:
        print(depth.shape)
        print(depth.dtype)
    depth = depth.astype('float32')
    return depth[:, :, None] / 100.


def open_depth_tum(fname):
    import cv
    depth = np.asarray(cv2.imread(fname))[:, :].astype('float32')
    return depth[:, :, None] / 5000.


class Preprocessor(object):
    """Applies transformations to batch"""

    def __init__(self, random_flip=False, flip_type='horizontal',
                 crop=False, random_crop=True, crop_size=256,
                 downscale=False, downscale_factor=2,
                 normalize=False):
        
        self._random_flip = random_flip
        self._flip_type = flip_type
        self._random_crop = random_crop
        self._crop_size = crop_size
        self._crop = crop
        self._downscale = downscale
        self._factor = downscale_factor
        self._normalize = normalize

    @staticmethod
    def random_flip(batch, flip_type='horizontal'):
        if flip_type == 'horizontal':
            for i in range(batch.shape[0]):
                batch[i] = batch[i, :, ::-1, :]
        elif flip_type == 'vertical':
            for i in range(batch.shape[0]):
                batch[i] = batch[i, ::-1, :, :]
        else:
            raise ValueError('Unknown flip type "{0}"'.format(flip_type))
        return batch

    @staticmethod
    def crop(batch, random_crop=False, crop_size=256):
        """Crops square patch from random place or center """
        sz = crop_size
        if isinstance(sz, tuple) and len(sz) == 2:
            pass
        elif isinstance(sz, int):
            sz = (sz, sz)
        else:
            raise ValueError('Crop size must be tuple or int')
        height = batch.shape[1]
        width = batch.shape[2]
        batch_cropped = np.zeros((batch.shape[0], sz[0], sz[1], batch.shape[-1]))
        if random_crop:
            for i in range(batch.shape[0]):
                crop_y = np.random.randint(0, height - sz[0] + 1)
                crop_x = np.random.randint(0, width - sz[1] + 1)
                batch_cropped[i, :, :, :] = batch[i, crop_y:crop_y + sz[0], crop_x:crop_x + sz[1], :]
        else:
            for i in range(batch.shape[0]):
                crop_y = height / 2 - sz[0] / 2
                crop_x = width / 2 - sz[1] / 2
                batch_cropped[i, :, :, :] = batch[i, crop_y:crop_y + sz[0], crop_x:crop_x + sz[1], :]
        return batch_cropped

    @staticmethod
    def downscale(batch, factor):
        from skimage.transform import rescale
        downscaled_batch = np.zeros((batch.shape[0], batch.shape[1] / factor,
                                     batch.shape[2] / factor, batch.shape[3]))

        for i in range(batch.shape[0]):
            downscaled_batch[i, :, :, :] = rescale(batch[i, :, :, :], 1. / factor, preserve_range=True)
        return downscaled_batch

    @staticmethod
    def normalize(batch):
        batch = batch.astype('float32')
        for i in range(batch.shape[0]):
            batch[i] = (batch[i] - np.min(batch[i])) / (np.max(batch[i]) - np.min(batch[i]))
        return batch

    def preprocess(self, batch):
        """Shape = (batch, row, col, chan)"""

        if self._crop:
            batch = self.crop(batch, self._random_crop, self._crop_size)
        if self._downscale:
            batch = self.downscale(batch, self._factor)
        if self._random_flip:
            if np.random.randint(0, 2) == 1:
                batch = self.random_flip(batch, flip_type=self._flip_type)
        if self._normalize:
            batch = self.normalize(batch)
        return batch
