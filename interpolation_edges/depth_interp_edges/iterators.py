import numpy as np
import threading
import sys
from utils.utils import open_rgb, open_depth_synthia

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue


class SimpleBatchIterator(object):
    def __init__(self, *args, **kwargs):

        if 'batchsize' in kwargs:
            self._batchsize = kwargs['batchsize']
        else:
            self._batchsize = 10
        if 'shuffle' in kwargs:
            self._shuffle = kwargs['shuffle']
        else:
            self._shuffle = True
        self._batch_generator = self._generate_batches(list(args))

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        batch = self._batch_generator.next()
        return batch

    def _generate_batches(self, data):
        num_obj = len(data[0])
        b_s = self._batchsize
        if not all([len(arg) == num_obj for arg in data]):
            raise ValueError("All iterables should contain the same number of objects")
        idx = np.arange(num_obj)
        if self._shuffle:
            np.random.shuffle(idx)
        for i in range(0, num_obj - b_s + 1, b_s):
            batch_idxs = idx[i:i + b_s]
            yield tuple([self._read(np.asarray(arg)[batch_idxs]) for arg in data])

    def _read(self, data):
        return data


class AsyncBatchIterator(SimpleBatchIterator, threading.Thread):

    '''Code for this class is based on https://github.com/justheuristic/prefetch_generator'''
    def __init__(self, *args, **kwargs):

        super(AsyncBatchIterator, self).__init__(*args, **kwargs)

        if 'buffer_size' in kwargs:
            self._buf_size = kwargs['buffer_size']
        else:
            self._buf_size = -1

        threading.Thread.__init__(self)
        self._stop_flag = threading.Event()
        self._stop_flag.clear()
        self._queue = Queue.Queue(self._buf_size)
        self._batch_generator = self._generate_batches(list(args))
        self._daemon = True
        self.start()

    def stop(self):
        self._stop_flag.set()

    def run(self):
        for item in self._batch_generator:
            if self._stop_flag.is_set():
                break
            self._queue.put(item)
        self._queue.put(None)

    def next(self):
        next_item  = self._queue.get()
        if next_item is None:
            raise StopIteration
        return next_item




class SynthiaIterator(AsyncBatchIterator):
    def __init__(self, *args, **kwargs):
        super(SynthiaIterator, self).__init__(*args, **kwargs)

    def _read(self, files):
        if any(['Depth' in f for f in files]):
            #returns depth in meters
            images = np.asarray([open_depth_synthia(f)  for f in files])
        else:
            images = np.asarray([open_rgb(f) for f in files])
        return images




