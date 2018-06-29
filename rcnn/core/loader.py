import mxnet as mx
import numpy as np
import Queue
import threading
import time

from rcnn.io.rpn import get_rpn_testbatch


class SequentialLoader(mx.io.DataIter):
    def __init__(self, iters):
        super(SequentialLoader, self).__init__()
        self.iters = iters
        self.exhausted = [False] * len(iters)

    def __getattr__(self, attr):
        # delegate unknown keys to underlying iterators
        first_non_empty_idx = self.exhausted.index(False)
        first_non_empty_iter = self.iters[first_non_empty_idx]
        return getattr(first_non_empty_iter, attr)

    def next(self):
        while True:
            if all(self.exhausted):
                raise StopIteration
            first_non_empty_idx = self.exhausted.index(False)
            first_non_empty_iter = self.iters[first_non_empty_idx]
            try:
                result = first_non_empty_iter.next()
                return result
            except StopIteration:
                self.exhausted[first_non_empty_idx] = True

    def reset(self):
        for it in self.iters:
            it.reset()
        self.exhausted = [False] * len(self.iters)

    @property
    def provide_data(self):
        return self.iters[0].provide_data

    @property
    def provide_label(self):
        return self.iters[0].provide_label


class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size=1, shuffle=False, has_rpn=False):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn
        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = range(self.size)

        # decide data and label names (only for training)
        if has_rpn:
            self.data_name = ['data', 'im_info']
        else:
            raise NotImplementedError
        self.label_name = None

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = None
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.get_first_batch()

        self._start_thread()
        self.reset()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)
        for batch in [self.index[start:start+self.batch_size] for start in range(0, self.size, self.batch_size)]:
            self.index_queue.put(batch)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            data_batch = mx.io.DataBatch(data=self.data, label=self.label,
                                         pad=self.getpad(), index=self.getindex(),
                                         provide_data=self.provide_data, provide_label=self.provide_label)
            self.cur += self.batch_size
            return self.im_info, data_batch

        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_first_batch(self):
        roidb = [self.roidb[idx] for idx in range(self.batch_size)]

        if self.has_rpn:
            data, label, im_info = get_rpn_testbatch(roidb)
        else:
            raise NotImplementedError
        self.data = [mx.nd.array(data[name]) for name in self.data_name] 
        self.im_info = im_info

    def get_batch(self):
        result = self.result_queue.get()
        self.data = result["data"]
        self.im_info = result["im_info"]

    def _start_thread(self, num_thread=8):
        self.last_index = 0
        self.index_queue = Queue.Queue()
        self.result_queue = Queue.Queue(maxsize=num_thread)
        self.workers = [threading.Thread(target=self.worker) for _ in range(num_thread)] 
        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def worker(self):
        while True:
            indexes = self.index_queue.get()

            if indexes is None:
                return

            roidb = [self.roidb[idx] for idx in indexes]

            # pad last batch
            if len(roidb) < self.batch_size:
                for _ in range(self.batch_size - len(roidb)):
                    roidb.append(self.roidb[0])

            if self.has_rpn:
                data, label, im_info = get_rpn_testbatch(roidb)
            else:
                raise NotImplementedError
            result = {"data": [mx.nd.array(data[name]) for name in self.data_name], "im_info": im_info}

            while True:
                if self.last_index == indexes[0]:
                    self.last_index += len(indexes)
                    self.result_queue.put(result)
                    break
                else:
                    time.sleep(0.001)
