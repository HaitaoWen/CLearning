import random
from torch.utils.data.sampler import Sampler
# from torch._six import int_classes as _int_classes
_int_classes = int


class MemoryBatchSampler_intra(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler, for current task data.
        batch_size (int): Size of mini-batch.
        mem_size (int): number of stored samples.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, sampler, batch_size, mem_size=0, mem_batch_size=0, drop_last=False):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(mem_size, _int_classes) or mem_size < 0:
            raise ValueError('mem_size should be >= 0')
        if not isinstance(mem_batch_size, _int_classes) or mem_batch_size < 0:
            raise ValueError('mem_batch_size should be >= 0')
        if mem_batch_size >= batch_size:
            raise ValueError('batch_size should be > mem_batch_size')
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.mem_batch_size = mem_batch_size
        self.drop_last = drop_last
        self.cur_data_num = len(sampler)

    def init_batch(self):
        batch = [i + self.cur_data_num for i in range(self.mem_size)]
        if self.mem_batch_size > self.mem_size:
            times = self.mem_batch_size // self.mem_size
            remainder = self.mem_batch_size % self.mem_size
            batch = batch * times + random.sample(batch, remainder)
        else:
            batch = random.sample(batch, self.mem_batch_size)
        return batch

    def __iter__(self):
        batch = self.init_batch()
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = self.init_batch()
        if len(batch) > self.mem_batch_size and not self.drop_last:
            yield batch

    def __len__(self):
        batch_size = self.batch_size - self.mem_batch_size
        if self.drop_last:
            return len(self.sampler) // batch_size
        else:
            return (len(self.sampler) + batch_size - 1) // batch_size


class MemoryBatchSampler_extra(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler, for current task data.
        batch_size (int): Size of mini-batch.
        mem_size (int): number of stored samples.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, sampler, batch_size, mem_size=0, mem_batch_size=0, drop_last=False):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(mem_size, _int_classes) or mem_size < 0:
            raise ValueError('mem_size should be >= 0')
        if not isinstance(mem_batch_size, _int_classes) or mem_batch_size < 0:
            raise ValueError('mem_batch_size should be >= 0')
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.mem_batch_size = mem_batch_size
        self.drop_last = drop_last
        self.cur_data_num = len(sampler)

    def init_batch(self):
        batch = [i + self.cur_data_num for i in range(self.mem_size)]
        if self.mem_size > self.mem_batch_size:
            batch = random.sample(batch, self.mem_batch_size)
        return batch

    def __iter__(self):
        n = 0
        batch = self.init_batch()
        for idx in self.sampler:
            batch.append(idx)
            n += 1
            if n == self.batch_size:
                yield batch
                n = 0
                batch = self.init_batch()
        if n > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
