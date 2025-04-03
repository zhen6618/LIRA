import math
import time
import torch
import torch.distributed as dist
from torch.utils.data.distributed import Sampler


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices

    .. warning::
        In distributed mode, calling the ``set_epoch`` method is needed to
        make shuffling work; each process will use the same random seed
        otherwise.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas  # 2 num_replicas: 参与分布式训练的总进程数（通常是 GPU 数量）
        self.rank = rank  # 0 or 1
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))  # 3500
        self.total_size = self.num_samples * self.num_replicas  # 7000
        self.shuffle = shuffle  # False

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # print("rank: ", self.rank)  # 0 or 1
        # # print("indices: ", indices)  
        # print("num_samples: ", self.num_samples)  # 3500
        # print("total_size: ", self.total_size)  # 7000
        # print("len(indices): ", len(indices))  # 7000
        # print("shuffle: ", self.shuffle)  # False

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        subsample_num = self.total_size // self.num_replicas
        subsample_begin = subsample_num * self.rank
        indices = indices[subsample_begin:subsample_begin + subsample_num]
        assert len(indices) == self.num_samples

        # print("subsample_num: ", subsample_num)  # 3500
        # print("subsample_begin: ", subsample_begin)  # 0 or 3500
        # print("indices: ", indices)  # 0-3499 or 3500-7000
        # print("len(indices): ", len(indices))
        # time.sleep(10000)

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# class DistributedSampler(Sampler):
#     r"""Sampler that restricts data loading to a subset of the dataset.

#     It is especially useful in conjunction with
#     :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
#     process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
#     :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
#     original dataset that is exclusive to it.

#     .. note::
#         Dataset is assumed to be of constant size and that any instance of it always
#         returns the same elements in the same order.

#     Args:
#         dataset: Dataset used for sampling.
#         num_replicas (int, optional): Number of processes participating in
#             distributed training. By default, :attr:`world_size` is retrieved from the
#             current distributed group.
#         rank (int, optional): Rank of the current process within :attr:`num_replicas`.
#             By default, :attr:`rank` is retrieved from the current distributed
#             group.
#         shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
#             indices.
#         seed (int, optional): random seed used to shuffle the sampler if
#             :attr:`shuffle=True`. This number should be identical across all
#             processes in the distributed group. Default: ``0``.
#         drop_last (bool, optional): if ``True``, then the sampler will drop the
#             tail of the data to make it evenly divisible across the number of
#             replicas. If ``False``, the sampler will add extra indices to make
#             the data evenly divisible across the replicas. Default: ``False``.

#     .. warning::
#         In distributed mode, calling the :meth:`set_epoch` method at
#         the beginning of each epoch **before** creating the :class:`DataLoader` iterator
#         is necessary to make shuffling work properly across multiple epochs. Otherwise,
#         the same ordering will be always used.

#     Example::

#         >>> # xdoctest: +SKIP
#         >>> sampler = DistributedSampler(dataset) if is_distributed else None
#         >>> loader = DataLoader(dataset, shuffle=(sampler is None),
#         ...                     sampler=sampler)
#         >>> for epoch in range(start_epoch, n_epochs):
#         ...     if is_distributed:
#         ...         sampler.set_epoch(epoch)
#         ...     train(loader)
#     """

#     def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
#                  rank: Optional[int] = None, shuffle: bool = True,
#                  seed: int = 0, drop_last: bool = False) -> None:
#         if num_replicas is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             num_replicas = dist.get_world_size()
#         if rank is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             rank = dist.get_rank()
#         if rank >= num_replicas or rank < 0:
#             raise ValueError(
#                 f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.drop_last = drop_last
#         # If the dataset length is evenly divisible by # of replicas, then there
#         # is no need to drop any data, since the dataset will be split equally.
#         if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
#             # Split to nearest available length that is evenly divisible.
#             # This is to ensure each rank receives the same amount of data when
#             # using this Sampler.
#             self.num_samples = math.ceil(
#                 (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
#             )
#         else:
#             self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
#         self.total_size = self.num_samples * self.num_replicas
#         self.shuffle = shuffle
#         self.seed = seed

#     def __iter__(self) -> Iterator[T_co]:
#         if self.shuffle:
#             # deterministically shuffle based on epoch and seed
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch)
#             indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
#         else:
#             indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

#         # print("rank: ", self.rank)  # 0 or 1
#         # # print("indices: ", indices)  
#         # print("num_samples: ", self.num_samples)  # 3500
#         # print("total_size: ", self.total_size)  # 7000
#         # print("len(indices): ", len(indices))  # 7000
#         # print("shuffle: ", self.shuffle)  # False

#         if not self.drop_last:
#             # add extra samples to make it evenly divisible
#             padding_size = self.total_size - len(indices)
#             if padding_size <= len(indices):
#                 indices += indices[:padding_size]
#             else:
#                 indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
#         else:
#             # remove tail of data to make it evenly divisible.
#             indices = indices[:self.total_size]
#         assert len(indices) == self.total_size

#         # subsample
#         indices = indices[self.rank:self.total_size:self.num_replicas]
#         assert len(indices) == self.num_samples

#         # print("indices: ", indices)  # 0-3499 or 3500-7000
#         # print("len(indices): ", len(indices))
#         # time.sleep(10000)

#         return iter(indices)

#     def __len__(self) -> int:
#         return self.num_samples

#     def set_epoch(self, epoch: int) -> None:
#         r"""
#         Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
#         use a different random ordering for each epoch. Otherwise, the next iteration of this
#         sampler will yield the same ordering.

#         Args:
#             epoch (int): Epoch number.
#         """
#         self.epoch = epoch
