from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        # BEGIN_HW5_1_1
        return self.data[self.index[index]]
        # END_HW5_1_1

class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        ''' Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        '''
        n = len(self.data)
        idxs = [i for i in range(n)] 
        rng.shuffle(idxs)

        curr_idx = 0
        for i, size in enumerate(sizes):
            if i == len(sizes)-1:
                self.partitions.append(idxs[curr_idx:])
            else:
                offset = int(n*sizes[i])
                self.partitions.append(idxs[curr_idx:curr_idx+offset])
                curr_idx += offset


    def use(self, partition):
        ''' Return a simple dataset class `Partiton` by original data and partitioned indices

        Just one line of code. Think it simply.
        '''
        # BEGIN_HW5_1_1
        return Partition(self.data, self.partitions[partition])
        # END_HW5_1_1

def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader
    
    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN_HW5_1
    assert batch_size % world_size == 0, "need batch size to split evenly among workers"
    p_bs = batch_size // world_size
    p_sizes = [1/world_size]*world_size
    data_partitioner = DataPartitioner(dataset, p_sizes)
    partition = data_partitioner.use(rank)
    return DataLoader(partition, batch_size, collate_fn=collate_fn)
    # END_HW5_1
