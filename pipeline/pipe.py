from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    ''' # ajk is assumption this is only for the forward pass
    
    # BEGIN_HW5_2_1
    # keep list of micro batch elements that are currently in flight
    # each time step, build current scedule based on previous scedule
    # if there is still batches to add, add them
    # if value was at model j, now at j+1
    # if value was at last model, skip
    prev_flight = []
    TOTAL_TIME = num_batches + num_partitions - 1
    micro_count = 0
    for time in range(TOTAL_TIME):
        in_flight = []
        if prev_flight is None: 
            in_flight = [(0,0)]
            micro_count += 1
        else:
            if micro_count < num_batches:
                in_flight.append((micro_count,0))
                micro_count += 1
            for prev in prev_flight: 
                if prev[1] < num_partitions-1:
                    in_flight.append((prev[0], prev[1]+1))
        prev_flight = in_flight
        yield in_flight
    # END_HW5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)
        
        # only have as many worker threads as unique devices
        self.num_stages = len(self.devices)

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''

        bs, *_ = x.shape
        mu_xs = list(torch.chunk(x, self.num_stages, dim=0))
        schedule = _clock_cycles(len(mu_xs), self.num_stages)
        self.compute(mu_xs, schedule)
        for i in range(len(mu_xs)):
            mu_xs[i] = mu_xs[i].to(self.devices[-1])
        y = torch.cat(mu_xs, dim=0)
        return y


    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions # list of modules
        devices = self.devices

        # place all batches onto t0 work queue to start
        for mu_batch in batches:
            mu_batch = mu_batch.to(devices[0])
            task = Task(compute = lambda p=partitions[0], b=mu_batch: p(b))
            self.in_queues[0].put(task)
        last_dev_idx = len(devices)-1
        b_counter = 0
        # loop over the scedule. 
        for step, times in enumerate(schedule): 
            for _, partition_idx in times: 
                # partition idx = i means we are moving off that device onto next one
                (status, exc_info) = self.out_queues[partition_idx].get()
                _, mu_batch = exc_info         
                if partition_idx != last_dev_idx:
                    mu_batch = mu_batch.to(devices[partition_idx+1])
                    task = Task(compute = lambda p=partitions[partition_idx+1], b=mu_batch: p(b))
                    self.in_queues[partition_idx+1].put(task)
                else:                        
                    # if partition idx = last, then replace batch and increment counter
                    batches[b_counter] = mu_batch
                    b_counter += 1
        
        
        """ correct but effectivly serial code
        for step, times in enumerate(schedule): 
            for mu_batch_idx, partition_idx in times: 
                # model partition for current slice we want to compute
                partition = partitions[partition_idx]
                if partition_idx == 0: 
                    # create a task and send it to the in_queue of worker 0
                    mu_batch = batches[mu_batch_idx]
                else:
                    # spin until there is work populated in the previous queue
                    while (self.out_queues[partition_idx-1].empty()): True
                    (status, exc_info) = self.out_queues[partition_idx-1].get()
                    
                    if status == False and exc_info == None:
                        continue # I dont think this should ever be triggered?
                    elif status == False:
                        print(f"HAD A FAILURE CASE")
                        print(exc_info)
                        return 
                    _, mu_batch = exc_info
                # ensure you move the batch to the right device? 
                mu_batch = mu_batch.to(devices[partition_idx])
                task = Task(compute = lambda p=partition, b=mu_batch: p(b))
                self.in_queues[partition_idx].put(task)
        
        # loop over the last device work queue to get the output batches in fifo manner
        for idx in range(len(batches)):
            # spin until output queue ready
            while (self.out_queues[-1].empty()): True
            (status, exc_info) = self.out_queues[-1].get()
            if status == False:
                print(f"HAD A FAILURE CASE")
                print(exc_info)
                return 
            (task,mu_batch) = exc_info
            batches[idx] = mu_batch
    """