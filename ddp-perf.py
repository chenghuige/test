import torch
import os
import torch.distributed as dist
import time


def main(rank, world):
    torch.cuda.set_device(rank)
    N = 1000
    now = time.time()
    x = torch.ones(8196, 8196).cuda(rank)
    while True:
        for i in range(N):
            print("#", i)
            dist.all_reduce(x, op=dist.reduce_op.SUM)
        torch.cuda.synchronize(rank)
        time.sleep(1)
        print("Ave cost:", (time.time() - now) / N)
        break


if __name__ == '__main__':
    rank = int(os.environ["RANK"])
    print(">>> RANK:", rank)
    torch.distributed.init_process_group(
        "nccl", init_method='env://', rank=rank, world_size=4)
    main(dist.get_rank(), dist.get_world_size())
    
