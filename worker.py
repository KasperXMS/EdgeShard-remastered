import os
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.optim as optim
import yaml
import argparse
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

config = {}
with open("config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

os.environ['GLOO_SOCKET_IFNAME'] = config['master']['interface']
os.environ['TP_SOCKET_IFNAME'] = config['master']['interface']
os.environ['MASTER_ADDR'] = config['master']['ip']
os.environ['MASTER_PORT'] = config['master']['port']

parser = argparse.ArgumentParser()
parser.add_argument("--worker_rank", type=int, help="rank of this worker (start from 0)")
args = parser.parse_args()


if __name__ == '__main__':
    # 初始化主节点的RPC连接
    print(list(config['worker'][args.worker_rank - 1].keys())[0])
    rpc.init_rpc(list(config['worker'][args.worker_rank - 1].keys())[0], rank=args.worker_rank, world_size=len(config['worker']) + 1)
    print("RPC Start")


	# 等待主节点的调用
    rpc.shutdown()
