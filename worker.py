import os, torch.distributed.rpc as rpc
from core.load_config import load_config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--worker_rank", type=int, required=True)
args = parser.parse_args()

config = load_config("config/config_hf.yaml")
this_worker = config.workers[args.worker_rank - 1].name   # worker0 / worker1 …

os.environ["MASTER_ADDR"] = config.master.ip
os.environ["MASTER_PORT"] = config.master.port
os.environ["GLOO_SOCKET_IFNAME"] = config.workers[args.worker_rank - 1].interface
os.environ["TP_SOCKET_IFNAME"]   = config.workers[args.worker_rank - 1].interface

opts = rpc.TensorPipeRpcBackendOptions(rpc_timeout=120)
opts.set_device_map("master", {0: 0})

for peer in config.workers:
    if peer.name != this_worker:
        opts.set_device_map(peer.name, {0: 0})   

rpc.init_rpc(
    name=this_worker,
    rank=args.worker_rank,                     # 1, 2, …
    world_size=len(config.workers) + 1,
    rpc_backend_options=opts,
)

print(f"{this_worker}: RPC initialized with CUDA device maps")
rpc.shutdown()
