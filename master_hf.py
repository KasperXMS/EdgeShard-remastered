import os
import yaml
from torch.distributed import rpc
from hf.load_config import load_config
from hf.inference import inference
from datetime import timedelta

config = load_config("config/config_hf.yaml")

os.environ['GLOO_SOCKET_IFNAME'] = config.master.interface
os.environ['TP_SOCKET_IFNAME'] = config.master.interface
os.environ['MASTER_ADDR'] = config.master.ip
os.environ['MASTER_PORT'] = config.master.port

if __name__ == '__main__':
    rpc.init_rpc("master", rank=0, world_size=len(config.workers)+1, 
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        rpc_timeout=120))
    inference()
    rpc.shutdown()