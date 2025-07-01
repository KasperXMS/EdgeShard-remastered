import os
from torch.distributed import rpc
from core.load_config import load_config
from core.inference import inference
from datetime import timedelta

config = load_config("config/config_hf.yaml")

os.environ['GLOO_SOCKET_IFNAME'] = config.master.interface
os.environ['TP_SOCKET_IFNAME'] = config.master.interface
os.environ['MASTER_ADDR'] = config.master.ip
os.environ['MASTER_PORT'] = config.master.port
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

if __name__ == '__main__':
    options = rpc.TensorPipeRpcBackendOptions()
    options.set_device_map("worker0", {0: 0})
    options.set_device_map("worker1", {0: 0})
    rpc.init_rpc("master", rank=0, world_size=len(config.workers)+1, 
                 rpc_backend_options=options)
    inference()
    rpc.shutdown()