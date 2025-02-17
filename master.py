import os
import yaml
from torch.distributed import rpc
from llama.main import main

config = {}
with open("config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

os.environ['GLOO_SOCKET_IFNAME'] = config['master']['interface']
os.environ['TP_SOCKET_IFNAME'] = config['master']['interface']
os.environ['MASTER_ADDR'] = config['master']['ip']
os.environ['MASTER_PORT'] = config['master']['port']

if __name__ == '__main__':
    rpc.init_rpc("master", rank=0, world_size=len(config['worker'])+1)
    main(
        ckpt_dir="./model_shards",
        tokenizer_path="./llama_3.1_8b_tokenizer/tokenizer.model",
        max_batch_size=4,
        max_seq_len=1024,
        max_gen_len=128,
    )
    rpc.shutdown()