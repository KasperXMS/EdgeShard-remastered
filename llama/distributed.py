import torch
import torch.nn as nn
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from pathlib import Path
from typing_extensions import OrderedDict, Tuple
from llama.model import ModelArgs, TransformerBlock, RMSNorm, precompute_freqs_cis
from fairscale.nn.model_parallel.layers import VocabParallelEmbedding, ColumnParallelLinear
from dataclasses import dataclass

dict_map = {
    'self_attn': 'attention', 'mlp': 'feed_forward',
    'q_proj': 'wq', 'k_proj': 'wk', 'v_proj': 'wv', 'o_proj': 'wo',
    'up_proj': 'w3', 'down_proj': 'w2', 'gate_proj': 'w1',
    'input_layernorm': 'attention_norm', 'post_attention_layernorm': 'ffn_norm',
    'embed_tokens': 'tok_embeddings', 'lm_head': 'output',
}


class DistributedTransformer(nn.Module):
    def __init__(self, args: ModelArgs, ckpt_dir: str):
        super().__init__()
        self.params = args
        self.p1_rref = rpc.remote("worker0", DTShard, args=(args, (0, 16), ckpt_dir))
        self.p2_rref = rpc.remote("worker1", DTShard, args=(args, (16, 32), ckpt_dir))

        print("All workers initiated")

    def forward(self, tokens: torch.Tensor, start_pos: int):
        with torch.no_grad():
            
            start_time = time.time()
            x_rref = RRef(tokens.to('cpu'))
            out0 = self.p1_rref.remote().forward(x_rref, start_pos)
            out = self.p2_rref.rpc_async().forward(out0, start_pos)

        return torch.futures.wait_all([out])[0].cuda()

class DTShard(nn.Module):
    def __init__(self, params: ModelArgs, offset: Tuple[int, int], ckpt_dir: str):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        if offset[0] == 0:
            self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim).to("cuda")

        self.layers = torch.nn.ModuleList()
        for layer_id in range(offset[0], offset[1]):
            self.layers.append(TransformerBlock(layer_id, params).to("cuda"))

        if offset[1] == params.n_layers:
            self.norm = RMSNorm(params.dim, eps=params.norm_eps).to("cuda")
            self.output = nn.Linear(params.dim, params.vocab_size, bias=False).to("cuda")

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

        self.checkpoints_dir = ckpt_dir
        self.offset = offset

        self._initialize_weight()

    
    def _initialize_weight(self):
        # load pretrained weights
        prev_time = time.time()
        
        checkpoints = sorted(Path(self.checkpoints_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {self.checkpoints_dir}"
        ckpt_path = checkpoints[0]
        print(f'Loading checkpoint "{ckpt_path}"')
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        print(f"Loaded checkpoint in {time.time() - prev_time:.6f}s")
        prev_time = time.time()

        # del checkpoint['rope.freqs']
        new_state_dict = OrderedDict()
        for k,v in checkpoint.items():
            if 'layers' in k:
                names_list = k.split('.')[1:]
                if self.offset[0] <= int(names_list[1]) < self.offset[1]:
                    names_list[1] = str(int(names_list[1]) - self.offset[0])
                names_list = [dict_map[n] if n in dict_map else n for n in names_list]
                name = '.'.join(names_list)
                new_state_dict[name] = v
            else:
                names_list = k.split('.')[-2:]
                names_list = [dict_map[n] if n in dict_map else n for n in names_list]
                name = '.'.join(names_list)
                new_state_dict[name] = v

        # from llama original .pth
        # for k,v in checkpoint.items():
        #     if 'layers' in k:
        #         names_list = k.split('.')[1:]
        #         if self.offset[0] <= int(names_list[0]) < self.offset[1]:
        #             names_list[0] = 'layers.' + str(int(names_list[0]) - self.offset[0])
        #         names_list = [dict_map[n] if n in dict_map else n for n in names_list]
        #         name = '.'.join(names_list)
        #         new_state_dict[name] = v
        #     else:
        #         new_state_dict[k] = v

        print(new_state_dict.keys())

        self.load_state_dict(new_state_dict, strict=True)
        print(f"Loaded state dict in {time.time() - prev_time:.2f}s")


    @torch.inference_mode()
    def forward(self, input_data: any, start_pos: int):
        if self.offset[0] == 0:
            h = input_data.to_here().cuda()
            _bsz, seqlen = h.shape[:2]
            h = self.tok_embeddings(h)

            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(h.device)

        else:
            input_data = input_data.to_here()
            h = input_data['hidden_states'].cuda()
            _bsz, seqlen = h.shape[:2]
            start_pos = input_data['start_pos']
            freqs_cis = input_data['freqs_cis'].cuda()

        
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=h.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        if self.offset[1] == self.n_layers:
            h = self.norm(h)
            h = self.output(h).float()
            return h.cpu()
        
        else:
            h = h.cpu()
            freqs_cis = freqs_cis.cpu()
            return {'hidden_states': h, 'start_pos': start_pos, 'freqs_cis': freqs_cis}
    