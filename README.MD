Unofficial & experimental implementation of EdgeShard
==========================================================================

**Warning**: This repository is an **unofficial, half-finished implementation** of EdgeShard intended to demonstrate and experiment with integrating EdgeShard-style model sharding with the HuggingFace `transformers` ecosystem. **It is incomplete: many features are missing, some APIs are simplified, and this code is provided for internal experimentation only — unable for learning and production use.** 

**This repository will also face multiple rounds of refactorization, to solve the existing issues and experiment on new features.**

**For official code, please contact the authors of the paper.**

Key notes
---------
- Purpose: provide a lightweight, experimental bridge between an EdgeShard-like partitioned model runtime and HuggingFace transformers to allow local testing and development.
- Status: partial. Core inference paths and a basic single-node interactive CLI exist, but advanced features (robust multi-node orchestration, fault tolerance, full model-parallel correctness checks, comprehensive tests, and performance optimizations) are not implemented.
- **Known issue: inference latency increases as generated sequence length grows**

Requirements
------------
- OS: tested on Linux and Windows dev machines (your environment may vary).
- GPU: CUDA-capable GPU strongly recommended for reasonable latency. CPU-only runs are possible but will be slow.
- Python: 3.10+ recommended.
- Install dependencies (use a virtual environment):

```powershell
pip install -r requirements.txt
```

Files of interest
-----------------
- `master.py` — small driver that initializes a PyTorch RPC master and calls the inference loop. Used in a distributed setup.
- `worker.py` — helper for initializing worker RPC processes when running a distributed configuration.
- `inference_solo.py` — interactive single-process inference loop (useful for quick experiments).
- `core/load_config.py` and `config/config_hf.yaml` — configuration loader and default runtime config.
- `models/llama_model.py` — custom model wrapper used to plug a sharded model into HF-style generation APIs.

Model setup — download, partition, and distribute
------------------------------------------------
This project expects you to download a model from HuggingFace, save it locally, partition its weights into shards, and then distribute those shards and tokenizer files across master/worker nodes.

The default example uses `meta-llama/Llama-3.1-8B-Instruct` and stores artifacts under `./llama-3.1-8b/` and `./model_shards/`. Adjust to your model as needed.

1) Download model and tokenizer from HuggingFace
-----------------------------------------------
Requirements:
- You need a valid HuggingFace access token with permission for the chosen model.
- Set it as an environment variable `HF_TOKEN` before running the script.

Windows (PowerShell):
```powershell
$env:HF_TOKEN = "<your_hf_token>"
python huggingface.py
```

Linux / macOS:
```bash
export HF_TOKEN="<your_hf_token>"
python huggingface.py
```

What this does:
- Downloads tokenizer and model weights using `transformers`.
- Saves tokenizer files to `./llama-3.1-8b/` and a consolidated checkpoint `llama-3.1-8b.pth` to the same folder. You can modify the intended name of the output file in the code.

2) Partition the weights into shards
-----------------------------------
By default, the included script partitions the state dict into ranges (example: `[0, 16, 32]`) and produces multiple `shard_*.pth` files under `./model_shards/`. You can edit the partition points in the code, according to your need.

Run:
```powershell
python weight_partitioner.py
```

Outputs (example):
- `model_shards/shard_0.pth` — layers start..end for worker0 plus embeddings
- `model_shards/shard_1.pth` — layers start..end for worker1 (example)
- `model_shards/shard_2.pth` — `lm_head.weight` (used by master per `config/config_hf.yaml`)

Make sure `config/config_hf.yaml` matches your intended layout:
- `master.lm_head_weight_path` should point to the `lm_head` shard (e.g., `model_shards/shard_2.pth`).
- Each worker has `ckpt_path` set to its shard (e.g., `model_shards/shard_0.pth`, `model_shards/shard_1.pth`).
- `workers[i].start`/`end` should match the ranges used during partitioning.

3) Distribute shards and tokenizer via scp
-----------------------------------------
Distribute the artifacts to each node so the master/worker processes can load their parts locally. You typically need to send:
- For workers: the corresponding `model_shards/shard_*.pth` file.
- For master: the `lm_head` shard (see `master.lm_head_weight_path`) and the tokenizer files (from `llama-3.1-8b/`).

Examples (PowerShell using scp; replace hosts and paths):
```powershell
# Copy worker0 shard to worker0 host
scp model_shards/shard_0.pth user@192.168.1.119:/home/user/EdgeShard-remastered/model_shards/

# Copy worker1 shard to worker1 host
scp model_shards/shard_1.pth user@192.168.1.121:/home/user/EdgeShard-remastered/model_shards/

# Copy tokenizer files to master (and optionally to workers)
scp llama-3.1-8b/tokenizer.json user@192.168.1.119:/home/user/EdgeShard-remastered/llama-3.1-8b/
scp llama-3.1-8b/tokenizer_config.json user@192.168.1.119:/home/user/EdgeShard-remastered/llama-3.1-8b/
scp llama-3.1-8b/special_tokens_map.json user@192.168.1.119:/home/user/EdgeShard-remastered/llama-3.1-8b/
```

Notes:
- Ensure the destination directories exist on the remote machines.
- Update the remote paths and usernames to match your environment.
- If Windows scp is unavailable, install OpenSSH client, or use `pscp` (PuTTY) with similar syntax.
- If your nodes share a network filesystem, you can instead place shards in a shared location and point `config/config_hf.yaml` to the shared paths.

Quick start — single-node interactive inference
----------------------------------------------
1. Create and activate a virtual environment (optional but recommended).

Windows (PowerShell):
```powershell
conda create -n edgeshard python=3.10 -y
conda activate edgeshard
```

Linux / macOS:
```bash
conda create -n edgeshard python=3.10 -y
conda activate edgeshard
```

2. Install required Python packages:
```powershell
pip install -r requirements.txt
```

3. (Optional) Edit `config/config_hf.yaml` to match your environment (device names, master/worker IPs and ports, and other runtime settings).

4. Run interactive single-process inference:
```powershell
python inference_solo.py
```
- The script will load an HF model configuration + tokenizer and run an interactive REPL. Type messages and receive model output. Type `/exit` to quit.

Distributed (master/worker) run (experimental)
---------------------------------------------
This repository includes basic scaffolding to start an RPC-based master and worker processes. This path is experimental and may require adjustments to `config/config_hf.yaml` and network interfaces.

1. Configure `config/config_hf.yaml` with your `master` and `workers` entries (IP, port, interface names).
2. Start the master process (on the machine designated as master):
```powershell
python master.py
```
3. Start each worker process (on worker machines or additional terminals). Example for worker rank 1:
```powershell
python worker.py --worker_rank 1
```

Tip: Before launching, verify that each node has the correct shard file path as specified in `config/config_hf.yaml` (master’s `lm_head_weight_path`, and each worker’s `ckpt_path`).

Notes
-----
- Environment variables: the code relies on a few environment variables set at runtime from `config/config_hf.yaml` (for example `MASTER_ADDR`, `MASTER_PORT`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`). If you run into networking or RPC errors, verify those values match your network and interface names.
- Memory tuning: a few places set `PYTORCH_CUDA_ALLOC_CONF` and call `torch.cuda.empty_cache()` and `gc.collect()` to reduce memory fragmentation; you may need to tweak these for large models.
