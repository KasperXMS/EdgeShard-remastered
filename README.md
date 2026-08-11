# EdgeShard

> **⚠️ Unofficial Development**  
> This is an unofficial implementation of the EdgeShard architecture. It is not affiliated with or endorsed by the original EdgeShard project.

Config-driven distributed inference for heterogeneous edge and GPU resources.

EdgeShard automatically discovers worker capabilities, profiles model execution, plans model sharding and placement, deploys runtime instances, and serves distributed autoregressive inference — all from a single service YAML.

## Installation

### Prerequisites

- **Conda** (Miniconda or Anaconda)
- **Git**
- **GPU**: NVIDIA Driver compatible with CUDA 12.1 (for GPU setup)

### Setup Environment

Choose the appropriate environment for your hardware:

#### Option 1: NVIDIA GPU (CUDA 12.1)

```bash
# Clone the repository
git clone <repo-url>
cd EdgeShard-remastered

# Create conda environment with CUDA support
conda env create -f environment.yml

# Activate the environment
conda activate edgeshard
```

#### Option 2: AMD GPU (ROCm)

```bash
# Create conda environment with ROCm support
conda env create -f environment-rocm.yml

# Activate the environment
conda activate edgeshard-rocm
```

#### Option 3: CPU Only (no GPU)

```bash
# Create conda environment for CPU-only
conda env create -f environment-cpu.yml

# Activate the environment
conda activate edgeshard-cpu
```

### Install EdgeShard

```bash
# Install EdgeShard in editable mode
pip install -e .

# Verify installation
edgeshard --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Environment Details

**GPU Environment** (`environment.yml`):
- Python 3.10
- PyTorch 2.1.0 with CUDA 12.1
- Transformers 4.44.0
- pynvml for GPU monitoring

**CPU Environment** (`environment-cpu.yml`):
- Python 3.10
- PyTorch 2.1.0 (CPU only)
- Transformers 4.44.0
- No GPU-specific dependencies

### Troubleshooting

**CUDA driver mismatch**: If you see "NVIDIA driver is too old", your driver doesn't support CUDA 12.1. Update your driver or modify `environment.yml` to use `pytorch-cuda=11.8`.

**Transformers version**: Must be 4.44.0. Newer versions have incompatible Qwen2 API changes.

**Recreate environment**:
```bash
conda env remove -n edgeshard
conda env create -f environment.yml
```

**List all environments**:
```bash
conda env list
```

## Quick Start

```bash
pip install -e ".[all]"

# Start a master
edgeshard master start

# Register a worker
edgeshard worker start --master 192.168.1.10:10500

# Discover cluster
edgeshard node list --master 192.168.1.10:10500

# Start a shard (first shard, layers 0-11)
edgeshard shard start models/Qwen2.5-0.5B-Instruct \
    --shard-id shard-0 --layers 0:12 --first --port 50100

# Start a shard (last shard, layers 12-23 + LM head)
edgeshard shard start models/Qwen2.5-0.5B-Instruct \
    --shard-id shard-1 --layers 12:24 --last --port 50101

# Run distributed inference
edgeshard infer "Hello world" --shards localhost:50100,localhost:50101
```

---

## Usage Guide

### End-to-End Workflow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 1. Start │───▶│ 2. Start │───▶│ 3. Probe │───▶│ 4. Init  │───▶│ 5. Plan  │───▶│ 6. Infer │
│  Master  │    │  Workers │    │ Snapshot │    │  Config  │    │ Schedule │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                       auto-gen             DP optimize
                                                       service.yaml         + deploy shards
                                                       + cluster.yaml
```

### Step 1: Start Master

```bash
# Default config (master.yaml)
edgeshard master start

# Custom config
edgeshard master start --config my-master.yaml
```

Master listens on `0.0.0.0:10500` (gRPC control plane) by default.

### Step 2: Start Workers

Each worker auto-detects hardware (GPUs via NVML, CPU/memory via psutil) and registers with the master:

```bash
# On each worker node
edgeshard worker start --master 192.168.1.10:10500

# With custom config
edgeshard worker start --master 192.168.1.10:10500 --config worker.yaml
```

Workers send heartbeats every 10 seconds with hardware metrics.

### Step 3: Discover Cluster & Profile

```bash
# List workers
edgeshard node list --master 192.168.1.10:10500

# Show detailed metrics (GPU util, temp, power, network)
edgeshard node metrics --master 192.168.1.10:10500

# Auto-export cluster topology to cluster.yaml (for offline planning)
edgeshard cluster snapshot --master 192.168.1.10:10500 -o cluster.yaml

# Profile a model on the local device
edgeshard profile run Qwen/Qwen2.5-7B-Instruct --device cuda:0 --dtype float16

# List all stored profiles
edgeshard profile list
```

`edgeshard cluster snapshot` connects to the Master, fetches all worker information (devices, memory, network topology), and writes a `cluster.yaml` file. This file can be used for offline planning later without needing a live cluster.

Profiling measures:
- **Layer forward latency** — average single-layer forward pass time (ms)
- **KV cache per token** — memory per token for key-value cache (MB)
- **Prefill throughput** — tokens/second for initial prompt processing
- **Decode throughput** — tokens/second for autoregressive generation

### Step 4: Plan Placement

The scheduler uses DP optimization (from the EdgeShard paper) to decide which model layers go on which workers:

```bash
# Auto-generate service.yaml from model name (smart defaults for known models)
edgeshard service init Qwen/Qwen2.5-7B-Instruct -o service.yaml

# Plan with live cluster data (requires master running)
edgeshard plan service.yaml --master 192.168.1.10:10500 -o plan.yaml

# Plan offline with a cluster YAML
edgeshard plan service.yaml --cluster-yaml cluster.yaml -o plan.yaml
```

`edgeshard service init` auto-detects model architecture (layer count, hidden size, estimated memory) for known models (Qwen2.5, Llama2 families) and generates a complete service.yaml with sensible defaults.

**Service YAML** (auto-generated or hand-written):
```yaml
name: qwen25-7b-demo
model:
  name: Qwen/Qwen2.5-7B-Instruct
  revision: main
  dtype: float16
runtime:
  backend: torch
  max_batch_size: 1
  max_sequence_length: 4096
scheduling:
  policy: default           # default | latency-first | memory-balanced
  hints:
    latency_weight: 0.5     # >0.5 = prefer latency, <0.5 = prefer throughput
deployment:
  backend: docker
  shard_image: edgeshard/shard:latest
  resource_limits:
    memory: 16Gi
```

**Cluster YAML** (auto-generated by `edgeshard cluster snapshot` or hand-written for offline planning):
```yaml
source_worker_id: w-001
workers:
  - worker_id: w-001
    hostname: gpu-node-1
    available_memory_mb: 32768
    cpu_count: 16
    status: online
    devices:
      - device_id: cuda:0
        device_type: cuda
        name: NVIDIA GeForce RTX 4090
        total_memory_mb: 24576
  - worker_id: w-002
    hostname: jetson-node-1
    available_memory_mb: 16384
    cpu_count: 8
    status: online
    devices:
      - device_id: jetson:0
        device_type: jetson
        name: Jetson AGX Orin
        total_memory_mb: 32768
bandwidth_matrix:
  - from: w-001
    to: w-002
    bandwidth_mb_per_s: 125.0   # 1 Gbps = 125 MB/s
```

**Output** (`plan.yaml`):
```yaml
service_name: qwen25-7b-demo
version: 1
model: Qwen/Qwen2.5-7B-Instruct
model_revision: main
dtype: float16
shards:
  - shard_index: 0
    worker_id: w-001
    layer_start: 0
    layer_end: 16
    device: cuda:0
  - shard_index: 1
    worker_id: w-002
    layer_start: 16
    layer_end: 32
    device: jetson:0
metadata:
  total_layers: 32
  num_shards: 2
  num_workers: 2
```

### Step 5: Deploy & Infer

```bash
# Start shards according to the plan
edgeshard shard start Qwen/Qwen2.5-7B-Instruct \
    --shard-id shard-0 --layers 0:16 --first --host 0.0.0.0 --port 50100

edgeshard shard start Qwen/Qwen2.5-7B-Instruct \
    --shard-id shard-1 --layers 16:32 --last --host 0.0.0.0 --port 50101

# Run inference
edgeshard infer "What is edge computing?" \
    --shards gpu-node-1:50100,jetson-node-1:50101 \
    --max-tokens 200
```

---

### Scheduling Policies

| Policy | Description | Best For |
|--------|-------------|----------|
| `default` | Auto-selects based on `latency_weight` hint | General purpose |
| `latency-first` | Minimizes total sequential latency (Algorithm 1) | Single-user, interactive |
| `memory-balanced` | Minimizes pipeline bottleneck (Algorithm 2) | Multi-user, batch serving |

**Algorithm 1 (Latency)**: DP minimizes `Σ comp_time + Σ comm_time` across all layers. Best when response time matters.

**Algorithm 2 (Throughput)**: DP minimizes `max(stage_time)` across pipeline stages. Best when maximizing tokens/second matters.

Both algorithms enforce:
- **Privacy constraint**: First layer always on source worker (raw input never leaves the user's device)
- **Memory constraint**: `layers × per_layer_mem + seq_len × kv_cache_per_token ≤ device_budget`
- **Contiguous partitions**: Each shard gets a contiguous block of layers (no intra-layer splitting)

---

### CLI Reference

```
edgeshard
├── master
│   └── start [--config master.yaml]
├── worker
│   └── start --master HOST:PORT [--config worker.yaml]
├── node
│   ├── list [--master HOST:PORT]
│   └── metrics [--master HOST:PORT]
├── cluster
│   └── snapshot [--master HOST:PORT] [-o cluster.yaml]
├── profile
│   ├── run MODEL [--device cuda:0] [--dtype float16]
│   └── list
├── plan SERVICE_YAML [--master HOST:PORT | --cluster-yaml FILE] [-o plan.yaml]
├── shard
│   └── start MODEL --shard-id ID --layers START:END [--first] [--last] [--port 50100]
├── infer PROMPT --shards HOST:PORT[,HOST:PORT...] [--max-tokens N]
└── service
    ├── init MODEL [-o service.yaml] [--name NAME] [--dtype float16] [--max-seq-len N] [--policy default]
    ├── deploy SERVICE_YAML
    ├── list
    ├── status NAME
    └── stop NAME
```

---

### Python API

```python
from edgeshard.scheduler import (
    schedule, ClusterSnapshot, ProfileSnapshot, SchedulingPolicy,
    PlacementPlan,
)
from edgeshard.common.config import ModelSpec

# Build cluster snapshot
cluster = ClusterSnapshot(
    workers=[...],
    bandwidth_matrix={("w-001", "w-002"): 125.0},
    source_worker_id="w-001",
)

# Load profiles
profiles = ProfileSnapshot(entries=[...])

# Schedule
plan = schedule(
    model=ModelSpec(name="Qwen/Qwen2.5-7B-Instruct"),
    cluster=cluster,
    profiles=profiles,
    policy=SchedulingPolicy(name="latency-first"),
)

# Use plan
for shard in plan.shards:
    print(f"Shard {shard.shard_index}: layers {shard.layer_start}-{shard.layer_end} "
          f"on {shard.worker_id}:{shard.device}")

# Serialize
plan.to_yaml("plan.yaml")
```

## Architecture

See [EdgeShard_v2_Technical_Architecture_Design_v0.2.md](./EdgeShard_v2_Technical_Architecture_Design_v0.2.md) for the full architecture document.

**Key invariants:**
- Master orchestrates; Workers execute; Shards compute.
- Master never participates in the token data path.
- Scheduler is a pure planner: snapshots in, immutable PlacementPlan out.
- KV cache belongs to `Session × Shard` and stays on the owning Worker.
- Prefill and decode are first-class runtime operations.

## Implementation Status

### ✅ Implemented (M0–M7)

#### M0: Package + CLI Skeleton
- [x] Python package with `pip install -e .` support
- [x] Typer-based CLI with all subcommands
- [x] Pydantic configuration schemas (MasterConfig, WorkerConfig, ServiceSpec)
- [x] YAML configuration loading
- [x] Structured logging with Rich

#### M1: Local Shard Correctness
- [x] Qwen2/Qwen2.5 ModelAdapter implementation
- [x] Layer range loading from safetensors checkpoints
- [x] Embedding layer (first shard) and LM head (last shard) support
- [x] Forward pass through transformer layers
- [x] Model-specific KV cache initialization
- [x] Model registry for adapter discovery

#### M2: Stateful KV Runtime
- [x] Greedy decoder with autoregressive generation loop
- [x] KV cache accumulation across prefill and decode steps
- [x] Session state management (create, prefill, decode, release)
- [x] KV cache memory tracking and statistics
- [x] GenerationConfig for controlling output (max_tokens, EOS, etc.)
- [x] GenerationResult with token IDs and decoded text

#### M3: Master/Worker Control Plane
- [x] Protobuf definitions for gRPC control plane
- [x] Worker daemon with automatic registration
- [x] Heartbeat mechanism for health monitoring
- [x] Master gRPC server with WorkerService
- [x] WorkerManager for cluster state tracking
- [x] Hardware discovery (NVIDIA GPUs via NVML, CPU/memory via psutil)
- [x] Cluster discovery API (`edgeshard node list`)

#### M4: Distributed Inference Data Plane
- [x] Protobuf definitions for shard-to-shard tensor transfer
- [x] gRPC tensor transport (serialization/deserialization)
- [x] ShardServer for receiving tensors from other shards
- [x] PipelineOrchestrator for multi-shard coordination
- [x] PipelineDecoder for distributed greedy decoding
- [x] ShardDaemon for standalone shard processes
- [x] CLI commands: `edgeshard shard start`, `edgeshard infer`

#### M5: Resource Discovery Enhancement
- [x] Detailed GPU metrics (utilization, temperature, power)
- [x] CPU metrics (utilization, memory usage)
- [x] Network topology discovery (latency measurement)
- [x] Bandwidth estimation
- [x] Jetson device profiling (tegrastats integration)
- [x] Enhanced heartbeat with dynamic metrics
- [x] CLI command: `edgeshard node metrics`

#### M6: Automatic Profiling
- [x] Profile executor for measuring model performance
- [x] Layer forward pass latency measurement
- [x] KV cache memory cost estimation per token
- [x] Prefill/decode throughput measurement
- [x] Profile storage (SQLite)
- [x] CLI commands: `edgeshard profile run`, `edgeshard profile list`
- [x] Master Profile RPC integration

#### M7: Scheduler and Placement
- [x] ClusterSnapshot and ProfileSnapshot data models
- [x] Bandwidth matrix and source worker tracking
- [x] Scheduling policy interface (SchedulingPolicyBase)
- [x] DP-based latency optimization (Algorithm 1 from EdgeShard paper)
- [x] DP-based throughput optimization (Algorithm 2 from EdgeShard paper)
- [x] Single-device placement fallback
- [x] Memory constraint satisfaction (model weights + KV cache)
- [x] Privacy constraint (first layer on source worker)
- [x] PlacementPlan YAML serialization/deserialization
- [x] Data bridges (WorkerManager → ClusterSnapshot, ProfileStore → ProfileSnapshot)
- [x] Model info extraction from config.json and known model registry
- [x] CLI command: `edgeshard plan` (live cluster + offline modes)
- [x] CLI command: `edgeshard cluster snapshot` (auto-export cluster.yaml from live Master)
- [x] CLI command: `edgeshard service init` (auto-generate service.yaml from model name)
- [x] Protobuf messages for placement transmission

### 🚧 Not Yet Implemented (M8–M10)

#### M8: Deployment
- [ ] DeploymentBackend interface
- [ ] Docker deployment backend
- [ ] Shard container lifecycle management
- [ ] Health checks and readiness probes
- [ ] Service deployment orchestration
- [ ] Rollback on failure

#### M9: Serving API
- [ ] FastAPI REST API for inference
- [ ] OpenAI-compatible chat completions endpoint
- [ ] Streaming responses
- [ ] Request queuing and batching
- [ ] Rate limiting and authentication
- [ ] Service status and metrics endpoints

#### M10: Observability
- [ ] Prometheus metrics export
- [ ] OpenTelemetry tracing integration
- [ ] Distributed request tracking
- [ ] Performance dashboards (Grafana)
- [ ] Alerting rules

### 📋 Additional TODO

- [ ] Multi-model serving support
- [ ] Dynamic repartitioning (re-shard on load change)
- [ ] KV cache checkpointing and replication
- [ ] Worker failover and high availability
- [ ] Quantization support (INT8, INT4)
- [ ] Additional model families (LLaMA, Mistral, etc.)
- [ ] NCCL/RDMA transport optimization
- [ ] Kubernetes backend (K3s)
- [ ] Master HA (multi-master with consensus)

## Testing

```bash
# Run all tests
pytest

# Run specific milestone tests
pytest tests/unit/test_m1_local_shard.py -v
pytest tests/unit/test_m2_stateful_kv.py -v
pytest tests/unit/test_m3_control_plane.py -v
pytest tests/unit/test_m4_distributed.py -v
pytest tests/unit/test_m5_resource_discovery.py -v
pytest tests/unit/test_m6_profiling.py -v
pytest tests/unit/test_m7_scheduler.py -v

# Run M7 scheduler tests only (no GPU required)
pytest tests/unit/test_m7_scheduler.py -v

# Run with coverage
pytest --cov=edgeshard --cov-report=html
```

**Note:** M1-M4 tests require a local model in `models/` directory and CUDA GPU. M5-M7 tests use synthetic data and run without GPU.

## Real-Machine Testing

For distributed inference testing on real hardware, see [M4_TESTING_GUIDE.md](./M4_TESTING_GUIDE.md).

## Development

```bash
# Install with all dependencies
pip install -e ".[all]"

# Generate gRPC code from protobuf
python scripts/generate_proto.py

# Code quality
ruff check .
ruff format .
mypy src/

# Run tests
pytest
```

## Project Structure

```
src/edgeshard/
├── cli/                    # Typer CLI commands (master, worker, node, profile, plan, shard, infer)
├── api/                    # REST API schemas
├── common/                 # Shared utilities, config, errors, identifiers
├── master/                 # Master server, worker manager
├── worker/                 # Worker daemon, hardware probe, network probe
├── scheduler/              # Placement planning (M7)
│   ├── planner.py          #   Core schedule() function — pure planner
│   ├── policies/           #   DP algorithms: latency + throughput optimization
│   │   └── default.py      #   Algorithm 1 & 2 from EdgeShard paper
│   ├── model_info.py       #   Model metadata from config.json
│   ├── bridge.py           #   WorkerManager/ProfileStore → typed snapshots
│   ├── snapshot.py         #   ClusterSnapshot, ProfileSnapshot data models
│   └── placement.py        #   PlacementPlan, ShardPlacement (YAML serializable)
├── profiler/               # Model profiling (M6)
│   ├── executor.py         #   ProfileExecutor — layer latency, KV cache, throughput
│   ├── store.py            #   ProfileStore — SQLite persistence
│   └── profile_data.py     #   ProfileResult data model
├── runtime/                # Shard runtime, adapters, pipeline, KV cache
├── transport/              # Tensor data plane (gRPC)
├── models/                 # Model registry (Qwen2/Qwen2.5)
├── deployment/             # Deployment backends (TODO: M8)
├── observability/          # Metrics and tracing (TODO: M10)
└── _grpc/                  # Generated gRPC code

proto/
├── edgeshard.proto         # Master/Worker control plane + placement messages
└── shard.proto             # Shard-to-shard data plane

tests/unit/
├── test_m1_local_shard.py       # Model adapter correctness (GPU required)
├── test_m2_stateful_kv.py       # KV cache + generation (GPU required)
├── test_m3_control_plane.py     # gRPC registration + heartbeat
├── test_m4_distributed.py       # Multi-shard pipeline (GPU required)
├── test_m5_resource_discovery.py # GPU/network metrics
├── test_m6_profiling.py         # Profile executor + store
└── test_m7_scheduler.py         # DP scheduling (no GPU required)

examples/
├── service.yaml            # Service specification (what to serve)
├── cluster.yaml            # Cluster topology (for offline planning)
├── m2_single_shard_generation.py
└── m4_distributed_inference.py
```

## Supported Models

- **Qwen2 / Qwen2.5** — via `Qwen2Adapter`
- More model families planned (LLaMA, Mistral, etc.)

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU inference)
- PyTorch 2.2+
- transformers 4.40+
- grpcio 1.62+

## License

Apache-2.0
