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

## Architecture

See [EdgeShard_v2_Technical_Architecture_Design_v0.2.md](./EdgeShard_v2_Technical_Architecture_Design_v0.2.md) for the full architecture document.

**Key invariants:**
- Master orchestrates; Workers execute; Shards compute.
- Master never participates in the token data path.
- Scheduler is a pure planner: snapshots in, immutable PlacementPlan out.
- KV cache belongs to `Session × Shard` and stays on the owning Worker.
- Prefill and decode are first-class runtime operations.

## Implementation Status

### ✅ Implemented (M0–M5)

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

### 🚧 Not Yet Implemented (M6–M10)

#### M6: Automatic Profiling
- [ ] Profile executor on Worker
- [ ] Forward pass latency measurement
- [ ] KV cache memory cost estimation
- [ ] Prefill/decode throughput measurement
- [ ] Profile storage (SQLite)
- [ ] Profile-driven scheduling hints

#### M7: Scheduler and Placement
- [ ] ClusterSnapshot and ProfileSnapshot data models
- [ ] Scheduling policy interface
- [ ] Default scheduling policy implementation
- [ ] PlacementPlan serialization to YAML
- [ ] Constraint satisfaction (memory, latency, topology)
- [ ] Heterogeneous device handling

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

# Run with coverage
pytest --cov=edgeshard --cov-report=html
```

**Note:** Tests require a local model in `models/` directory and CUDA GPU for runtime tests.

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
├── cli/                    # Typer CLI commands
├── api/                    # REST API schemas
├── common/                 # Shared utilities, config, errors
├── master/                 # Master server and managers
├── worker/                 # Worker daemon and hardware probe
├── scheduler/              # Placement planning (TODO)
├── profiler/               # Model profiling (TODO)
├── runtime/                # Shard runtime, adapters, pipeline
├── transport/              # Tensor data plane (gRPC)
├── models/                 # Model registry
├── deployment/             # Deployment backends (TODO)
├── observability/          # Metrics and tracing (TODO)
└── _grpc/                  # Generated gRPC code

proto/
├── edgeshard.proto         # Master/Worker control plane
└── shard.proto             # Shard-to-shard data plane

tests/
├── unit/                   # Unit tests by milestone
└── integration/            # Integration tests

examples/
├── service.yaml            # Example service specification
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
