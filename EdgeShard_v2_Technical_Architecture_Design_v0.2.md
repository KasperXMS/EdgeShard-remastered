# EdgeShard v2 Technical Architecture Design

**Version:** 0.2  
**Status:** Architecture Baseline  
**Date:** 2026-08-07

## 1. Scope and Design Goals

EdgeShard v2 is a config-driven distributed inference system for heterogeneous edge and GPU resources. It automatically discovers worker capabilities, profiles model execution, plans model sharding and placement, deploys runtime instances, and serves distributed autoregressive inference.

**Primary goal:** turn a model service specification into a runnable distributed inference pipeline without users manually assigning layers, IP addresses, or worker processes.

### Goals

- Greenfield implementation with explicit, testable interfaces.
- Master/Worker architecture with clear state ownership.
- Config-driven operation through a Python package and terminal CLI.
- Automatic worker registration, hardware discovery, telemetry, and model profiling.
- Profile-driven heterogeneous layer partitioning and placement.
- Stateful shard runtime with shard-local KV cache.
- Direct Worker-to-Worker inference data path; Master stays out of token forwarding.
- Containerized deployment with Docker as the first backend.
- Reuse mature infrastructure wherever it does not define EdgeShard's core value.

### Non-goals for v2.0

Kubernetes Operator/CRDs, dynamic repartitioning, stateful failover/KV replication, multi-model global scheduling, per-layer mixed quantization, agent serving, Master HA, and a custom monitoring dashboard are deferred.

## 2. Architectural Principles

1. Master orchestrates; Workers execute; Shards perform model computation.
2. Master never participates in the token data path.
3. Worker and Shard are different entities: one Worker may host multiple Shards and profiler jobs.
4. Scheduler is a pure planner: input snapshots in, immutable PlacementPlan out.
5. Deployment executes a PlacementPlan but does not change scheduling decisions.
6. KV cache belongs to a specific Session × Shard and never travels as a control-plane object.
7. Prefill and decode are first-class runtime operations, not one generic remote `forward()`.
8. Model-specific implementation is isolated behind ModelAdapter.
9. Control plane and tensor data plane use separate interfaces and may evolve independently.
10. External infrastructure is replaceable behind adapters; EdgeShard-owned logic stays narrow.

## 3. System Overview

```text
                         CLI / Python SDK / REST
                                   |
                                   v
+----------------------------------------------------------------+
|                       EdgeShard Master                         |
|                                                                |
|  API / Service Manager                                         |
|        |                                                       |
|        +--> Resource Manager <---- Worker state                |
|        +--> Profile Manager  <---- Profile Store               |
|        +--> Scheduler -------> PlacementPlan                   |
|        +--> Deployment Manager                                 |
+-------------------------------+--------------------------------+
                                |
                       Control Plane (gRPC)
                                |
                 +--------------+--------------+
                 |              |              |
                 v              v              v
             Worker A       Worker B       Worker C
             AGX Orin       RTX 3090       RTX 4090
                 |              |              |
             Shard 0        Shard 1        Shard 2
                 +--------------+--------------+
                     Direct Tensor Data Plane
```

### Core entities

- **Master:** logical cluster control plane.
- **Worker:** node-level execution daemon.
- **Shard:** deployed model partition.
- **Session:** one inference context.
- **PlacementPlan:** immutable deployment/execution plan.

## 4. Master Architecture

```text
Master
|-- API Server
|-- Service Manager
|-- Worker Manager
|-- Resource Manager
|-- Profile Manager
|-- Scheduler Manager
|-- Deployment Manager
`-- State Store
```

Master stores service/profile/plan metadata, but not model weights, KV cache, or hidden states. Scheduling always consumes an immutable cluster snapshot.

## 5. Worker Architecture

```text
Worker Daemon
|-- Registration Client
|-- Hardware Probe
|-- Telemetry Collector
|-- Profile Executor
|-- Model Cache Manager
|-- Deployment Executor
|-- Runtime Manager
`-- Metrics / Trace Exporter
```

A Worker is not a Shard. One Worker may host multiple service Shards and profiler jobs.

## 6. Resource and Telemetry Model

Device-specific sources are normalized behind metric providers:

- NVML/DCGM for discrete NVIDIA GPUs.
- `tegrastats` for Jetson.
- psutil/node_exporter for host CPU/RAM.
- lightweight network probes for RTT/bandwidth estimates.

The scheduler consumes a normalized `ClusterState`, never raw monitoring output.

## 7. Profiling Architecture

Profiling is deployment-oriented: estimate shard memory, prefill/decode performance, KV cost, and communication cost. It supports partial profiling on devices that cannot load the full model. Results are stored in SQLite initially and keyed by model revision, device fingerprint, dtype, and runtime/backend version.

## 8. Scheduler and PlacementPlan

The Scheduler is pure planning logic:

```python
PlacementPlan schedule(
    ModelSpec model,
    ClusterSnapshot cluster,
    ProfileSnapshot profiles,
    SchedulingPolicy policy,
)
```

It may not start containers or mutate Workers. The first policy can reimplement the original EdgeShard algorithm on the new interfaces.

## 9. Runtime Architecture

```text
ModelShard
|-- ModelAdapter
|-- LayerRange
|-- SessionManager
|-- KV Cache
|-- TensorTransport endpoint
`-- ExecutionEngine
```

Runtime API:

```text
create_session()
prefill()
decode()
release_session()
```

**KV ownership rule:** KV belongs to Session × Shard and remains on the Worker responsible for those layers.

## 10. Control Plane and Data Plane

- **Control plane:** Master ↔ Worker using gRPC for registration, profiling, lifecycle, and health operations.
- **Data plane:** Worker/Shard ↔ Worker/Shard for hidden-state transfer. Start with a portable transport implementation and optimize later with torch.distributed/NCCL.
- **Invariant:** Master never forwards per-token hidden states.

## 11. Model Abstraction and Loading

Model-specific structure is hidden behind `ModelAdapter`. Start with Qwen2/Qwen2.5, then add other model families. Reuse Transformers, Safetensors, Hugging Face Hub, and Accelerate for checkpoint/model handling. Workers maintain local caches and should load only shard-relevant weights where practical.

## 12. Deployment Architecture

The Master sends placement/lifecycle commands to Workers. Workers use their local Docker Engine; the Docker socket is never exposed remotely. `DeploymentBackend` is abstract so a later K3s/Kubernetes backend can be added without changing Scheduler or Runtime semantics.

## 13. State Ownership

- ServiceSpec, profiles, PlacementPlan: Master.
- Hardware facts and telemetry source: Worker.
- Local model cache and container lifecycle: Worker.
- Model state, Session state, KV cache: Shard Runtime.
- Hidden states: ephemeral data plane.

## 14. Package and CLI

Install as one Python package:

```bash
pip install -e .
edgeshard master start
edgeshard worker start --master 192.168.1.10:10500
edgeshard node list
edgeshard profile run Qwen/Qwen2.5-7B-Instruct
edgeshard plan service.yaml
edgeshard service deploy service.yaml
```

Recommended source tree:

```text
src/edgeshard/
|-- cli/
|-- api/
|-- common/
|-- master/
|-- worker/
|-- scheduler/
|-- profiler/
|-- runtime/
|-- transport/
|-- models/
|-- deployment/
`-- observability/
```

## 15. Configuration

Use separate Master, Worker, and Service YAML schemas. Service YAML expresses desired model/runtime/scheduler/deployment configuration, not explicit layer mappings.

## 16. End-to-End Deployment Flow

```text
CLI/API
 -> validate ServiceSpec
 -> snapshot resources
 -> ensure profiles
 -> schedule
 -> PlacementPlan
 -> prepare Workers
 -> start shard containers
 -> establish data-plane topology
 -> health check
 -> Service READY
```

## 17. Mature Components vs EdgeShard-Owned Components

**Reuse:** Typer, Pydantic/YAML, FastAPI, gRPC/protobuf, Transformers/Safetensors, NVML/DCGM, tegrastats, psutil/node_exporter, Prometheus/Grafana, OpenTelemetry, Docker Engine, SQLite.

**EdgeShard core:** ModelAdapter, Shard Runtime, tensor transport abstraction, profiling orchestration/performance model, Scheduler/Placement, and Master orchestration semantics.

## 18. Implementation Milestones

- M0: package + CLI skeleton.
- M1: local shard correctness.
- M2: stateful KV runtime.
- M3: Master/Worker control plane.
- M4: remote data plane.
- M5: hardware/resource discovery.
- M6: automatic profiling.
- M7: scheduler and PlacementPlan.
- M8: Docker deployment.
- M9: serving API.
- M10: metrics and tracing.

**Important sequencing rule:** do not implement the scheduler before the distributed runtime and state semantics are proven correct.

## 19. v2.0 Definition of Done

- Package installs and exposes terminal commands.
- One Master and multiple Workers register automatically.
- Workers expose normalized hardware/resource state.
- Missing model/device profiles are generated automatically.
- A service YAML produces an immutable PlacementPlan and automatic deployment.
- At least one supported model runs across at least two heterogeneous nodes.
- Prefill/decode use shard-local KV and distributed greedy output matches the single-node reference.
- Master is not in the per-token data path.
- Worker-local Docker deployment and basic observability work end to end.

## 20. Future Extension Points

K3s/Kubernetes, dynamic repartitioning, worker failover, KV checkpoint/replication, quantization, multi-model serving, additional model families, NCCL/RDMA transports, and Master HA fit behind existing interfaces.

## 21. Architecture Baseline

> **The Master maintains cluster intelligence and produces immutable execution plans; Workers manage node-local resources and runtime instances; Shards own model execution, session state, and KV cache; distributed inference uses direct Worker-to-Worker data paths.**

Any future component that violates this boundary should be treated as an explicit architecture change rather than an implementation convenience.
