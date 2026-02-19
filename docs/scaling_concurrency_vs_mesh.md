# Concurrency Scaling vs Mesh Scaling

This project distinguishes scaling concerns that are often conflated:

1. **Online user concurrency** (many simultaneous requests)
2. **Local CPU worker parallelism** (many execution lanes on one workstation/server)
3. **Optional distributed/volunteer mesh compute** (untrusted remote workers)

These are related, but they are not the same architecture problem.

## 1) 1000 online users on one machine (concurrency)

This is a serving/runtime problem.

You generally need:

- Async API server (FastAPI/Node/Rust async)
- Request queue
- Dynamic batching window
- A shared model instance loaded once

You generally do **not** need:

- 1000 model copies
- 1000 dedicated expert clusters
- 1000 world models

### Reference serving pipeline

```text
Internet users
  -> load balancer
  -> async API replicas
  -> inference queue
  -> shared model workers
  -> batch processor
```

### Why batching matters

Many user requests can be merged into one forward pass (`batch_size = N`) when latency budget allows. This is usually the first and highest-leverage optimization for throughput.

## 2) Local "1000-core" style execution on one workstation

This is **parallel task scheduling** on a shared-memory host, not "1000 brains".

Typical reality:

- 32–128 physical cores are common on high-end machines
- Hyperthreads increase logical schedulable workers
- "1000 workers" can be lightweight processes/tasks mapped onto available cores
- RAM is shared across workers (major advantage)

### Correct local shape

```text
Shared model backbone (loaded once)
  -> worker pool (many logical workers)
  -> deterministic router
  -> shared memory/state layer
  -> aggregate output
```

### Recommended implementation pattern

- `torch.multiprocessing` (or equivalent runtime)
- Shared-memory model weights (`share_memory()` / fork-friendly process model)
- Core-count-sized process pool, with larger logical task queue
- Deterministic routing (e.g., stable hash / modulo expert assignment)

### What to avoid

- Spawning full model replicas per worker
- Per-worker world model copies
- Per-worker persistence authority

A local 1000-worker queue should represent **1000 execution lanes over shared components**, not 1000 duplicated model stacks.

## 3) Optional volunteer/distributed mesh compute

This is a separate, optional system for parallel side workloads.

```text
Clients (browser/native workers)
  -> coordinator
  -> task shard dispatch
  -> redundant compute
  -> result verification
  -> central aggregation
```

### Trust model requirements

Client-side compute is untrusted by default. Coordinator must include:

- Redundant execution (N-of-M agreement)
- Hash validation of payload/result
- Replayable verification logs
- Quarantine of outlier/malicious workers

### Good mesh candidates

- Monte Carlo rollouts
- Speculative decoding branches
- Embedding similarity batches
- Expert scoring/ranking side paths

### Keep central authority for

- Final response assembly
- Core model integrity and secrets
- Persistent state/memory writes
- Routing policy and admission control

## Recommended scaling order

1. Async serving + queueing
2. Dynamic batching
3. Shared-memory multiprocess worker pool on single host
4. Horizontal replication across machines
5. Optional verified mesh for parallelizable sub-tasks

Mesh-first approaches are fragile unless the core serving plane is already stable.
