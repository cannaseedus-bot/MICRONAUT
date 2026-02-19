# Concurrency Scaling vs Mesh Scaling

This project distinguishes two different scaling problems that are often conflated.

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

## 2) Optional volunteer/distributed mesh compute

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
3. Multiprocess/multithread workers
4. Horizontal replication across machines
5. Optional verified mesh for parallelizable sub-tasks

Mesh-first approaches are fragile unless the core serving plane is already stable.
