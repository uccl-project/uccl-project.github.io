---
title: "mKernel: Fast Multi-GPU, Multi-Node Fused Kernels"
slug: mkernel
description: "mKernel is a collection of multi-GPU, multi-node fused kernels that put intra-node NVLink communication, inter-node RDMA, and compute inside a single persistent  kernel."
category:
  - One
tags:
  - Fused Kernels
  - RDMA
pubDate: 2026-05-08
cover: https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/mkernel/mkernel.png
coverAlt: mKernel
author: UCCL Team
---

<p>
<strong>By: the UCCL team
<br>
Date: May 8, 2026
</strong>
</p>

<div class="tldr">
<p>
<strong>mKernel</strong> is a collection of fast multi-GPU, multi-node fused kernels that enables intra-node NVLink communication, inter-node RDMA, and dense compute inside a <em>single persistent CUDA kernel</em>. Compute and communication overlap at tile granularity; CTAs self-assign roles (compute / intra-comm / inter-send / inter-reduce). Networking is GPU-driven via <code>libibverbs</code> on both ConnectX-7 and AWS EFA — there is no dependency on NCCL or NVSHMEM.
</p>
<p>
Code: <a href="https://github.com/uccl-project/mKernel">github.com/uccl-project/mKernel</a>
</p>
</div>

## The problem: host-driven communication is the long pole

AI training and serving are increasingly limited by communication at scale. In production, communication can consume **43.6% of the forward pass and 32% of end-to-end training time** on GPUs, and inter-device communication can account for **up to 47% of total execution time** across popular MoE models and frameworks. 

The traditional model is **host-driven**: the CPU runs the control path, calls into a library (NCCL/NVSHMEM), and the library issues the collective. It is increasingly mismatched with modern AI workloads for two reasons:

1. **Fine-grained overlap requires sub-kernel scheduling.** Compute kernels produce data incrementally — a tile at a time. The most efficient schedule communicates ready chunks the moment their dependencies are satisfied, not when the next library call returns. Host-driven systems overlap by launching compute and communication on separate streams, but their decisions are still made at coarse *kernel boundaries* — leaving tile-level overlap on the table.
2. **CPU-mediated control becomes visible as accelerators get faster.** Per-chip throughput is now multi-PFLOP-scale and intra-rack bandwidth is hundreds of TB/s. At these speeds, even microsecond-scale host orchestration overhead — a `cudaLaunchKernel`, a CPU-side "all writes done" check, an inter-stream event — shows up directly as pipeline bubbles.

The natural answer is **GPU-driven communication**: let the GPU itself trigger fine-grained transfers, fused into the same kernel as the compute, so producer tiles can be pushed to peers the instant they are ready. However, most existing kernel libraries stop at a single node, if not, a single GPU. 

mKernel is our attempt at the missing piece: a GPU-driven, **fused** kernel design that delivers fine-grained compute–communication overlap across both intra-node NVLink and inter-node RDMA, while staying portable across NIC backends (ConnectX-7 today, AWS EFA today, more on the way) without depending on NCCL or NVSHMEM.

## What mKernel does

mKernel is a small, focused library of persistent CUDA kernels — one per workload — each of which fuses intra-node NVLink communication, inter-node RDMA, and dense compute into a single kernel launch.

- **Multi-GPU + multi-node, in one kernel.** Intra-node NVLink and inter-node RDMA both live inside the same persistent kernel. Tiles are produced by compute CTAs and consumed by communication CTAs (and vice-versa) without ever leaving the kernel.
- **Fine-grained intra-kernel overlap.** Compute and communication overlap at *tile/chunk* granularity, covering both the intra-node and inter-node legs. There is no "the collective finishes, then the GEMM starts" boundary inside the kernel.
- **Persistent kernel with SM specialization.** CTAs self-assign roles — `compute`, `intra-comm`, `inter-send`, `inter-reduce` — based on a small role table the host sets up before launch. The split is tunable per shape.
- **GPU-driven networking, built on `libibverbs`.** mKernel uses GPU-initiated RDMA writes (and write-with-immediate) without depending on NCCL or NVSHMEM. The same on-GPU kernel runs against either ConnectX-7 (`-DINTERNODE_BACKEND_IBVERBS`) or AWS EFA / SRD (`-DINTERNODE_BACKEND_EFA`); only the host-side proxy / session implementation differs. This is what makes the design portable across heterogeneous clouds.

## The five fused kernels

| Kernel | What it fuses | One-line description |
|---|---|---|
| **AllGather + GEMM** | AllGather → GEMM | Each rank holds a shard of `A`. While ranks gather peers' shards over NVLink/RDMA, the local GEMM consumes tiles as soon as they arrive. The matmul starts well before the collective finishes. |
| **GEMM + AllReduce** | GEMM → AllReduce | Computes `C = A @ B` and reduces partial outputs across all 16 ranks in one launch. Output tiles are pushed into the reduction tree the instant they're produced, hiding the AllReduce inside the GEMM tail. |
| **MoE Dispatch + GEMM** | All-to-All dispatch → grouped GEMM | Routes MoE tokens to their expert ranks (intra-node NVLink + inter-node all-to-all) and runs the per-expert grouped GEMM in the same kernel. Tokens are matmul'd as soon as they land — no staging buffer round-trip. |
| **Ring Attention** | Ring KV exchange → FlashAttention | Sequence-parallel attention across 16 ranks: each step rotates a KV chunk around the ring while the local FlashAttention consumes the previously-received chunk. Compute and ring send/recv run concurrently in one persistent kernel. |
| **GEMM + ReduceScatter** | GEMM → ReduceScatter | Computes `C = A @ B` and reduce-scatters the output across ranks. Each output tile is reduced and forwarded to its owning rank as soon as it's produced — the scatter overlaps the GEMM rather than following it. |

All five kernels share the same backend abstraction; the only thing that changes between CX-7 and EFA deployments is the proxy / session header (`session.h` vs. `session_efa.h` in `include/comm/internode/`).

## Results on AWS EFA (2 × 8 × H200, 16x 200 Gb/s EFA)

mKernel is benchmarked against the best published baseline for each workload — NCCL, Triton-distributed, Flux, Mercury, MagiAttention, Transformer-Engine-CP, and ring-flash-attention — with steady-state (no per-iter sync) timing. Lower is better; mKernel is the leftmost bar in each plot.

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/mkernel/ag_gemm_efa.png" alt="AllGather + GEMM on EFA" width="700"/>
  <br><em>AllGather + GEMM. mKernel leads NCCL by 18–30% at M ≤ 16k and edges Triton-distributed at the largest shapes.</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/mkernel/gemm_ar_efa.png" alt="GEMM + AllReduce on EFA" width="700"/>
  <br><em>GEMM + AllReduce. The fused tile-level reduction hides the AllReduce inside the GEMM tail; mKernel leads Triton-distributed by 10–26% on M ≥ 4k.</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/mkernel/dispatch_gemm_efa.png" alt="MoE Dispatch + GEMM on EFA" width="700"/>
  <br><em>MoE Dispatch + GEMM. The largest gap: −68 to −77% vs. NCCL across 8k–131k tokens. Tokens are matmul'd as soon as they land, no staging.</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/mkernel/ring_attention_efa.png" alt="Ring Attention on EFA" width="700"/>
  <br><em>Ring Attention. mKernel leads MagiAttention by 30–67% across sequence lengths 768 → 12 288. Single persistent kernel = ring rotation overlapped with FlashAttention.</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/mkernel/gemm_rs_efa.png" alt="GEMM + ReduceScatter on EFA" width="700"/>
  <br><em>GEMM + ReduceScatter. mKernel leads NCCL by 8–17% at M ≥ 4k.</em>
</p>

The pattern across all five plots: the gap is largest at small/medium shapes — exactly the latency-bound regime where launch overhead, host-side synchronization, and coarse pipelining hurt the most, and where keeping everything inside one persistent kernel matters most. The gap narrows but stays positive at the largest shapes, where bandwidth dominates and there is less wall-clock to hide.

mKernel also runs on ConnectX-7 with the same kernels and a different proxy backend; results there follow the same shape (see the [README](https://github.com/uccl-project/mKernel) for the full plot set).

## Roadmap

- ✅ Fused, GPU-driven multi-node kernels (AG+GEMM, GEMM+AR, dispatch+GEMM, ring attention, GEMM+RS).
- ✅ ConnectX-7 and AWS EFA backends behind a single kernel surface.
- 🚧 Full support for heterogeneous accelerators and NICs, with topology-aware accelerator/NIC discovery, placement, and routing.
- 🚧 Inter-node *megakernels*: collapsing several fused steps into a single persistent kernel that spans an entire transformer layer.
- 🚧 Blackwell GPU support.

