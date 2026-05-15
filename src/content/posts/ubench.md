---
title: "CommBench: Can LLMs Write Efficient GPU Communication code?"
slug: llm-gpu-comm-kernels
description: 
category:
  - One
tags:
  - CUDA
  - RDMA
  - NCCL
  - mscclpp
  - LLM
  - Benchmark
pubDate: 2026-05-13
cover: https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/blog-placeholder-1.avif
coverAlt: LLM GPU Comm Kernels Benchmark
author: Shuang Ma, xx, xx, xx, Yang Zhou, and the UCCL Team.
---

<p>
<strong>By: Shuang Ma, xx, xx, xx, Yang Zhou, and the UCCL Team.
<br>
Date: May 13, 2026
</strong>
</p>

<div class="tldr">
<p>
Today's frontier LLMs write great single-device code but consistently fail on multi-device GPU communication — the kind of code that actually bottlenecks large-scale LLM training, post-training, and inference. We present a benchmark spanning <strong>NCCL</strong>, <strong>mscclpp</strong>, <strong>RDMA verbs</strong>, and <strong>compute-communication fusion</strong> kernels, evaluate closed and open frontier models with a compilation-feedback loop, and share case studies of where and why models break down. We also outline a post-training research agenda to close this gap.
</p>
<p>
Dataset and benchmark scripts: <a href="https://github.com/uccl-project/llm-for-gpu-comm/tree/main">CommBench</a>
</p>
</div>

## Why Multi-Device Coding Matters — and Why LLMs Are Bad at It

Communication and compute-communication fusion sit at the critical path of every serious LLM workload today. In production training, communication can consume **43.6% of the forward pass** [1]; in MoE inference with wide expert parallelism, inter-device communication accounts for **up to 47% of total execution time** [2]. Getting this code **right** and **fast** is not a nice-to-have.

Three forces are making hand-written, NCCL-style communication increasingly inadequate:

**GPUs are getting faster, so communication must move onto the GPU.** Per-chip throughput is now multi-PFLOP-scale. At these speeds, the CPU intervention in conventional collective libraries — a `cudaLaunchKernel`, an inter-stream event, a host-side "all writes done" check — shows up directly as a pipeline bubble. Communication needs to be **GPU-initiated**: triggered from inside the kernel, without bouncing through the host. Libraries like **MSCCLPP** and **IBGDA** exist precisely for this, but they expose very low-level primitives that are far outside the typical training distribution of a coding model.

**GPU time is expensive, so kernels must be customized per architecture.** The right all-to-all implementation for an H200 over NVLink looks nothing like the right one for an AMD MI325x over InfiniBand. Squeezing idle GPU cycles requires writing kernels from scratch, tailored to the memory hierarchy, warp scheduler, and NIC capabilities of the specific hardware. This is exactly the kind of long-tail, architecture-specific code that LLMs have the least training signal on.

**Communication is becoming irregular and fine-grained, especially in MoE.** Expert Parallelism in mixture-of-experts models produces dynamic, non-uniform all-to-all patterns that NCCL's bulk-synchronous collective model handles poorly. Efficient MoE dispatch requires custom GPU kernels that interleave routing decisions, RDMA writes, and local compute — compute-communication fusion at the tile level. No off-the-shelf library does this well, and LLMs have almost no exposure to the relevant code patterns.

Despite all this, multi-device coding has been **largely overlooked** in LLM coding benchmarks. HumanEval, MBPP, LiveCodeBench — these measure single-device reasoning. There is no established benchmark for whether a model can write a correct, performant mscclpp kernel, an RDMA write loop with proper memory ordering, or a fused AllGather+GEMM across NVLink and InfiniBand. We aim to take the first step in filling that gap.

---

## Benchmark Structure

Our benchmark covers four categories of multi-device communication tasks, drawn from real industry use cases in the UCCL project:

| Category | Examples |
|---|---|
| **Inter-node RDMA basics** | libibverbs QP setup, RDMA write, write-with-IMM, memory registration |
| **Intra-node NVLink basics** | TMA-based transfers, DMA engines, register-level copy |
| **GPU-initiated communication** | MSCCLPP MemoryChannel/ProxyChannel, NVSHMEM |
| **Compute-communication fusion** | AllGather+GEMM, MoE dispatch+GEMM, QKNorm+AllReduce |

Each task asks a model to implement a specific primitive using a specified library and target hardware. The prompt includes the communication pattern, library API, and cluster topology (number of ranks, message sizes, NIC type).

**Compilation feedback loop.** After each generation, we compile the kernel and feed the full compiler output back as context for the next round. We allow up to **five rounds**. A task is marked a compilation failure if no round produces a clean build. For all kernels that do compile, we measure achieved bandwidth or latency against a hand-crafted reference implementation.

Models evaluated: **DeepSeek V4 Pro**, **GPT-5.2**, **Gemini-3-Pro**, **Claude Sonnet-4.5**, **Grok-3**

Hardware: Nvidia B300x8, Nvidia GH200x2 (two nodes), AMD MI325x x8

---

## Case Studies

### NCCL

*[TBD]*

### RDMA Verbs

*[TBD]*

### ThunderKitten

*[TBD]*

### MSCCLPP All-to-All

<a href="https://github.com/microsoft/mscclpp">mscclpp</a> is Microsoft's low-level GPU communication library designed for fine-grained control over RDMA and NVLink transfers. mscclpp has very elegant and efficient abstraction like memorychannels and portchannels.

First, here is a brief overview of what **all-to-all** fulfills: every rank simultaneously sends a distinct data chunk to every other rank. This is among the most demanding collectives, requiring coordination of N×(N-1) concurrent transfers, each with its own buffer offset, channel handle, and synchronization barrier.

#### DeepSeek V4 Pro: five rounds, zero compilations

Over five rounds of prompting, with full compiler feedback provided after each attempt, DeepSeek V4 Pro failed to produce a single compilable kernel. The generated code repeatedly hallucinated APIs, relied on nonexistent abstractions, and assumed outdated MSCCL++ interfaces that no longer matched the installed runtime. Even after iterative correction attempts, the model never converged to a buildable implementation.

#### GPT-5.5 triage, Human in the loop

To recover a working baseline, we switched to GPT-5.5 through Codex. Getting from a broken build to a compiling one took under three minutes; getting all the way to correct took another seventeen. A human stayed in the loop mainly to relay whether builds and correctness checks passed — the model handled the rest.

The bulk of the work was untangling how thoroughly the original generated code had hallucinated the MSCCL++ API. A header hallucinated by DeepSeek, broken bootstrap initialization, and a fundamental misunderstanding of ```mscclpp::Communicator``` kicked off a cascade of bad calls: wrong memory registration paths, invented pack/unpack routines, fabricated semaphore tables, and channel construction methods that simply don't exist. Each fix exposed the next.

Compilation passing turned out to be only the beginning. The kernel ran but produced wrong results, requiring six more patches. The deepest ones stemmed from the same conceptual gap: the model knew all-to-all requires synchronization, but didn't know where MSCCL++ puts it. Others were more mundane: every thread copying the full local slice instead of a strided subset, and a verification harness that silently rounded BF16 fill values to clean multiples, making it blind to an entire class of data corruption until the pattern was changed.

Taken together, these patches reveal that the LLM failed at four distinct levels simultaneously: mscclpp API semantics, collective communication logic, basic GPU parallelism, and numerical precision of the test harness. Compiler feedback alone, however many rounds, cannot surface any of them.

#### Performance

![alt text](../graphs/all2all_ref_vs_generated_latency_throughput.png)

The DeepSeek-generated kernel (after 5 rounds of compiler-feedback prompting, 2 rounds of GPT-5.5 triage and human in the loop) passes correctness but is catastrophically slower than the reference across all message sizes. 

At 1 MiB, the reference achieves 22.5 GB/s while the LLM kernel achieves 0.067 GB/s. This is a 336× gap. At 1536 MiB the gap narrows to 15× (680 GB/s vs. 46 GB/s), but never closes. 

#### Why does this happen?

The generated kernel is conceptually clean: split each peer's slice into channel-sized tasks, let blocks walk the (peer, channel) assignments, and use MemoryChannel::put() to write directly into the remote rank's output buffer. The problem isn't the data movement — it's the synchronization. The kernel bolts it on as a global phase boundary: bulk transfer everything, then grid.sync(), then one single thread serially signals and waits for every peer, then grid.sync() again. That design incurs a fixed ~15ms overhead on every launch regardless of message size, and even at large messages where that cost amortizes, throughput plateaus because the serialized fence can't keep up with what the hardware can actually move.


The reference kernel is organized around the lower-level motif MSCCL++ actually expects: not "copy everything, then synchronize," but "pipeline chunk movement and synchronization together at warp granularity." It splits each block into 16 put-warps and 16 copy-warps, chooses a 256 KiB pipeline unit, and rotates through peers in steps — put-warps copy a chunk and immediately signal(), while copy-warps wait() for the corresponding incoming chunk and move it onward. Named barriers (bar.sync IDs 14 and 15) synchronize only the relevant warp groups rather than stalling the whole grid.
The generated kernel understood that all-to-all requires synchronization. It didn't know where MSCCL++ expects that synchronization to live — not as a global fence, but woven into the data movement itself, at warp granularity, chunk by chunk.

```
// generated: global phase boundary
grid.sync();  // ~15ms fixed tax, every launch

// reference: warp-local, per-chunk
bar.sync 15, 512;  // only the put-warps, per chunk
```

---

## Next Steps: Post-Training with RL
*[TBD]*

---

## Conclusion
*[TBD]*


---

## Acknowledgements

We thank Mibura and AMD for sponsoring the testbed for this benchmark.

## References

1. Chao Jin et al. *MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production*. EuroSys, 2026.
2. Shulai Zhang et al. *Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts*. MLSys, 2025.
3. Changho Hwang et al. *MSCCL++: Rethinking GPU Communication Abstractions for AI Inference*. ASPLOS, 2026