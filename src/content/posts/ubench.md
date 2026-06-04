---
title: "CommBench: Can LLMs Write Correct and Efficient GPU Communication code?"
slug: llm-gpu-comm-kernels
description: "CommBench evaluates how effectively frontier LLMs generate multi-device GPU communication code, covering diverse communication functionalities and compute–communication fusion kernels."
category:
  - One
tags:
  - LLMs
  - Benchmark
  - Code Generation
  - GPU Communication
  - NCCL
  - RDMA
  - CUDA
  - NCCL
  - MSCCLPP
pubDate: 2026-05-21
cover: /ubench/commbench-logo.png
coverAlt: CommBench logo
author: Shuang Ma, Yuyi Li, xx, xx, Yang Zhou, and the UCCL Team.
---

<p>
<strong>By: Shuang Ma, Yuyi Li, xx, xx, Yang Zhou, and the UCCL Team.
<br>
Date: May 21, 2026
</strong>
</p>

<div class="tldr">
<p>
Today's frontier LLMs write excellent single-device code yet consistently fail on multi-device GPU communication, precisely the code that bottlenecks large-scale LLM training and inference. We present CommBench, a benchmark spanning <strong>point-to-point</strong>, <strong>collective</strong>, <strong>expert-parallel</strong>, <strong>compute and communication fusion</strong>, and <strong>utilities</strong> (building blocks for GPU communication interfaces, such as a GPU–CPU FIFO queue), drawn from hand-written implementations and production codebases including Mscclpp, NCCL, NVSHMEM, DeepEP, ThunderKittens, vLLM, and SGLang. We evaluate leading closed and open models under a cheat-resistant harness and present case studies of where and why they succeed or break down. As future work, we plan to post-train LLMs on these datasets to close this gap.
</p>
<p>
Open-sourced at <a href="https://github.com/uccl-project/llm-for-gpu-comm/tree/main">uccl-project/llm-for-gpu-comm</a>.
</p>
</div>

## Why Multi-Device Coding Matters — and Why LLMs Are Bad at It

Communication and compute-communication fusion sit at the critical path of every serious LLM workload today. In production training, communication can consume **43.6% of the forward pass** [1]; in MoE inference with wide expert parallelism, inter-device communication accounts for **up to 47% of total execution time** [2]. Getting this code **right** and **fast** is not a nice-to-have.

**The demand for customized GPU communication and compute-communication fusion is rapidly growing.** Established libraries like NCCL offer comprehensive interfaces, but optimize for generality over frontier performance. As a result, companies often maintain in-house GPU communication and computation stacks for tighter control and optimization. GPU communication also remains a rapidly evolving area: new hardware and new LLM architectures continuously introduce new requirements for higher performance and specialized workloads, while communication abstractions are still evolving:

- **Modern GPUs are extremely powerful and expensive**, motivating highly customized kernels and tighter compute–communication fusion to maximize hardware utilization across architectures such as Hopper, Blackwell, and AMD GPUs. As GPUs become faster, communication increasingly needs to be initiated directly from GPUs instead of relying on CPU-mediated execution paths used in traditional libraries such as NCCL.
- **New LLM architectures, such as MoE expert parallelism**, introduce increasingly irregular and fine-grained communication patterns that are not well supported by existing collective libraries.

Multi-device coding is inherently harder than single-device coding, for three reasons:

- **It demands niche expertise**, requiring deep knowledge of both GPU kernels and networking.
- **It requires coordinating many devices over fail-prone interconnects**, which is intrinsically difficult.
- **It lacks data**, as practical, faithful datasets for multi-GPU coding are largely missing.

Despite all this, multi-device coding has been **largely overlooked** in LLM coding benchmarks. HumanEval, MBPP, LiveCodeBench — these measure single-device reasoning. No benchmark tests whether a model can write correct GPU communication, or fused communication-plus-compute, functionality (e.g., components like Mscclpp channels, a collective interface, or a fused AllGather+GEMM across NVLink and InfiniBand).

---

## Benchmark and Framework Structure

### Benchmark Structure

The dataset is organized as a list of independently runnable examples. Some implement complete, ready-to-use functionality (e.g., P2P/collective interfaces, or MoE expert-parallel dispatch and combine); others are reusable communication building blocks (e.g., Mscclpp channels). Drawing on our hands-on experience from the UCCL project, we manually assign each example one of three difficulty levels: **Easy / Medium / Hard**.

Some examples we hand-wrote on top of base libraries; others are extracted from production-grade communication and LLM-serving frameworks. **By function**, they fall into:

- **P2P** — point-to-point transfer between a pair of devices.
- **Collective** — group operations across all ranks (AllReduce, AllGather, All-to-All, …).
- **EP** — dynamic, non-uniform dispatch/combine traffic for MoE models.
- **Fusion** — kernels that interleave communication with compute (e.g., AllGather+GEMM).
- **Utilities** — supporting components such as connection setup, buffer registration, and topology queries.

**By source**, they span cuda-runtime, ibverbs, Mscclpp, nccl, nvshmem, deepep, nccl-device-api, thunderkittens, vllm, and sglang.

### Framework Structure

We built a framework that automatically evaluates different models on the dataset and supports multi-round refinement of generated code. Careful design of each example and its execution environment enforces strict checking and prevents the model from cheating.

**Example structure.** Each example has three parts:

- **Reference solution** — high-quality, hand-written code organized in an object-oriented style (Python / CUDA / HIP / C++). It ships with a test harness that uses randomized inputs, so a model cannot hard-code expected outputs.
- **Empty solution** — the reference with its core implementation stripped out and marked `// TODO`, but with the functional interface preserved. This file becomes part of the prompt: the model must honor the interface semantics, which also keeps its output directly testable. We verify that only the regions meant to be edited were changed, guarding against cheating.
- **Build-and-run script** — independently compiles and runs an example (reference or generated) and is hidden from the model. By controlling the compile command, we strictly ensure the generated code uses only the intended libraries.

**Multi-round refinement.** When `max-round > 1`, if the generated code fails to run or underperforms the reference, we feed the compile/run output back into the prompt and ask the model to iterate — repeating until performance improves or the round limit is reached.

---

## Leaderboard

> Sorted by **Pass×GM** ⭐ — pass rate scaled by geometric-mean code quality on passing examples.
> Bars `▓░` are scaled to the column maximum.

| Rank | Model | Price | Pass×GM | Pass Rate | PASS+Good | GM‑Speedup |
|:----:|:------|:-----:|:-------:|:---------:|:---------:|:----------:|
| 🥇 | **gpt-5.5** |  | 🟢 **0.539** `▓▓▓▓▓▓▓▓` | 🟢 59.4% `▓▓▓▓▓▓▓▓` | 🟢 **32.7%** `▓▓▓▓▓▓▓▓` | 🟡 0.908 `▓▓▓░░░░░` |
| 🥈 | **gemini-3.1-pro-preview** |  | 🟡 0.305 `▓▓▓▓▓░░░` | 🟡 36.6% `▓▓▓▓▓░░░` | 🟢 **25.7%** `▓▓▓▓▓▓░░` | 🔴 0.832 `░░░░░░░░` |
| 🥉 | **claude-opus-4-7** |  | 🟡 0.282 `▓▓▓▓▓░░░` | 🟡 33.7% `▓▓▓▓▓░░░` | 🟡 **20.8%** `▓▓▓▓▓░░░` | 🔴 0.836 `░░░░░░░░` |
| 4️⃣ | **glm-5.1** |  | 🟡 0.281 `▓▓▓▓░░░░` | 🟡 29.7% `▓▓▓▓░░░░` | 🟡 **17.8%** `▓▓▓▓░░░░` | 🟢 0.947 `▓▓▓▓▓░░░` |
| 5️⃣ | **kimi-k2.6** |  | 🟡 0.275 `▓▓▓▓░░░░` | 🟡 30.7% `▓▓▓▓░░░░` | 🟡 **18.8%** `▓▓▓▓▓░░░` | 🟡 0.895 `▓▓▓░░░░░` |
| 6️⃣ | **qwen3.7-max** |  | 🟡 0.269 `▓▓▓▓░░░░` | 🟡 26.7% `▓▓▓▓░░░░` | 🟡 **15.8%** `▓▓▓▓░░░░` | 🟢 1.008 `▓▓▓▓▓▓▓▓` |
| 7️⃣ | **deepseek-v4-pro** |  | 🔴 0.197 `▓▓▓░░░░░` | 🔴 19.8% `▓▓▓░░░░░` | 🔴 **12.9%** `▓▓▓░░░░░` | 🟢 0.995 `▓▓▓▓▓▓▓░` |

**Color:** 🟢 top tier &nbsp;·&nbsp; 🟡 mid tier &nbsp;·&nbsp; 🔴 bottom tier &nbsp;&nbsp;|&nbsp;&nbsp; GM‑Speedup bar scaled to [0.832, 1.008] range (not from zero).

### Metric Definitions

| Metric | Formula | What it measures |
|:-------|:--------|:----------------|
| **Pass×GM** ⭐ | Pass Rate × GM‑Speedup | Pass rate scaled by geometric-mean code quality on passing examples. Combines coverage and quality: a model that rarely passes but generates very fast code scores the same as one that always passes at reference speed. Primary ranking metric. |
| **Pass Rate** | PASS / Total | Fraction of examples where code compiled, ran, and produced correct results. |
| **PASS+Good** | (on\_compare + better) / Total | Fraction of all examples with correct **and** performant code (within −5% of reference). |
| **GM‑Speedup** | `GM` over scored PASS of `scoreᵢ` | Geometric mean of per-example speedup scores over passing examples. The per-example score is `scoreᵢ = GMₛ( gen(i,s) / ref(i,s) )`, where `s` iterates over measured data sizes and `gen(i,s)`, `ref(i,s)` are the performance metrics of the generated and reference code on example `i` at data size `s`, expressed so that **higher is better** (lower-is-better metrics such as latency are inverted to `1/latency` first). The choice of primary metric per example is determined by human review. Taking the GM across sizes (rather than averaging absolute throughput first) ensures each data point contributes equally regardless of absolute throughput magnitude; without it, large sizes (e.g. 128 MB at ~100 GB/s) would dominate small ones (e.g. 256 KB at ~5 GB/s). |
| **Price** | — | Average cost per example (USD). |

**Performance verdict thresholds** (4-tier):
`better` ≥ +20% &nbsp;·&nbsp; `on_compare` −5% to +20% &nbsp;·&nbsp; `degraded` −40% to −5% &nbsp;·&nbsp; `severely_degraded` < −40%

---

## Analysis: Top vs. Bottom Model

We select the highest- and lowest-scoring models on CommBench, gpt-5.5 (Pass×GM = 0.539) and deepseek-v4-pro (Pass×GM = 0.197), for a detailed breakdown across difficulty levels, task types, library coverage, and code performance.

### Difficulty Breakdown

<table>
<tr>
<td align="center" width="50%"><b>GPT-5.5</b></td>
<td align="center" width="50%"><b>DeepSeek-V4-Pro</b></td>
</tr>
<tr>
<td><img src="/ubench/fig1_level_breakdown_gpt-5.5.png" width="100%"></td>
<td><img src="/ubench/fig1_level_breakdown_deepseek-v4-pro.png" width="100%"></td>
</tr>
</table>

Human-defined difficulty does not always align with model difficulty. deepseek's weak Easy performance (14%) is largely attributable to gaps in library-specific API knowledge — for libraries such as mscclpp and ThunderKittens, it lacks the interface semantics needed to call APIs correctly, causing failures on examples that only require invoking a small number of functions to implement straightforward functionality. On the Hard end, many examples ask the model to implement communication primitives from near-scratch (e.g. AllReduce over NVLink or RDMA) which exposes deepseek's limited capability on complex GPU communication-compute tasks.

### Performance Quality Among PASS Examples

<table>
<tr>
<td align="center" width="50%"><b>GPT-5.5</b></td>
<td align="center" width="50%"><b>DeepSeek-V4-Pro</b></td>
</tr>
<tr>
<td><img src="/ubench/fig2_pass_performance_gpt-5.5.png" width="100%"></td>
<td><img src="/ubench/fig2_pass_performance_deepseek-v4-pro.png" width="100%"></td>
</tr>
</table>

gpt-5.5 has the widest quality spread: 8 severely degraded cases (13% of PASS), all concentrated in Hard examples. This is partly because examples that gpt-5.5 manages to compile and run (but with degraded performance) would simply fail to compile or execute under deepseek, and therefore never appear in deepseek's performance distribution at all.

### Tag and Library Coverage

<table>
<tr>
<td align="center" width="50%"><b>GPT-5.5</b></td>
<td align="center" width="50%"><b>DeepSeek-V4-Pro</b></td>
</tr>
<tr>
<td><img src="/ubench/fig3_radar_tag_library_gpt-5.5.png" width="100%"></td>
<td><img src="/ubench/fig3_radar_tag_library_deepseek-v4-pro.png" width="100%"></td>
</tr>
</table>

gpt-5.5 dominates on Collective (72% vs 22%) and is the **only** model with meaningful coverage of specialized libraries — mscclpp (33%), nccl-device-api (40%), and thunderkittens (54%). deepseek is competitive on NCCL (100%) and P2P tags, but scores 0% across all three specialized libraries. For widely adopted libraries such as vllm, NCCL, and nvshmem, both models perform well.

### DeepSeek with max_rounds = 5

Due to budget constraints, gpt-5.5 was evaluated with a single generation round. Despite deepseek's weaker first-round performance, its API cost is only xx of gpt-5.5's, which makes multi-round self-correction economically viable. Allowing deepseek up to 5 self-correction rounds substantially changes its profile.

#### Cumulative PASS by Round

```text
Round 1:  16 PASS  (15.8%)
Round 2:  28 PASS  (27.7%)   +12  ← largest single gain
Round 3:  32 PASS  (31.7%)   +4
Round 4:  34 PASS  (33.7%)   +2
Round 5:  41 PASS  (40.6%)   +7
```

#### Difficulty Breakdown: max=1 vs max=5

<table>
<tr>
<td align="center" width="50%"><b>DeepSeek max=1</b></td>
<td align="center" width="50%"><b>DeepSeek max=5</b></td>
</tr>
<tr>
<td align="center"><img src="/ubench/fig1_level_breakdown_deepseek-v4-pro.png" style="height: 240px; width: auto; max-width: 100%;"></td>
<td align="center"><img src="/ubench/fig1_level_breakdown_deepseek_v4_pro_max5.png" style="height: 240px; width: auto; max-width: 100%;"></td>
</tr>
</table>

#### Performance Quality: max=1 vs max=5

<table>
<tr>
<td align="center" width="50%"><b>DeepSeek max=1</b></td>
<td align="center" width="50%"><b>DeepSeek max=5</b></td>
</tr>
<tr>
<td><img src="/ubench/fig2_pass_performance_deepseek-v4-pro.png" width="100%"></td>
<td><img src="/ubench/fig2_pass_performance_deepseek_v4_pro_max5.png" width="100%"></td>
</tr>
</table>

#### Tag and Library Coverage: max=1 vs max=5

<table>
<tr>
<td align="center" width="50%"><b>DeepSeek max=1</b></td>
<td align="center" width="50%"><b>DeepSeek max=5</b></td>
</tr>
<tr>
<td><img src="/ubench/fig3_radar_tag_library_deepseek-v4-pro.png" width="100%"></td>
<td><img src="/ubench/fig3_radar_tag_library_deepseek_v4_pro_max5.png" width="100%"></td>
</tr>
</table>

Multi-round self-correction doubles deepseek's overall pass rate and unlocks Medium-difficulty commodity-library tasks. It does not unlock Hard examples or specialized libraries — those require domain knowledge the model does not have. The practical implication: deepseek with retries is a reasonable choice when tasks are restricted to commodity libraries (NCCL, vllm, cuda-runtime, nvshmem) and a retry budget is acceptable.

---

## Case Studies

### NCCL

*[TBD]*

### RDMA Verbs

*[TBD]*

### ThunderKitten

*[TBD]*

### MSCCLPP All-to-All

<a href="https://github.com/microsoft/mscclpp">Mscclpp</a> is Microsoft's low-level GPU communication library designed for fine-grained control over RDMA and NVLink transfers. Mscclpp has very elegant and efficient abstraction like memorychannels and portchannels.

First, here is a brief overview of what **all-to-all** fulfills: every rank simultaneously sends a distinct data chunk to every other rank. This is among the most demanding collectives, requiring coordination of N×(N-1) concurrent transfers, each with its own buffer offset, channel handle, and synchronization barrier.

#### DeepSeek V4 Pro: five rounds, zero compilations

Over five rounds of prompting, with full compiler feedback provided after each attempt, DeepSeek V4 Pro failed to produce a single compilable kernel. The generated code repeatedly hallucinated APIs, relied on nonexistent abstractions, and assumed outdated MSCCL++ interfaces that no longer matched the installed runtime. Even after iterative correction attempts, the model never converged to a buildable implementation.

#### GPT-5.5 triage, Human in the loop

To recover a working baseline, we switched to GPT-5.5 through Codex. Getting from a broken build to a compiling one took under three minutes; getting all the way to correct took another seventeen. A human stayed in the loop mainly to relay whether builds and correctness checks passed — the model handled the rest.

The bulk of the work was untangling how thoroughly the original generated code had hallucinated the MSCCL++ API. A header hallucinated by DeepSeek, broken bootstrap initialization, and a fundamental misunderstanding of ```mscclpp::Communicator``` kicked off a cascade of bad calls: wrong memory registration paths, invented pack/unpack routines, fabricated semaphore tables, and channel construction methods that simply don't exist. Each fix exposed the next.

Compilation passing turned out to be only the beginning. The kernel ran but produced wrong results, requiring six more patches. The deepest ones stemmed from the same conceptual gap: the model knew all-to-all requires synchronization, but didn't know where MSCCL++ puts it. Others were more mundane: every thread copying the full local slice instead of a strided subset, and a verification harness that silently rounded BF16 fill values to clean multiples, making it blind to an entire class of data corruption until the pattern was changed.

Taken together, these patches reveal that the LLM failed at four distinct levels simultaneously: Mscclpp API semantics, collective communication logic, basic GPU parallelism, and numerical precision of the test harness. Compiler feedback alone, however many rounds, cannot surface any of them.

#### Performance

<p align="center">
  <img src="/ubench/all2all_ref_vs_generated_latency_throughput.png" alt="All2All" width="600"/>
  <em>Figure 1: xx.</em>
</p>


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