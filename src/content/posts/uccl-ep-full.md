---
title: "UCCL-EP: Portable Expert-Parallel Communication — Full Results"
slug: uccl-ep-full
description: "Full evaluation of UCCL-EP across NVIDIA and AMD GPUs, AWS EFA, InfiniBand, and Broadcom NICs — with application-level results on SGLang inference and Megatron-LM training."
category:
  - One
tags:
  - MoE
  - DeepEP
  - RDMA
  - Expert Parallelism
  - AMD
  - EFA
pubDate: 2026-03-28
cover: https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/UEP-intro-figure.jpg
coverAlt: UCCL-EP
author: UCCL Team
---

<p>
<strong>By: <a href="https://maoziming.github.io/">Ziming Mao</a> (UC Berkeley), Chon Lam Lao (UC Berkeley), <a href="https://yangzhou1997.github.io/">Yang Zhou</a> (UC Davis), <a href="github.com/CalebZ9909" class="no-github-icon">Yihan Zhang</a> (UC Davis), <a href="https://github.com/HermesCui" class="no-github-icon">Chihan Cui</a> (UW-Madison), <a href="https://zhongjiechen.github.io/" class="no-github-icon">Zhongjie Chen</a> (Tsinghua), <a href="https://xuzhiying9510.github.io/">Zhiying Xu</a> (AWS), and other UCCL-EP contributors
<br>
Date: Dec 20, 2025
</strong>
</p>

<div class="tldr">
<p>
We present the full evaluation of <strong>UCCL-EP</strong>, a portable expert-parallel communication system that achieves DeepEP-level performance across heterogeneous GPU and NIC hardware. UCCL-EP outperforms the best existing EP solution (PPLX) by up to <strong>2.3x</strong> for dispatch and combine on AWS EFA. On the NVIDIA-only InfiniBand platform, UCCL-EP achieves performance within 5% of the original DeepEP. For end-to-end applications, UCCL-EP speeds up SGLang inference throughput by up to <strong>40%</strong> over NCCL and improves Megatron-LM training throughput by up to <strong>45%</strong> over RCCL on AMD GPUs.
</p>
<p>
Paper: <a href="https://arxiv.org/pdf/2512.19849">arxiv.org/pdf/2512.19849</a> | Code: <a href="https://github.com/uccl-project/uccl/tree/main/ep">uccl-project/uccl/ep</a>
</p>
</div>

## Recap: Why UCCL-EP?

In our [previous blog post](/uccl-ep), we introduced UCCL-EP and the key challenge it addresses: state-of-the-art EP communication systems like DeepEP achieve high performance through GPU-initiated token-level RDMA, but are tightly coupled to the NVIDIA GPU + NVIDIA NIC ecosystem (via IBGDA). This tight coupling prevents running on public clouds like AWS (which use EFA NICs), on AMD GPUs, or on alternative NIC vendors like Broadcom.

UCCL-EP solves this with a clean separation of concerns:

- **GPUs** retain fine-grained token-level communication initiation for maximal overlap with computation.
- **CPUs** handle the control-intensive networking aspects — queue management, flow control, ordering enforcement — through a lightweight, multi-threaded proxy.

This architecture reduces the porting effort from O(m x n) (for m GPU vendors and n NIC vendors) down to O(m), as the CPU proxy speaks the portable `libibverbs` API that all RDMA NICs support.

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/uep_architecture.png" alt="UCCL-EP Architecture" width="600"/>
  <em>UCCL-EP architecture. GPUs delegate token-routing commands to multi-threaded CPU proxies via lock-free FIFO channels. CPU proxies issue GPUDirect RDMA on behalf of GPUs, while managing ordering, flow control, and completion handling.</em>
</p>

---

## Key Techniques

### CPU Proxy with Lock-Free FIFO Channels

UCCL-EP uses a 128-bit `TransferCmd` descriptor that GPU threads enqueue into shared, lock-free FIFO channels. CPU proxy threads dequeue these commands and issue the corresponding RDMA operations. The FIFO design caches the tail index on GPU memory, minimizing PCIe traversals. Each GPU uses multiple FIFO channels, each mapped to a dedicated RDMA Queue Pair (QP), enabling over **50 million RDMA operations per second** per GPU.

### Immediate-Data-Based Ordering for Heterogeneous NICs

Different NICs provide different delivery guarantees. For example, AWS EFA uses the Scalable Reliable Datagram (SRD) protocol with multi-pathing, which does not guarantee in-order delivery within a single QP. UCCL-EP embeds per-channel sequence numbers into RDMA immediate data, allowing the receiver-side CPU proxy to reorder out-of-sequence control messages before committing them to GPU memory. This approach also enables **software-level atomics** — piggybacking completion notifications on regular RDMA writes — which is both more portable (no hardware atomic support required) and more efficient (one RDMA operation instead of two for write + atomic).

---

## Evaluation Testbeds

We evaluate UCCL-EP on a diverse set of platforms spanning NVIDIA and AMD GPUs with EFA, InfiniBand, and Broadcom NICs. All testbeds are rented from public cloud providers.

| Name | Servers | GPU | NIC | CPU | Cloud |
|:----:|:-------:|:---:|:---:|:---:|:-----:|
| NV_EFAv3 | 4 | NVIDIA H200 x8 | AWS EFAv3 200G x16 | 192 cores | AWS (p5en) |
| NV_EFAv4 | 4 | NVIDIA B200 x8 | AWS EFAv4 400G x8 | 192 cores | AWS (p6-b200) |
| NV_IB | 4 | NVIDIA H100 x8 | ConnectX-7 400G x8 | 128 cores | Nebius |
| NV_GH200 | 2 | NVIDIA GH200 x1 | ConnectX-7 200G x1 | 72 cores | Lambda |
| AMD_IB | 4-16 | AMD MI300X x8 | ConnectX-7 400G x8 | 128 cores | OCI |
| AMD_BRC | 4 | AMD MI300X x8 | Broadcom Thor-2 400G x8 | 128 cores | Vultr |

**Baselines:** NCCL/RCCL, DeepEP (NVIDIA-only), Perplexity Kernels (PPLX), and CPU-assisted IBGDA. UCCL-EP uses 4 CPU proxy threads per GPU.

---

## Microbenchmark Results

### On AWS EFA (NVIDIA GPUs)

#### H200 + EFA (p5en)

On p5en instances (8x H200, 16x 200 Gb/s EFA), we measure EP32 dispatch and combine latency while varying the number of tokens. UCCL-EP uses the minimum of high-throughput (HT) and low-latency (LL) mode latency, while PPLX operates in a single mode.

At small batch sizes (128 tokens), PPLX achieves lower latency because UCCL-EP (extending DeepEP) issues messages at 7 KB token granularity, whereas PPLX packs tokens into larger messages. However, as the token count increases, **UCCL-EP quickly overtakes PPLX** — delivering **2.3x lower dispatch latency** and **1.1–1.5x lower combine latency** for medium and large batches.

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/p5en_dispatch_latency_vs_tokens_pplx_uccl.jpg" alt="p5en dispatch latency vs tokens" width="400"/>
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/p5en_combine_latency_vs_tokens_pplx_uccl.jpg" alt="p5en combine latency vs tokens" width="400"/>
  <em>EP32 dispatch (left) and combine (right) latency vs. number of tokens on p5en (H200 + EFA 400 Gbps).</em>
</p>

##### Normal Mode Bandwidth (p5en)

Following the DeepSeek-V3 pretraining configuration (4096 tokens, 7168 hidden, top-4 groups, top-8 experts, FP8 dispatch / BF16 combine):

| Type | Dispatch FP8 #EP | Bottleneck BW & Latency | Combine BF16 #EP | Bottleneck BW & Latency |
|:---:|:---:|:---:|:---:|:---:|
| Intranode | 8 | 320 GB/s (NVLink), 500 us | 8 | 319 GB/s (NVLink), 973 us |
| Internode | 16 | 50 GB/s (RDMA), 1196 us | 16 | 18 GB/s (RDMA), 6379 us |
| Internode | 24 | 53 GB/s (RDMA), 1633 us | 24 | 26 GB/s (RDMA), 6365 us |
| Internode | 32 | 54 GB/s (RDMA), 2022 us | 32 | 43 GB/s (RDMA), 4899 us |

##### Low-Latency Mode (p5en)

Following a DeepSeek-V3 inference setting (128 tokens, 7168 hidden, top-8 experts, FP8 dispatch / BF16 combine):

| Dispatch #EP | Latency | RDMA BW | Combine #EP | Latency | RDMA BW |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 16 | 226 us | 36 GB/s | 16 | 293 us | 48 GB/s |
| 24 | 386 us | 20 GB/s | 24 | 580 us | 26 GB/s |
| 32 | 465 us | 16 GB/s | 32 | 694 us | 25 GB/s |

#### B200 + EFA (p6-b200)

On p6-b200 instances (8x B200, 8x 400 Gb/s EFA), we see similar trends with UCCL-EP outperforming PPLX at larger token counts:

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/p6_dispatch_ll_ht.png" alt="p6 dispatch" width="400"/>
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/p6_combine_ll_ht.png" alt="p6 combine" width="400"/>
  <em>EP32 dispatch (left) and combine (right) comparison on p6-b200 (B200 + EFA 400 Gbps).</em>
</p>

##### Normal Mode Bandwidth (p6-b200)

| Type | Dispatch FP8 #EP | Bottleneck BW & Latency | Combine BF16 #EP | Bottleneck BW & Latency |
|:---:|:---:|:---:|:---:|:---:|
| Intranode | 8 | 280 GB/s (NVLink), 571 us | 8 | 426 GB/s (NVLink), 727 us |
| Internode | 16 | 53 GB/s (RDMA), 1141 us | 16 | 60 GB/s (RDMA), 1965 us |
| Internode | 24 | 53 GB/s (RDMA), 1637 us | 24 | 59 GB/s (RDMA), 2887 us |
| Internode | 32 | 53 GB/s (RDMA), 2072 us | 32 | 57 GB/s (RDMA), 3724 us |

##### Low-Latency Mode (p6-b200)

| Dispatch #EP | Latency | RDMA BW | Combine #EP | Latency | RDMA BW |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 16 | 228 us | 33 GB/s | 16 | 318 us | 46 GB/s |
| 24 | 448 us | 17 GB/s | 24 | 566 us | 26 GB/s |
| 32 | 406 us | 19 GB/s | 32 | 617 us | 24 GB/s |

---

### On InfiniBand (NVIDIA CX7)

On the Nebius testbed (H100 + CX7 InfiniBand), we compare UCCL-EP against both the original DeepEP and PPLX at EP32.

In **LL mode**, UCCL-EP incurs slightly higher latency than DeepEP and PPLX due to the CPU proxy overhead on small messages. However, in **HT mode**, UCCL-EP achieves latency **within 5% of DeepEP** for dispatch while outperforming PPLX by **2.1x** (dispatch) and **1.6x** (combine). This shows that UCCL-EP preserves DeepEP-level performance on throughput-oriented workloads even without IBGDA.

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/nebius_dispatch_ll_ht.png" alt="Nebius dispatch" width="400"/>
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/nebius_combine_ll_ht.png" alt="Nebius combine" width="400"/>
  <em>EP32 dispatch (left) and combine (right) comparison on H100 + CX7 InfiniBand. UCCL-EP matches DeepEP in HT mode and significantly outperforms PPLX.</em>
</p>

---

### On AMD GPUs

UCCL-EP is the first system to enable GPU-initiated token-level EP communication on AMD GPUs. We evaluate on MI300X with both CX7 InfiniBand (OCI) and Broadcom Thor-2 NICs (Vultr).

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/amd_dispatch_ll_ht.png" alt="AMD dispatch" width="400"/>
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/amd_combine_ll_ht.png" alt="AMD combine" width="400"/>
  <em>EP32 dispatch (left) and combine (right) on AMD MI300X with CX7 IB and Broadcom Thor-2 NICs.</em>
</p>

UCCL-EP achieves similar performance across both NIC vendors, demonstrating true portability. On MI300X + CX7 IB, UCCL-EP achieves comparable performance to DeepEP on the NVIDIA-only platform.

#### Normal Mode Bandwidth on AMD MI300X + CX7 IB

| Type | FP8 Dispatch #EP | BW | BF16 Dispatch #EP | BW | Combine #EP | BW |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Intranode | 8 | 260 GB/s (xGMI) | 8 | 295 GB/s (xGMI) | 8 | 304 GB/s (xGMI) |
| Internode | 16 | 74 GB/s (RDMA) | 16 | 82 GB/s (RDMA) | 16 | 78 GB/s (RDMA) |
| Internode | 32 | 60 GB/s (RDMA) | 32 | 61 GB/s (RDMA) | 32 | 60 GB/s (RDMA) |
| Internode | 64 | 52 GB/s (RDMA) | 32 | 53 GB/s (RDMA) | 64 | 51 GB/s (RDMA) |

#### Normal Mode Bandwidth on AMD MI355X + Pollara AI NIC IB

| Type | FP8 Dispatch #EP | BW | BF16 Dispatch #EP | BW | Combine #EP | BW |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Intranode | 8 | 299 GB/s (xGMI) | 8 | 336 GB/s (xGMI) | 8 | 333 GB/s (xGMI) |
| Internode | 16 | 82 GB/s (RDMA) | 16 | 82 GB/s (RDMA) | 16 | 82 GB/s (RDMA) |
| Internode | 32 | 59 GB/s (RDMA) | 32 | 58 GB/s (RDMA) | 32 | 59 GB/s (RDMA) |
| Internode | 64 | 50 GB/s (RDMA) | 32 | 49 GB/s (RDMA) | 64 | 49 GB/s (RDMA) |

#### Normal Mode Bandwidth on AMD MI300X + Broadcom Thor-2

| Type | FP8 Dispatch #EP | BW | BF16 Dispatch #EP | BW | Combine #EP | BW |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Internode | 16 | 71 GB/s (RDMA) | 16 | 81 GB/s (RDMA) | 16 | 45 GB/s (RDMA) |
| Internode | 32 | 49 GB/s (RDMA) | 32 | 55 GB/s (RDMA) | 32 | 50 GB/s (RDMA) |

#### Low-Latency Mode on AMD MI300X + CX7 IB

| Dispatch #EP | Latency | RDMA BW | Combine #EP | Latency | RDMA BW |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | 65 us | 114 GB/s | 8 | 92 us | 157 GB/s |
| 16 | 136 us | 55 GB/s | 16 | 207 us | 70 GB/s |
| 32 | 224 us | 30 GB/s | 32 | 341 us | 42 GB/s |

---

## Application-Level Results

### SGLang Inference on AWS (NVIDIA + EFA)

We evaluate UCCL-EP in SGLang v0.5.3 on a prefill-heavy workload (input length 4096, output length 5) on p5en instances. We compare against NCCL, as DeepEP cannot run on EFA and PPLX had not been integrated into open-source inference engines at the time of evaluation.

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/sglang_deepseek_r1_throughput.png" alt="SGLang DeepSeek R1 throughput" width="400"/>
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/sglang_qwen3_throughput.png" alt="SGLang Qwen3 throughput" width="400"/>
  <em>SGLang throughput comparison using UCCL-EP vs. NCCL on p5en. Left: DeepSeek R1 (671B). Right: Qwen3 (235B).</em>
</p>

**DeepSeek R1 (671B):**
- EP=16: UCCL-EP reaches **46K tok/s** input throughput, about 5% higher than NCCL.
- EP=32: UCCL-EP improves to **74K tok/s**, a **1.6x** prefill speedup over its own EP=16 run.

**Qwen3 (235B):**
- EP=32: UCCL-EP reaches **62K tok/s** vs. 44K tok/s for NCCL — about **40% higher** throughput.

UCCL-EP enables larger EP configurations (EP=32) where NCCL either significantly underperforms or cannot run, as confirmed with SGLang maintainers. CPU utilization increases modestly from an average 8% to 22%.

---

### Megatron-LM Training on AMD (MI300X + Broadcom)

We evaluate end-to-end Megatron-LM training of DeepSeek-V3 (downscaled to 32 layers and 379B parameters) on 16 MI300X nodes using the AMD Primus/Megatron-LM framework.

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/amd_megatron_deepseekv3_tflops.png" alt="Megatron-LM TFLOPS" width="400"/>
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/amd_megatron_deepseekv3_tokens.png" alt="Megatron-LM tokens/s" width="400"/>
  <em>Megatron-LM training throughput (TFLOPS and tokens/s) on 16-node AMD MI300X + Broadcom Thor-2.</em>
</p>

Across all configurations, UCCL-EP matches or exceeds the TFLOPS by **7–36%** and throughput by **7–45%** compared to RCCL. These results show that UCCL-EP provides significant performance benefits for MoE training on AMD hardware.

---

## The Importance of Flow Control

A key advantage of UCCL-EP's CPU proxy architecture is the ability to implement **flow control and congestion management** — something that is extremely difficult with IBGDA, where GPU threads blindly issue one-sided RDMA operations without awareness of NIC queue utilization.

We observe that the number of outstanding RDMA requests can have a significant impact on various NICs, particularly affecting **tail latency**. This becomes increasingly critical as the number of destinations increases, where a single straggler can slow down the entire dispatch or combine operation.

UCCL-EP's CPU proxy supports request tracking and pacing:
- If outstanding requests grow too high, the proxy **temporarily buffers messages** at the sender to avoid incast at the receiver.
- The proxy can **shard outgoing requests across multiple NICs and QPs** to avoid congestion and adapt to NIC-specific characteristics.

This is particularly important for **Broadcom Thor-2** and **AMD Pollara AI** NICs, where strict flow control is necessary to avoid transport errors:
- For Broadcom Thor-2: `UCCL_IB_MAX_INFLIGHT_BYTES=1572864 UCCL_IB_MAX_INFLIGHT_NORMAL=1`
- For AMD Pollara AI NIC: `UCCL_IB_MAX_INFLIGHT_BYTES=2097152 UCCL_IB_MAX_INFLIGHT_NORMAL=1`

Without flow control, these NICs can encounter CQE error 12 (Transport Retry Counter Exceeded) under high load.

---

## Sensitivity Analysis

### FIFO Channel Performance

The FIFO queue latency is an order of magnitude smaller than the network latency, confirming that the CPU proxy is not a bottleneck. UCCL-EP FIFO queues scale to **8 Mops** (million operations per second), capable of handling modern MoE workloads.

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/fifo_latency.png" alt="FIFO latency" width="400"/>
  <em>FIFO queue latency vs. network latency with increasing 7 KB message throughput. Note the log scale on the y-axis.</em>
</p>

### Sensitivity to EP Degree and CPU Threads

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/ll_latency_vs_ep_sizes.png" alt="LL latency vs EP sizes" width="350"/>
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/ht_latency_vs_ep_sizes.png" alt="HT latency vs EP sizes" width="350"/>
  <em>Sensitivity to EP degree. UCCL-EP achieves better latency than PPLX in HT mode but higher latency in LL mode due to per-token message granularity on EFA.</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/ll_latency_vs_cpu_threads.png" alt="LL latency vs CPU threads" width="350"/>
  <img src="https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/uccl-ep-full/ht_latency_vs_cpu_threads.png" alt="HT latency vs CPU threads" width="350"/>
  <em>Sensitivity to number of CPU proxy threads. Performance significantly improves from 1 to 4 threads, with modest CPU utilization increase.</em>
</p>

---

## UCCL-EP LL: Towards More Efficient Small Messages

UCCL-EP's low-latency (LL) kernel, extending DeepEP, currently issues one 7 KB token per RDMA message. On EFA NICs, this is suboptimal because the current EFA firmware cannot process small tokens at a high enough rate (AWS has confirmed they are working on a firmware fix). PPLX, by contrast, packs tokens into larger messages, giving it an advantage at small batch sizes.

A natural optimization is to pack tokens in a **best-effort manner** before sending — combining the per-token flexibility of DeepEP with the batched efficiency of PPLX. We consider this optimization orthogonal to UCCL-EP's core contribution in portable EP communication architecture.

---

## A Note on Fair Comparison with PPLX Kernels

When comparing against PPLX kernels, we used the latest version ([pplx-garden](https://github.com/perplexityai/pplx-garden)) rather than the older [pplx-kernels](https://github.com/perplexityai/pplx-kernels), as the newer version has better performance. We use the same GPU resources (same number of SMs per GPU) to ensure fairness.

Key observations:
- PPLX has an inherent advantage at **small token counts** because it packs tokens into contiguous larger transfers, which benefits from EFA's higher throughput for larger messages.
- UCCL-EP (and DeepEP) has an advantage at **larger token counts** because GPU-initiated token-level communication enables better overlapping, deduplication, and hierarchical reduce.
- At EP=32, PPLX errored out for 4096 tokens on the older p5en testbed, while UCCL-EP continued to function correctly.

---

## Acknowledgements

We thank AWS, Lambda Labs, Nebius, OCI, and Vultr for providing us with testbeds. We especially thank Kaichao You, Lequn Chen, Zhen Huang, Zhenyu Gu, Costin Raiciu, Scott Shenker, Ion Stoica for their discussions and feedback. This research is supported by gifts from Accenture, AMD, Anyscale, AWS, Broadcom, Cisco, Google, IBM, Intel, Intesa Sanpaolo, Lambda, Lightspeed, Mibura, Microsoft, NVIDIA, Samsung SDS, and SAP.
