---
title: "UCCL"
slug: about-uccl
description: "Introducing UCCL, an extensible software transport layer for GPU networking."
category:
  - One
tags:
  - Sky Computing
pubDate: 2025-01-01
cover: https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/images/uccl.jpg
coverAlt: About
author: UCCL Team
---

# UCCL: An Extensible Software Transport Layer for GPU Networking.

UCCL is a software-only extensible transport layer for GPU networking. It is designed to augment the slow-evolving networking to meet the challenging performance requirements fast-evolving ML workloads. UCCL provides a flexible and extensible framework that allows developers to implement custom transport protocols and congestion control algorithms tailored to the specific needs of ML workloads, works with different vendor NICs, and supports many transport protocols.

## Fast-evolving ML workloads outpaces slow-evolving networking.

Machine learning (ML) workloads and their requirements for networking are evolving rapidly. Less than ten years ago, deep neural networks only had millions of parameters, and were trained atop hundreds of CPUs/GPUs with parameter servers or allreduce collective communication. After five years, large language models (LLMs) began to surge with billions of parameters, and were trained atop thousands of more powerful GPUs with multi-level parallelism and diverse collectives like allreduce, allgather, and reduce-scatter. In the recent two years, large-scale LLM serving has become the norm; prefill-decode disaggregation as an efficient serving technique, requires fast peer-to-peer communication. This year, serving Mixture-of-Experts (MoE) models like DeepSeek-V3 became very popular, featuring challenging all-to-all communication among hundreds of GPUs.

However, networking techniques especially the host network transport on RDMA NICs are hard to adapt and evolve to better suit the needs of ML workloads. Essentially, hardware changes are time-consuming and take much longer time than software changes. This can lead to a **mismatch between the application needs and existing hardware optimizations**, which often translates into poor performance. 

* Meta has reported that DCQCN — a popular congestion control (CC) algorithm in datacenters supported by RDMA NICs—does not work well for LLM training workloads with low flow entropy and high traffic burstiness. As a result, Meta decided to disable the CC support in NICs and instead implement traffic scheduling at the application layer.
* DeepSeek disabled the CC when running large-scale all-to-all for serving MoE models. However, running a large-scale RDMA network without CC is brittle, as it can lead to deadlocks, head-of-line blocking, and pervasive congestion 
* Alibaba has observed severe performance degradation for collective communication during LLM training. This was due to the high level of flow collisions, which in turn was caused by the RDMA NICs supporting only single-flow/path per connection. To avoid this problem, Alibaba has redesigned the network topology for LLM training using a rail-optimized dual-plane architecture. However, such a redesign is costly to build and maintain. 

## UCCL: a software-only extensible transport layer for GPU networking.

UCCL is a software-only extensible transport layer for GPU networking. It is designed to address the challenges of fast-evolving ML workloads and the limitations of existing RDMA NICs. UCCL provides a flexible and extensible framework that allows developers to implement custom transport protocols and congestion control algorithms tailored to the specific needs of ML workloads.

## Key challenges addressed by UCCL.

### How to decouple the data and control paths for existing RDMA NICs?

### How to achieve hardware-level performance for software control path?

### How to support multiple vendor NICs and their different capabilities?

## Core UCCL Insights.

### Moving control paths to CPU for more states handling and faster processing compared to wimpy ARM/on-chip cores.


### Harnessing multi-path for avoiding path collision.


## Evaluation.


## Future dev plan.