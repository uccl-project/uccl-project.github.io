---
title: "A Practitioner Guide to AWS EFA Programming"
slug: efa-programming
description: "Programming AWS EFA NICs for efficient GPU communication."
category:
  - One
tags:
  - RDMA
  - EFA
pubDate: 2026-04-05
cover: https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/efa-programming/efa-programming.png
coverAlt: UCCL-EP
author: UCCL Team
---

<p>
<strong>By: <a href="https://yangzhou1997.github.io/">Yang Zhou</a> (UC Davis), and the UCCL team
<br>
Date: April 05, 2026
</strong>
</p>

<div class="tldr">
<p>
We talk about how to program AWS EFA NICs for GPU workloads <strong>UCCL-EP</strong>
</p>
<p>
Code: <a href="https://github.com/uccl-project/uccl/blob/main/experimental/misc/efa_rdma_write.cc">uccl-project/uccl/ep</a> (Apache-2.0)
</p>
</div>

## Introduction

AWS is using customized RDMA NICs called EFA, for example, in their Hopper GPUs p5, p5e, p5en VMs and Blacewell GPUs p6 VMs. 
EFA NICs run a special type of multi-path transport called SRD paper [link](https://assets.amazon.science/a6/34/41496f64421faafa1cbe301c007c/a-cloud-optimized-transport-protocol-for-elastic-and-scalable-hpc.pdf). 
It supports efficient multi-pathing to avoid single-path network congestion in the datacenter networks without relying on PFC that is considered hard to managet at large scale. 

[Insert single_path.png and multi_path.png. Mention that multi-pathing allows the NIC to dynamically adjust how much traffic it should inject on different paths. ]


However, AWS EFA was known to have different behaviors from normal RDMA NICs like NVIDIA CX-7 and Broadcom Thor-2, especially when used with GPUs. 
For a long time, people do not need to directly program EFA NICs, but just NCCL with [NCCL-EFA plugin](https://github.com/aws/aws-ofi-nccl). 
Recently, new communication paradiams like GPU-initiated communication (eg, DeepEP) and P2P communication (eg, KV cache transfer, RL weight sync), and efficiently supporting them requires tight integration between the GPU kernel and the RDMA devices. We cannot just rely on the one-side-fit-all NCCL collective communication. 

The UCCL team recently did many investigation on AWS EFA NICs and build efficient EP lib (API compatible with DeepEP) and P2P lib that support heterogeneous RDMA NICs, including the AWS EFA NICs. 
Thus, we would like to share our experience in this blogs. We will have a heavy emphasis on how to use EFA for advanced non-NCCL usage cases. 

## EFA libibverbs programming

EFA support both libibverbs and libfabric. From my perspective, libfabric is basically a wrapper for libibverbs. Most people are usually more famailar with libibverbs interface, given that other RDMA NICs like Nvidia and Broadcom all choose libibverbs as the main interface provider. Thus this blog mainly look at the EFA libibverbs interface. 

### EFA create QPs

[Showing code example]
check uccl efa_rdma_write.cc

No need to exchange do RTS, RTR.

### EFA address handler

[Showing code example]
check uccl efa_rdma_write.cc

Like UDP, specify address hander for each WR (work request), so that EFA NIC can send data to the dst QPs

This also means one EFA QP can be used to send to multiple remote EFA NICs, thus being connection-less. 

### EFA write and write-with-imm

[Showing code example]
check uccl efa_rdma_write.cc

Generally, since EFAv2 on p5 instances, EFA support these operations. 

EFA write and write-with-imm

### Other EFA operations

EFA also support EFA read, and EFA send recv, but these operations are less used in GPU communication and are less flexible. 
I also want to note that some version of EFA NICs only support send/recv for message that is under a single MTU size (eg, under 9KB).

## EFA ordering

EFA does not natively support atomics, not mention atomics over GPU memory. 
It also does not guarantee ordering between any two operations: say, GPU A does two EFA write-with-imms, GPU B might see the later write-with-imm arrive first from its completion queue polling; GPU A might also see the later one succeeding first from its completion queue. 
DeepEP EP communication library heavily relies on the write+atomics: GPU A does write1, write2, write3, then atomic to GPU B, while GPU B is polling the atomic counter to decide if previous writes have arrived. 

To support atomics, we can use write-with-imm to emulate atomics by the CPU proxy thread in the receiver side. Basically, we can allocate a cudaMemoryHost memory on CPU memory that is accessible to GPU and EFA. When reciveing a empty write-with-imm, the proxy will does CPU atomics 

Code like: 
```
counter->fetch_add(std::memory_order_release)
```

The GPU kernel code could be: 
```
// please parse uccl ep code to 
```

Next, to guarantee write-then-atomic ordering, we needs some kinds of sequence number in both write and atomics (emualated by write-with-imm). This is actually a solved problem in TCP reordering handling---we need to attach sequence number to every write and we can do it by always use write-with-imm. Within each write-with-imm, we embed monolithically sequence counter into the imm and let the receiver side decide if all previous writes have arrived when receiving a empty write-with-imm. 

The receiver-side logic looks like: 
```
ce = poll_cq()
if ce is empty: // emulated atomic
    if all previous writes have arrived: update atomic counter
    else: add to pending atomics queue
else: // real write
    add write to finished write queue
    check if there is an pending atomic operation that can really fire. 
```

There is also another approach: the sender can hold the atomic operation untill it polls the completiton of all previous write operation. However, doing so will increase the latency, as the sender needs to wait half RTT until the last write finishes (indicated by a transport-layer ack packet). UCCL-EP shows this is less efficient than the sender-driven appraoch. 

[Insert sender_vs_receiver_latency.png]

## EFA 

## EFA performance

EFA performance is pretty good for large messages over dozens of MB, on par with NVIDIA/Broadcom NICs, but far more reliable (at least I have never seen a retranmission count exceed error 12, while I see many for other RDMA vendors). 
From my perspective, this is exactly because of EFA's advanced host-driven multi-pathing feature, that handles network congestion in a principly better way than other vendors' NIC who still uses single path possible with fancy adaptive routing + PFC feature inside the network core. 

EFA performance sacrifaces small-message (< 8KB) performance and atomics+ordering support. For example, I usually observes 15-25 us RTT on EFA networks, while ~10us RTT for Nvidia RDMA networks. Lacks of atomics+ordering hinders many advanced communication library's support such as DeepEP. But UCCL has full support for it and address the atomics and ordering constraints. 