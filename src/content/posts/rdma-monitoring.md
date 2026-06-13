---
title: "rdmatop: Real-Time, Vendor-Neutral RDMA Network Monitoring"
slug: rdma-monitoring
description: "rdmatop is htop for RDMA traffic: a vendor-neutral TUI for real-time RDMA NIC monitoring across InfiniBand, RoCE, and EFA, shown via real NCCL and NVSHMEM debugging case studies."
category:
  - One
tags:
  - RDMA
  - EFA
  - InfiniBand
  - RoCE
  - NCCL
  - NVSHMEM
  - Monitoring
pubDate: 2026-06-13
cover: /rdma-monitoring/rdma-monitoring.png
coverAlt: rdmatop
author: UCCL Team
---
**By: UCCL Team -- June 13, 2026**

<div class="tldr">
<p>
RDMA is the backbone of modern GPU communication, yet most of us run it blind: when throughput is half of what it should be, there is rarely an easy way to see <em>which</em> NIC is hot, which is idle, or whether the bottleneck is on the transmit or the receive side. We built <strong><a href="https://github.com/uccl-project/rdmatop">rdmatop</a></strong>—"htop, but for RDMA traffic"—a real-time TUI that works across RDMA providers (NVIDIA ConnectX, AWS EFA, Broadcom, and any Linux RDMA device) through the RDMA netlink interface. In this post we motivate why such a tool is needed, explain why production dashboards fall short for interactive debugging, and walk through real NCCL and NVSHMEM performance issues that a per-NIC, per-process monitor makes obvious at a glance.
</p>
</div>

## Introduction

If you run InfiniBand fabrics, you have probably used [`ibtop`](https://github.com/jhammond/ibtop)—a small but invaluable tool that reads InfiniBand hardware performance counters (via the UMAD interface) and organizes bandwidth and traffic by job or host. It answers the everyday operational question: *who is using the fabric, and how much?*

The trouble is that the RDMA world is no longer just InfiniBand. GPU clusters today run RDMA over an expanding set of providers, each with its own NIC, counter definitions, and behavioral quirks:

- **NVIDIA / Mellanox ConnectX** (RoCE and InfiniBand)
- **AWS EFA** (Elastic Fabric Adapter)
- **Broadcom Thor / `bnxt`** NICs
- **AMD** Pensando / Pollara NICs

An InfiniBand-only tool like `ibtop` cannot see any of these, and writing a separate monitor per vendor does not scale. What practitioners actually need is a **provider-agnostic** view of RDMA traffic.

That is exactly what `rdmatop` provides. Instead of talking to one vendor's counters, it uses **RDMA netlink**—the same interface behind the standard `rdma statistic` command—so it works on any Linux RDMA device. It runs in four steps:

1. **Discover** RDMA devices and ports via netlink
2. **Collect** hardware counters per device/port
3. **Map** queue pairs (QPs) back to the processes that own them
4. **Compute** throughput from interval snapshots

The result is a live terminal dashboard of per-device throughput (Gb/s, packets/s, drops), RDMA read/write counters, retransmissions, and—crucially—**which process is driving each device**. That per-NIC, per-process, Tx-vs-Rx visibility is what turns "the job is slow" into "GPU 0's traffic is all landing on a single NIC."

## Case Study 1: AWS Already Has an EFA Exporter—So Why a TUI?

Isn't this already solved on AWS? Not for *debugging*. AWS ships an [EFA node exporter](https://github.com/awslabs/awsome-distributed-ai/tree/main/4.validation_and_observability/3.efa-node-exporter) that scrapes EFA RDMA counters into Prometheus and renders them in Grafana on EKS—great for long-lived, fleet-wide dashboards. But standing it up takes a custom Docker image, ECR, a Helm-installed Prometheus stack, a DaemonSet, and a hosted Grafana: the right machinery for continuous observability, the wrong machinery for answering "why is this job slow *right now*?"

On a node, a dashboard is a poor debugging instrument—coarse scrape intervals, no per-process attribution, and a whole stack to deploy just to inspect one host. That is the gap `rdmatop` fills: a single binary that shows live per-NIC, per-process Tx/Rx rates instantly, with no cluster, no ECR, and no Grafana. The case studies that follow show what that immediacy buys you.

## Case Study 2: NCCL Silently Falling Back to TCP Sockets

NCCL is the default collective library for distributed training and inference, and on EFA it should move data over RDMA through the libfabric (OFI) plugin. If that plugin is mislinked or misconfigured, NCCL silently falls back to kernel **TCP sockets**—RDMA disabled—and collective throughput can crater by up to an order of magnitude (~10×). The job still runs and still converges; it is just far slower.

The only clue is a single line in the `NCCL_DEBUG=INFO` output:

```text
# wrong — silently fell back to TCP sockets
NCCL INFO Using network Socket

# correct — using EFA via libfabric
NCCL INFO Using network Libfabric
```

In practice, nobody is reading initialization logs during a multi-node training run or a hosted inference service, and the sheer log volume buries that one line (see [uccl#734](https://github.com/uccl-project/uccl/issues/734)). `rdmatop` surfaces the fallback instantly: if NCCL is on sockets, the EFA NICs show **near-zero RDMA traffic** even while the GPUs are obviously communicating. Flat RDMA counters mean you are not on RDMA—no log archaeology required.

## Case Study 3: NVSHMEM ≤ 3.5.21 Was Stuck on a Single EFA NIC

AWS GPU instances ship with **multiple EFA NICs per node** precisely so that each GPU can drive a lot of network bandwidth. But for a long time, NVSHMEM[^1] could not take advantage of them.

In NVSHMEM **3.5.21 and earlier**, the libfabric transport bound **each GPU to exactly one EFA NIC**. No matter how many NICs the instance had, a single GPU's point-to-point throughput was capped at one NIC's bandwidth, leaving the rest of an expensive multi-NIC system idle. Workloads looked mysteriously "slow," but the application-level numbers gave no hint as to why.

This is the textbook case for an RDMA monitor. With `rdmatop` running on the node, the picture is immediate and unambiguous: **one EFA NIC pinned near line rate while its siblings sit at zero.** There is no theory to test and no guesswork—the imbalance is right there on screen. (For the full write-up of the single- vs. multi-NIC behavior, see the NVSHMEM Multi-NIC notes.[^2])

## Case Study 4: Multi-Rail Was Added, But Throughput Did Not Scale

NVSHMEM **3.6.5** added **round-robin NIC selection**, letting a single GPU spray traffic across all of its EFA NICs. With four NICs per GPU we expected point-to-point throughput to scale roughly **4×**—but all-to-all stubbornly refused to, and could even come out *slower than a single NIC*.

Running `rdmatop` on the destination node made the cause obvious: **transmit (Tx) traffic spread evenly across all NICs, but receive (Rx) traffic funneled onto a single NIC** while the others sat idle. Round-robin balanced sends but not receives—every sender picked the same remote NIC for a given destination—and that lone receive-side hotspot capped the whole job.

The fix in [NVIDIA/nvshmem#76](https://github.com/NVIDIA/nvshmem/pull/76) spreads remote-NIC selection per sender, so receives land on different NICs; with it applied, Rx balances across all NICs and throughput scales as expected. The PR has the implementation details and benchmark numbers.

## Conclusion

These case studies share a theme: the hardware was capable, but it was being used wrong—idle because traffic fell back to TCP, capped to a single rail, or funneled onto one NIC—and each failure was effectively invisible at the application layer. The job ran; it was simply slow. Each took real investigation to track down.

A per-NIC, per-process, Tx-vs-Rx monitor collapses that investigation into a glance. As RDMA fans out across EFA, ConnectX, Broadcom, and AMD, a tool that reads every one of them through a single vendor-neutral interface becomes essential rather than nice-to-have. That is the gap `rdmatop` is built to fill—`htop`, but for RDMA traffic. We welcome issues and contributions.

## References

[^1]: NVIDIA. *NVSHMEM*. [Link](https://developer.nvidia.com/nvshmem).
[^2]: pythonsheets. *NVSHMEM Multi-NIC: single-NIC vs. multi-rail behavior on AWS EFA*. 2026. [Link](https://www.pythonsheets.com/notes/appendix/nvshmem-multi-nic.html).
