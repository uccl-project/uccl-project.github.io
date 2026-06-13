---
title: "Why RDMA Monitoring Matters: A Tour of rdmatop"
slug: rdma-monitoring
description: "rdmatop is htop for RDMA traffic: a vendor-neutral TUI for real-time RDMA NIC monitoring across InfiniBand, RoCE, and EFA, shown via two NVSHMEM case studies."
category:
  - One
tags:
  - RDMA
  - EFA
  - InfiniBand
  - RoCE
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
RDMA is the backbone of modern GPU communication, yet most of us run it blind: when throughput is half of what it should be, there is rarely an easy way to see <em>which</em> NIC is hot, which is idle, or whether the bottleneck is on the transmit or the receive side. We built <strong><a href="https://github.com/uccl-project/rdmatop">rdmatop</a></strong>—"htop, but for RDMA traffic"—a real-time TUI that works across RDMA providers (NVIDIA ConnectX, AWS EFA, Broadcom, and any Linux RDMA device) through the RDMA netlink interface. In this post we motivate why such a tool is needed, explain why production dashboards fall short for interactive debugging, and walk through two real NVSHMEM multi-NIC performance issues that a per-NIC, per-process monitor makes obvious at a glance.
</p>
</div>

## Introduction

If you run InfiniBand fabrics, you have probably used [`ibtop`](https://github.com/jhammond/ibtop)—a small but invaluable tool that reads InfiniBand hardware performance counters (via the UMAD interface) and organizes bandwidth and traffic by job or host. It answers the everyday operational question: *who is using the fabric, and how much?*

The trouble is that the RDMA world is no longer just InfiniBand. GPU clusters today run RDMA over an expanding set of providers, each with its own NIC, counter definitions, and behavioral quirks:

- **NVIDIA / Mellanox ConnectX** (RoCE and InfiniBand)
- **[AWS EFA](/posts/efa-programming)** (Elastic Fabric Adapter)
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

It is fair to ask whether this is already solved on AWS. It is not, at least not for *debugging*. AWS publishes an [EFA node exporter](https://github.com/awslabs/awsome-distributed-ai/tree/main/4.validation_and_observability/3.efa-node-exporter): a Prometheus exporter that reads EFA RDMA counters from `/sys/class/infiniband/<device>`—RDMA read/write bytes and operations, packet and byte Tx/Rx, port errors—and runs as a DaemonSet on every EKS node. Scraped into Prometheus and rendered in Grafana, it provides exactly the long-lived, cluster-wide dashboards a production fleet should have.

The catch is everything you must stand up first: build a custom Docker image, push it to ECR, add a Helm repo, install the Prometheus stack, deploy the DaemonSet, and **host Grafana**. That is the right amount of machinery for continuous observability and the wrong amount for *debugging*. When you are SSH'd into a node trying to understand why a job is slow *right now*, a Grafana dashboard is a poor instrument: the scrape interval is coarse, there is no per-process attribution, and standing up the whole stack just to inspect one node is overkill. Dashboards are for watching a fleet over time; debugging wants a tool you point at a node and read instantly.

That is the gap `rdmatop` fills: a single binary you run on the node to immediately see live per-NIC, per-process Tx/Rx rates—no cluster, no ECR, no Grafana. The next two case studies show what that immediacy buys you.

## Case Study 2: NVSHMEM ≤ 3.5.21 Was Stuck on a Single EFA NIC

AWS GPU instances ship with **multiple EFA NICs per node** precisely so that each GPU can drive a lot of network bandwidth. But for a long time, NVSHMEM[^1] could not take advantage of them.

In NVSHMEM **3.5.21 and earlier**, the libfabric transport bound **each GPU to exactly one EFA NIC**. No matter how many NICs the instance had, a single GPU's point-to-point throughput was capped at one NIC's bandwidth, leaving the rest of an expensive multi-NIC system idle. Workloads looked mysteriously "slow," but the application-level numbers gave no hint as to why.

This is the textbook case for an RDMA monitor. With `rdmatop` running on the node, the picture is immediate and unambiguous: **one EFA NIC pinned near line rate while its siblings sit at zero.** There is no theory to test and no guesswork—the imbalance is right there on screen. (For the full write-up of the single- vs. multi-NIC behavior, see the NVSHMEM Multi-NIC notes.[^2])

## Case Study 3: Multi-Rail Was Added, But Throughput Did Not Scale

NVSHMEM **3.6.5** added **round-robin NIC selection**, letting a single GPU spray traffic across all of its EFA NICs. With four NICs per GPU we expected point-to-point throughput to scale roughly **4×**—but all-to-all stubbornly refused to, and could even come out *slower than a single NIC*.

Running `rdmatop` on the destination node made the cause obvious: **transmit (Tx) traffic spread evenly across all NICs, but receive (Rx) traffic funneled onto a single NIC** while the others sat idle. Round-robin balanced sends but not receives—every sender picked the same remote NIC for a given destination—and that lone receive-side hotspot capped the whole job.

The fix in [NVIDIA/nvshmem#76](https://github.com/NVIDIA/nvshmem/pull/76) spreads remote-NIC selection per sender, so receives land on different NICs; with it applied, Rx balances across all NICs and throughput scales as expected. The PR has the implementation details and benchmark numbers.

## Conclusion

Both NVSHMEM issues share a theme: the network *had* the bandwidth, but it was being used unevenly, and that imbalance was effectively invisible at the application layer. A single saturated NIC—on the send side under a single rail, the receive side under multi-rail—was enough to cap an entire job. Each took real investigation to track down.

A per-NIC, per-process, Tx-vs-Rx monitor collapses that investigation into a glance. As RDMA fans out across EFA, ConnectX, Broadcom, and AMD, a tool that reads every one of them through a single vendor-neutral interface becomes essential rather than nice-to-have. That is the gap `rdmatop` is built to fill—`htop`, but for RDMA traffic. We welcome issues and contributions.

[^1]: NVIDIA. [*NVSHMEM*](https://developer.nvidia.com/nvshmem).
[^2]: pythonsheets. [*NVSHMEM Multi-NIC: single-NIC vs. multi-rail behavior on AWS EFA*](https://www.pythonsheets.com/notes/appendix/nvshmem-multi-nic.html). 2026.
