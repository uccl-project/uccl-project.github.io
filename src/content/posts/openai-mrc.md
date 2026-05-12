---
title: "Reading OpenAI's MRC Through a UCCL-Tran Lens"
slug: openai-mrc
description: "A close reading of OpenAI/Microsoft/AMD/Broadcom/NVIDIA's MRC + SRv6 paper and the OCP MRC 1.0 specification, viewed from the UCCL-Tran perspective: a real step forward over traditional RoCE v2 RC, but with constraints that validate why a software-extensible transport still matters."
category:
  - One
tags:
  - MRC
  - SRv6
  - RDMA
  - Multipath
  - UCCL
pubDate: 2026-05-12
cover: https://raw.githubusercontent.com/uccl-project/uccl-project.github.io/main/assets/openai-mrc/openai-mrc.png
coverAlt: OpenAI MRC
author: Zhongjie Chen, UCCL Team
---

<p>
<strong>By: Zhongjie Chen, UCCL team
<br>
Date: May 12, 2026
</strong>
</p>

<div class="tldr">
<p>
OpenAI, together with Microsoft, AMD, Broadcom and NVIDIA, has published <strong>MRC</strong> (Multipath Reliable Connection) — a new RDMA transport that sprays each QP across hundreds of paths, actively load-balances, and rides out link/switch failures using static <strong>SRv6</strong> source routing. The OCP <a href="https://www.opencompute.org/documents/ocp-mrc-1-0-pdf">MRC 1.0 specification</a> has been released, and three different vendors (CX-8, AMD Pollara, Broadcom Thor Ultra) already implement it.
</p>
<p>
Our take: this is a <em>big step forward</em> compared to relying on lossless, single-path RoCE for AI fabrics. MRC joins AWS SRD and <a href="https://arxiv.org/pdf/2504.17307">UCCL-Tran</a> on the same architectural side of the table — <em>host-based packet spraying</em> — and brings it into multi-vendor hardware NICs. Several of the production findings strongly validate UCCL-Tran's core design choices, such as spraying with O(100) entropy values per QP, avoiding PFC and using selective retransmission, adopting more advanced congestion control rather than hardware-baked one.
</p>
<p>
References:
<a href="https://cdn.openai.com/pdf/resilient-ai-supercomputer-networking-using-mrc-and-srv6.pdf">MRC + SRv6 paper</a> ·
<a href="https://www.opencompute.org/documents/ocp-mrc-1-0-pdf">OCP MRC 1.0 spec</a> ·
<a href="https://arxiv.org/pdf/2504.17307">UCCL-Tran</a>
</p>
</div>

## 1. What MRC actually is

At a protocol level, MRC is a focused extension of RoCEv2 RC, deliberately scoped to what large-scale AI pretraining actually needs.

## 1.1 Modern Transport protocol

- **Per-packet entropy.** Every data packet carries a 32-bit entropy value (EV) striped across the UDP source port and IPv6 flow label. At QP startup the NIC builds an active EV set of typically 128–256 entries, plus a backup set, and rotates through it packet by packet. ECMP and SRv6-based source routing are both supported.
- **PFC off, lossy Ethernet.** MRC explicitly disables PFC. Spraying makes PFC's per-priority queueing approximately useless and head-of-line blocking actively harmful.
- **Congestion control: UET NSCC.** MRC uses NSCC — [UET](https://ultraethernet.org/wp-content/uploads/sites/20/2025/06/UE-Specification-6.11.25.pdf)'s Network-Signalled Congestion Control (§3.6.13.3–7). It is a **sender-side, SACK-clocked, window-based, ECN + RTT** algorithm whose goal is to keep the requestor→responder queueing delay under a configured target queuing delay.
- **Selective retransmit + packet trimming.** SACK identifies exactly which packets were lost; trimmed packets (header-only, priority-forwarded under congestion) trigger fast NACKs and let MRC distinguish congestion loss from link-failure loss.
- **Out-of-order placement.** Every data packet carries the RDMA virtual address and rkey, so the receiver can DMA each packet into its final memory location regardless of arrival order. Messages on the same QP can also complete out of order with respect to each other; the only ordering primitive left is `WRITE_WITH_IMM`, which acts as a barrier and is guaranteed to complete only after all preceding WRITEs on the QP have landed.
- **Implemented on the latest generation of smart RDMA NICs.** MRC is not a software protocol — it is baked into the data plane of three vendors' newest silicon: **NVIDIA ConnectX-8** (800 Gb/s), **AMD Pollara / Vulcano** (400/800 Gb/s), and **Broadcom Thor Ultra** (800 Gb/s). The host-side API is **`libibverbs`-compatible**: applications continue to use the standard verbs surface (QPs, MRs, work-request posting, CQ polling), so existing RDMA stacks like NCCL/RCCL plug in without a new user-space library.

> *UCCL-Tran lens.* Almost every protocol-level bet here mirrors decisions UCCL-Tran made in software: per-packet (or per-chunk) spraying across hundreds of logical paths, PFC-off lossy operation, out-of-order DMA placement, and selective retransmission instead of go-back-N. The MRC paper is, in many ways, a hardware realization of the same architectural answer — and that is exactly why we find it interesting. UCCL-Tran keeps that same surface in CPU software, so a new CC profile (e.g., receiver-driven EQDS for MoE incast), a new LB policy, or a new loss-tolerance scheme is a code change, not a new tape-out. We see MRC as raising the floor; UCCL-Tran keeps the ceiling open. **And just as importantly: MRC only runs on the very latest silicon (CX-8 / Pollara / Thor Ultra), while UCCL-Tran brings the same multipath, OoO, selective-retransmit power to the *legacy* RDMA NICs already deployed in the field — CX-5/6/7, BlueField, EFA, Thor 1/2 — without a hardware refresh, albeit with some design tradeoffs.** 

## 1.2 Multi-plane Topology
The topology piece is just as important as the transport piece. MRC leans on a key capability of modern NICs — **per-lane port breakout** — to turn one 800 Gb/s NIC port into 4×200 or 8×100 Gb/s independent network ports, and then builds **one Clos plane per lane**. This is the crucial distinction from the more familiar **multi-rail** design (e.g., Alibaba HPN, Meta's rail-optimized fabrics), where each NIC has a single port that lives in a single rail. Multi-plane via NIC breakout gets you:

- **Two switch tiers for 100K+ GPUs.** Each switch effectively has 8× the port count (a 64×800G switch becomes a 512×100G switch), so a 51.2 Tb/s switch alone can fan out to 512 NICs per tier. Two tiers cover 131,072 GPUs (vs. needing three tiers, oversubscription, or rails for a single-plane design). Two tiers means **fewer hops, lower tail latency, fewer optics (~2/3), fewer switches (~3/5)**, and fewer places for partial failures to hide.
- **Failure blast radius shrinks by an order of magnitude.** Losing one T0–T1 link removes 1/256 ≈ **0.4%** of a NIC's capacity in an 8-plane network, versus ~3% in a single-plane 800 Gb/s design. Losing one *NIC-side* port costs 12% of NIC bandwidth — survivable, the job keeps running on the remaining planes. Multi-rail can't ride out a port loss this gracefully because each rail typically maps to a different NIC, and a rail going down takes that NIC out of the job.
- **Locality is much easier.** A T0 switch reaches 256 NICs in one hop instead of 32, so collectives like all-gather and ring-attention can exploit T0-local placement far more often, cutting load on the T1 layer.
- **Built-in load-balancing leverage.** MRC's per-packet EV spraying directly fills all planes equally; the topology and the transport are co-designed for this.

> *UCCL-Tran lens.* We think multi-plane via NIC port breakout is **the right direction** — it gets you two-tier 100K-GPU reach, an order-of-magnitude smaller failure blast radius, and a topology that composes naturally with packet-level spraying. UCCL-Tran is fully on board with this destination. That said, the UCCL paper makes the equally important practical point that **rebuilding the fabric is slow and expensive**: new hardware, physical cabling, switch SKUs, optics inventory, and operator playbooks all have to change in lockstep. Most clusters today are single-plane or rail-optimized, and they will remain so for years. Software multipath transports like UCCL-Tran exist precisely to deliver most of the collision-avoidance benefit *on the fabric you already have*, while operators plan the longer multi-plane refresh. The two efforts are sequential, not competing.

## 1.3. SRv6-based Source Routing

The transport story gets most of the airtime, but the operational story is just as important — and arguably the harder thing to replicate. In the OpenAI production deployments, dynamic routing is *disabled* and each EV instead maps algorithmically (via SRv6 uSID templates) to a deterministic physical path. With SRv6 uSID source routing:

- Every EV deterministically encodes a physical path. Probes from a per-node Clustermapper agent take *exactly* the same path an MRC data packet would. There is no dynamic-routing layer in between to lie to you.
- Switches forward SRv6 in the dataplane at line rate. So every host can probe every directly attached T0 (and a sample of T1s) every millisecond. This gives ground truth about forwarding-plane health, independent of the switch control plane.
- Localizing a failure becomes trivial: if T0-loopback probes succeed but T1-loopback probes fail, the bad component is the T0–T1 link.

This is operationally a much better story than pingmesh-style end-to-end probing on hash-based ECMP, where you fundamentally cannot tell which physical path a probe took.

> *UCCL-Tran lens.* This is the area where MRC has a clear advantage over a pure software transport over commodity ECMP fabrics: source routing gives you *deterministic path↔EV mapping* that we cannot get from hash-based ECMP today. We see this as complementary rather than competitive — a software transport like UCCL-Tran would benefit enormously from being able to carry SRv6 segments on outgoing packets. The OCP MRC spec being open is genuinely good news here.

## 2. Even Silver Has Its Tarnish: A Few Limitations of MRC

MRC is genuinely impressive engineering but it still has some limitations that worth noting:

- **WRITE(w/ IMM) only verbs.** Only `RDMA WRITE` and `RDMA WRITE_WITH_IMM` are on the wire — and this is not just a packaging decision, it falls out of how MRC sprays packets. To allow out-of-order placement, MRC carries an **RETH header (remote virtual address + rkey) on *every* data packet**, so each packet is self-contained and can DMA directly into its final memory location. That trick is specific to one-sided WRITEs; it has no natural analog for `READ` (the responder, not the requestor, would need to spray), for `ATOMIC` (single-target, serialized), or for two-sided `SEND/RECV` (no remote address to begin with — the receiver picks the buffer via posted RQ entries). So the verb restriction is a direct consequence of the multipath data plane, not a temporary scoping choice.

  This matters more than it sounds. The "WRITE-only" pattern is fine for synchronous pretraining collectives, but it is awkward for several important newer workloads:

  - **MoE dispatch / expert-parallel all-to-all** (e.g., [DeepEP](https://github.com/deepseek-ai/DeepEP)) increasingly relies on **`ATOMIC` fetch-and-add** for fast, lock-free token-count exchange between senders and experts. Earlier DeepEP versions did fall back to `WRITE_WITH_IMM`, but that forced the receiver GPU to poll the CQ and re-post RQ entries on the critical path — extra GPU work that competes with the dispatch kernel and is generally regarded as a worse design. The switch to the atomic-based path landed in commit [2d0cf41](https://github.com/deepseek-ai/DeepEP/commit/2d0cf41).
  - **KV transfer for prefill–decode disaggregation** wants `READ` so that the decode side pulls KV on demand without a coordination round-trip.
  - **Parameter-server / control-plane traffic** wants two-sided `SEND/RECV`.

- **`WRITE_WITH_IMM` has a small in-flight cap.** `WRITE_WITH_IMM` is not just "WRITE plus 4 bytes." The immediate-data CQE must be delivered to the responder **in order with respect to all prior WRITEs on the QP** — i.e., a `WRITE_WITH_IMM` cannot complete until every preceding WRITE has landed. In a sprayed, out-of-order data plane that means the NIC has to track per-QP barrier state and hold completion resources for every outstanding `WRITE_WITH_IMM`, which is exactly the kind of bookkeeping that does not scale on-chip. As a result MRC implementations cap the number of in-flight `WRITE_WITH_IMM` operations per QP (the spec calls this out and adds a dedicated "Inflight WriteImm limit exceeded" NACK code). Workloads that try to use `WRITE_WITH_IMM` as a fine-grained signaling primitive — one immediate per chunk — will hit this cap before they hit bandwidth.
- **Built into the newest silicon only.** MRC ships on CX-8, AMD Pollara/Vulcano, and Broadcom Thor Ultra. The very large installed base of CX-5/6/7, BlueField, EFA, and Thor 1/2 cannot run MRC at all — a fleet-wide upgrade is on the order of years and many billions of dollars.
- **Last-hop incast.** NSCC is solid, but in practice MRC leans on packet trimming + selective retransmit to absorb receiver-side bursts. For workloads with very skewed receiver-side hot spots (MoE serving with hot experts, prefill-decode disaggregation, irregular all-to-all), a receiver-driven scheduler (EQDS-style) is a strictly better answer — and it's unclear whether MRC can support this.



