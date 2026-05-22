---
title: "Reading OpenAI's MRC Through a UCCL-Tran Lens"
slug: openai-mrc
description: "A close reading of OpenAI/Microsoft/AMD/Broadcom/NVIDIA's MRC + SRv6 paper and the OCP MRC 1.0 specification, viewed from the UCCL-Tran perspective: a real step forward over traditional RoCE v2 RC, but with constraints that validate why a software-extensible transport still matters."
category:
  - One
tags:
  - MRC
  - SRv6
  - SRD
  - Multi-plane
  - UCCL-Tran
pubDate: 2026-05-21
cover: /openai-mrc/openai-mrc.png
coverAlt: OpenAI MRC
author: Zhongjie Chen, Xizhi Zhang, the UCCL Team
---

<p>
<strong>By: Zhongjie Chen, Xizhi Zhang, and the UCCL team
<br>
Date: May 21, 2026
</strong>
</p>

<div class="tldr">
<p>
<strong><a href="https://cdn.openai.com/pdf/resilient-ai-supercomputer-networking-using-mrc-and-srv6.pdf">MRC</a></strong> (Multipath Reliable Connection) is a new RDMA transport from OpenAI, Microsoft, AMD, Broadcom, and NVIDIA. Based on RoCEv2 RC, it adds <strong>per-QP packet spraying, out-of-order delivery, and selective retransmission, UET's congestion control</strong>, together with <strong> multi-plane topology and static SRv6 source routing</strong>. MRC also releases an <a href="https://www.opencompute.org/documents/ocp-mrc-1-0-pdf">OCP spec</a>, and CX-8, AMD Pollara, and Broadcom Thor Ultra already ship it. This blog aims to understand the MRC protocol in depth and compare it with other alternative solutions IB/RoCEv2, UEC, AWS SRD, and UCCL-Tran. 
</p>
</div>

## 1. What MRC actually is

## 1.1 Modern Transport for AI Workloads

- **Packet spraying.** Each MRC QP maintains an Entropy Value (EV) profile. At QP startup the NIC builds an active EV set of typically 128–256 entries, plus a backup set. A programmable Path Selection stage selects an EV for each packet. Each EV is mapped to a specific path, so packets get sprayed across multiple paths. MRC provides three different EV→path mapping models, including ECMP hashing, Structured EV, and SRv6 source routing.  

<p align="center">
  <img src="/openai-mrc/mrc-multipath.png" alt="MRC multipath data path: ibv_post_send hands a message to the MRC QP, Path Selection sprays packets across multiple EV-defined paths, and per-path SACK/NACK feedback (with ECN or TRIMMED signals) flips EV entries to SKIP in the EV Profile table to avoid congested or lossy paths" width="430"/>
</p>
<p align="center"><em>Figure 1: After the message is handed to the MRC QP, the Path Selection stage picks an EV from the QP's active EV set for every packet, then packets get sprayed across the network paths in parallel. The sender uses SACK/NACK to update the EV state: an EV that sees ECN or trims gets flipped from `GOOD` to `SKIP` and is temporarily removed from the active set, so subsequent packets steer away from the congestion path.</em></p>

&nbsp;&nbsp;&nbsp;&nbsp;🔧 **Fixes RC pain point — *single-path-per-QP*.** RC QP pins all its traffic to one 5-tuple, so a single ECMP hash collision or one bad link silently bottlenecks (or kills) the QP and leaves the rest of the fabric unused.

- **Congestion control: UET NSCC.** MRC uses NSCC — UET's Network-Signalled Congestion Control [^1]. It's a **sender-driven, SACK-clocked, window-based** algorithm that uses both ECN and RTT to keep queueing delay within a configured target without sacrificing throughput. RTT provides quantitative congestion information, but it is a lagging signal; ECN is leading but coarser, giving a coarse one-bit signal of congestion.  Combining the two gives NSCC both faster reaction and more precise control.

<p align="center">
  <img src="/openai-mrc/cc.png" alt="MRC congestion control: NSCC uses both ECN and RTT to keep queueing delay within a target without sacrificing throughput" width="530"/>
</p>
<p align="center"><em>Figure 2: NSCC uses ECN and RTT to adjust the window. When ECN=0, it increases the window at different speeds depending on the delay. When ECN=1 and delay is high, the network is congested and NSCC decreases the congestion window. When ECN=1 but delay is low, congestion may occurs in a single path, so NSCC neither increases nor decreases the window and leaves it to the load balance mechanism.</em></p>

&nbsp;&nbsp;&nbsp;&nbsp;🔧 **Fixes RC pain point — *DCQCN*.** RoCEv2 relies on DCQCN, a rate-based congestion control scheme driven by ECN feedback, which suffers from slow convergence and throughput loss. In comparison, NSCC converges faster because of its window-based control and the combination of RTT and ECN. More importantly, NSCC also tackles a multipath-specific question: when should the sender switch paths, and when should it reduce the congestion window? NSCC leaves mild, single-path congestion to load balancing, and reduces the congestion window only when congestion is severe or widespread.

- **Selective retransmission + packet trimming.** MRC disables Priority Flow Control (PFC) and embraces lossy network with efficient selective retransmission. SACKs are used to identify exactly which packets were lost; trimmed packets (header-only, priority-forwarded under congestion) trigger fast NACKs and let MRC distinguish congestion loss from link-failure loss.  


<p align="center">
  <img src="/openai-mrc/sack-trim.png" alt="Selective retransmission + packet trimming: the SRC NIC sprays DATA across two switch paths; the DST NIC reports gaps via a SACK bitmap so SRC retransmits only the missing PSNs (RTX DATA); when a switch drops a payload due to congestion, packet trimming forwards a header-only stub that fires a fast NACK, so SRC retransmits exactly that packet without Go-Back-N" width="520"/>
</p>
<p align="center"><em>Figure 3: The destination NIC reports gaps to the source via a SACK bitmap, so the source retransmits only the missing PSNs instead of Go-Back-N. When a switch decides to drop a payload due to congestion, the switch forwards a header-only stub that elicits an immediate NACK, letting the source patch the loss.</em></p>

  &nbsp;&nbsp;&nbsp;&nbsp;🔧 **Fixes RC pain point — *Go-Back-N retransmission*.** RC drops everything after a single lost PSN and replays the entire window, which is catastrophic under spraying and lossy operation.

- **Out-of-order placement.** Every data packet carries the virtual address, rkey and dma length through RDMA Extended Transport Header (RETH), so the receiver can DMA each packet into its final memory location regardless of arrival order. Messages on the same QP can also complete out of order with respect to each other; the only ordering primitive left is `WRITE_WITH_IMM`, which acts as a barrier and is guaranteed to complete only after all preceding WRITEs on the QP have landed.

<p align="center">
  <img src="/openai-mrc/ooo-placement.png" alt="Out-of-order placement: every WRITE FIRST/MIDDLE/LAST packet carries its own RETH (VA, rkey, len) so the receiver NIC DMAs each payload directly to its final memory slot regardless of arrival order; a trailing WRITE_WITH_IMM is staged until all preceding WRITEs on the QP land, then generates a CQE to the CPU/GPU" width="520"/>
</p>
<p align="center"><em>Figure 4: Every <code>WRITE packet (FIRST/MIDDLE/LAST/LAST with IMM/ONLY/ONLY with IMM)</code> packet carries its own RETH (<code>VA</code>, <code>rkey</code>, <code>len</code>), so the receiver NIC can DMA each payload directly into its final memory slot in any arrival order. <code>WRITE_WITH_IMM</code> is always staged inside the NIC until every preceding WRITE on the QP has landed, then generates a CQE to the CPU/GPU.</em></p>

&nbsp;&nbsp;&nbsp;&nbsp;🔧 **Fixes RC pain point — *strict in-order delivery*.** RC requires Packet Sequence Numbers (PSNs) to land in order on the wire, which is fundamentally incompatible with spraying — any reordering becomes a loss event.

- **Implemented on the latest generation of smart RDMA NICs.** MRC is not a software protocol — it is baked into the data plane of three vendors' newest silicon: **NVIDIA ConnectX-8** (800 Gb/s), **AMD Pollara / Vulcano** (400/800 Gb/s), and **Broadcom Thor Ultra** (800 Gb/s). The host-side API is **`libibverbs`-compatible**: applications continue to use the standard verbs surface (QPs, MRs, work-request posting, CQ polling), so existing RDMA stacks like NCCL/RCCL plug in without a new user-space library.

> *UCCL-Tran lens.* Many protocol-level bet here mirrors decisions UCCL-Tran [^3] made in software: per-packet (or per-chunk) spraying across hundreds of logical paths, PFC-off lossy operation, out-of-order DMA placement, and selective retransmission instead of go-back-N. The MRC paper is, in many ways, a hardware realization of the same architectural answer — and that is exactly why we find it interesting. UCCL-Tran keeps that same surface in CPU software, so a new CC profile (e.g., receiver-driven EQDS for MoE incast), a new LB policy, or a new loss-tolerance scheme is a code change, not a new tape-out. We see MRC as raising the floor; UCCL-Tran keeps the ceiling open. **And just as importantly: MRC only runs on the very latest silicon (CX-8 / Pollara / Thor Ultra), while UCCL-Tran brings the same multipath, OoO, selective-retransmit power to the *legacy* RDMA NICs already deployed in the field — CX-5/6/7, BlueField, EFA, Thor 1/2 — without a hardware refresh, albeit with some design tradeoffs.** 

## 1.2 Multi-plane Topology
The topology piece is just as important as the transport piece. MRC leans on a key capability of modern NICs — **per-lane port breakout** — to turn one 800 Gb/s NIC port into 4×200 or 8×100 Gb/s independent network ports, and then builds **one Clos plane per lane**. This is the crucial distinction from the more familiar **multi-rail** design (e.g., Alibaba HPN, Meta's rail-optimized fabrics), where each NIC has a single port that lives in a single rail.

<p align="center">
  <img src="/openai-mrc/topo.png" alt="Two topology options for a 100K-GPU AI cluster: (a) a conventional 3-tier single-plane 800 Gb/s Clos that needs 64 pods × 32 T0s × 32 NICs = 65,536 NICs to cover the cluster, and (b) the MRC multi-plane design that breaks each 800 Gb/s NIC into 8×100 Gb/s lanes, builds one Clos plane per lane, and reaches 512 T0s × 256 NICs = 131,072 NICs across 8 planes in only 2 switch tiers" width="500"/>
</p>
<p align="center"><em>Figure 5: Two ways to wire 100K+ GPUs at full bisection. (a) A conventional <strong>3-tier single-plane</strong> 800 Gb/s Clos: 64 pods × 32 T0s × 32 NICs = <strong>65,536 NICs</strong>, with the longest path crossing 5 switch hops. (b) MRC's <strong>2-tier 8×100 Gb/s multi-plane</strong> design: each NIC's 800 Gb/s port is broken into 8 lanes that feed 8 parallel Clos planes built from the same 51.2 Tb/s switches (now seen as 512-port at 100G), reaching <strong>131,072 NICs</strong> with the longest path crossing only 3 switch hops, ~⅔ the optics and ~⅗ the switches, and a ~10× smaller blast radius per T0–T1 link loss.</em></p>

Multi-plane via NIC breakout gets you:

- **Two switch tiers for 100K+ GPUs.** Each switch effectively has 8× the port count (a 64×800G switch becomes a 512×100G switch), so a 51.2 Tb/s switch alone can fan out to 512 NICs per tier. Two tiers cover 131,072 GPUs (vs. needing three tiers, oversubscription, or rails for a single-plane design). Two tiers means **fewer hops, lower tail latency, fewer optics (~2/3), fewer switches (~3/5)**, and fewer places for partial failures to hide.
- **Failure blast radius shrinks by an order of magnitude.** Losing one T0–T1 link removes 1/256 ≈ **0.4%** of a NIC's capacity in an 8-plane network, versus ~3% in a single-plane 800 Gb/s design. Losing one *NIC-side* port costs 12% of NIC bandwidth — survivable, the job keeps running on the remaining planes. Multi-rail can't ride out a port loss this gracefully because each rail typically maps to a different NIC, and a rail going down takes that NIC out of the job.
- **Locality is much easier.** A T0 switch reaches 256 NICs in one hop instead of 32, so collectives like all-gather and ring-attention can exploit T0-local placement far more often, cutting load on the T1 layer.
- **Built-in load-balancing leverage.** MRC's per-packet EV spraying directly fills all planes equally; the topology and the transport are co-designed for this.

Multi-plane is also what makes MRC's **per-EV failure detection and recovery** practical. Because every EV maps to a specific path through a specific plane, the sender NIC can keep a small EV state table and treat path health as a per-entry property rather than a fabric-wide event. When a packet on EV 42 times out (figure below, left), that EV is locally marked `BAD` and immediately removed from the spraying set — the other EVs keep flowing through the remaining planes, so the QP never stalls and the blast radius of a single link fault stays at one entry out of hundreds.

<p align="center">
  <img src="/openai-mrc/topo-fail.png" alt="Per-EV failure detection: a link drop in one plane causes packet EV 42 to time out at the source NIC, which marks EV 42 as BAD in its local EV state table and stops spraying onto that path while the other EVs continue to deliver traffic through the surviving planes" width="900"/>
</p>
<p align="center"><em>Figure 6: When a link fails, only the EV (path) traversing it is timed out and marked <code>BAD</code>; the other EVs continue to drain traffic through the remaining planes.</em></p>

Recovery is the symmetric operation (figure below). The source NIC periodically emits tiny **probe packets** on its `BAD` EVs; as soon as a probe round-trips successfully, the corresponding entry flips back to `GOOD` and the EV rejoins the active spraying set. There is no fabric-wide reconvergence event, no controller in the loop, and no need to renegotiate the QP — the data plane self-heals at EV granularity, on the order of an RTT after the link comes back.

<p align="center">
  <img src="/openai-mrc/topo-recover.png" alt="Per-EV failure recovery: the source NIC sends probe packets on BAD EVs, and once a probe on EV 42 returns successfully it flips the EV state back to GOOD and resumes spraying onto that path, all without any control-plane coordination" width="900"/>
</p>
<p align="center"><em>Figure 7: Probe packets on a <code>BAD</code> EV elicit a response once the link is back; the entry flips to <code>GOOD</code> and is re-added to the spraying set within an RTT, with no control-plane involvement.</em></p>

> *UCCL-Tran lens.* We think multi-plane via NIC port breakout is **the right direction** — it gets you two-tier 100K-GPU reach, an order-of-magnitude smaller failure blast radius, and a topology that composes naturally with packet-level spraying. UCCL-Tran is fully on board with this destination. That said, the UCCL paper makes the equally important practical point that **rebuilding the fabric is slow and expensive**: new hardware, physical cabling, switch SKUs, optics inventory, and operator playbooks all have to change in lockstep. Most clusters today are single-plane or rail-optimized, and they will remain so for years. Software multipath transports like UCCL-Tran exist precisely to deliver most of the collision-avoidance benefit *on the fabric you already have*, while operators plan the longer multi-plane refresh. The two efforts are sequential, not competing.

## 1.3. SRv6 uSID Source Routing

The OCP MRC 1.0 spec actually defines three ways the EV in a packet can be turned into a physical path — ECMP hashing, Structured EV, and SRv6 uSID source routing. The following table shows the three models side by side:

<p align="center"><em>Table 1: Comparison of three EV→Path Mapping Models</em></p>

| Model                | EV → Path Mapping                                                                                     | Strength                                                                                         | Weakness                                                                                       |
|----------------------|-------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| ECMP Hashing         | The switch hashes the EV along with specific packet header fields (UDP port, QPN, etc.) to pick a path. | Easy to deploy | No deterministic path control <br> No visibility into which path a packet took |
| Structured EV       | The EV is split into fields (UDP port + IPv6 flow label) that directly map to switch forwarding decisions.| Deterministic control | Switch support required |
| SRv6 uSID Source Routing (**MRC default**) | SRv6 explicitly defines a path from source to destination using micro-segments (uSID) to identify each hop in the path. The switches forward based on the SRv6 segments. | Deterministic control <br> Standardized and observable | Extra header overhead <br> Switch support required |

The transport story gets most of the airtime, but the operational story is just as important — and arguably the harder thing to replicate. In the OpenAI production deployments, dynamic routing is *disabled* and each EV instead maps algorithmically (via SRv6 uSID templates) to a deterministic physical path. With SRv6 uSID source routing:

- Every EV deterministically encodes a physical path. Probes from a per-node Clustermapper agent take *exactly* the same path an MRC data packet would. There is no dynamic-routing layer in between to lie to you.
- Switches forward SRv6 in the dataplane at line rate. So every host can probe every directly attached T0 (and a sample of T1s) every millisecond. This gives ground truth about forwarding-plane health, independent of the switch control plane.
- Localizing a failure becomes trivial: if T0-loopback probes succeed but T1-loopback probes fail, the bad component is the T0–T1 link.
- Bad-path avoidance becomes more precise and controllable. With SRv6, changing to a replacement EV can deterministically move traffic onto a different physical path, while it cannot make the same guarantee with hash-based ECMP forwarding. With help from Clustermapper, SRv6 also lets bad-path avoidance extend beyond a single QP: the denylist can exclude bad EVs from a shared EV Profile, or exclude a failed NIC port so traffic is sprayed across the remaining planes.


This is operationally a much better story than pingmesh-style end-to-end probing on hash-based ECMP, where you fundamentally cannot tell which physical path a probe took.

> *UCCL-Tran lens.* This is the area where MRC has a clear advantage over a pure software transport over commodity ECMP fabrics: source routing gives you *deterministic path↔EV mapping* that we cannot get from hash-based ECMP today. We see this as complementary rather than competitive — a software transport like UCCL-Tran would benefit enormously from being able to carry SRv6 segments on outgoing packets. The OCP MRC spec being open is genuinely good news here.

## 2. Even Silver Has Its Tarnish: A Few Limitations of MRC

MRC is genuinely impressive engineering but it still has some limitations that worth noting:

- **WRITE(w/ IMM) only verbs.** Only `RDMA WRITE` and `RDMA WRITE_WITH_IMM` are on the wire — and this is not just a packaging decision, it falls out of how MRC sprays packets. To allow out-of-order placement, MRC carries an **RETH header (remote virtual address + rkey) on *every* data packet**, so each packet is self-contained and can DMA directly into its final memory location. That trick is specific to one-sided WRITEs; it has no natural analog for `READ` (the responder, not the requestor, would need to spray), for `ATOMIC` (single-target, serialized), or for two-sided `SEND/RECV` (no remote address to begin with — the receiver picks the buffer via posted RQ entries). So the verb restriction is a direct consequence of the multipath data plane, not a temporary scoping choice.

  This matters more than it sounds. The "WRITE-only" pattern is fine for synchronous pretraining collectives, but it is awkward for several important newer workloads:

  - **MoE dispatch / combine** (e.g., DeepEP [^2]) increasingly relies on **`ATOMIC` fetch-and-add** for fast, lock-free token-count exchange between senders and experts. Earlier DeepEP versions did use `WRITE_WITH_IMM`, but that forced the receiver GPU to poll the CQ and re-post RQ entries on the critical path — extra GPU work that competes with the dispatch kernel and is generally regarded as a worse design. The switch to the atomic-based path landed in commit [2d0cf41](https://github.com/deepseek-ai/DeepEP/commit/2d0cf41).
  - **KV transfer for PD disaggregation** wants `READ` so that the decode side pulls KV on demand without a coordination round-trip.

- **`WRITE_WITH_IMM` has a small in-flight cap.** The immediate-data CQE must be delivered to the responder **in order with respect to all prior WRITEs on the QP** — i.e., a `WRITE_WITH_IMM` cannot complete until every preceding WRITE has landed. In a sprayed, out-of-order data plane that means the NIC has to track per-QP barrier state and hold completion resources for every outstanding `WRITE_WITH_IMM`, which is exactly the kind of bookkeeping that does not scale on-chip. As a result MRC implementations cap the number of in-flight `WRITE_WITH_IMM` operations per QP (the spec calls this out and adds a dedicated "Inflight WriteImm limit exceeded" NACK code). Workloads that try to use `WRITE_WITH_IMM` as a fine-grained signaling primitive — one immediate per chunk — will hit this cap before they hit bandwidth.
- **Built into the newest silicon only.** MRC ships on CX-8, AMD Pollara/Vulcano, and Broadcom Thor Ultra. The very large installed base of CX-5/6/7, BlueField, EFA, and Thor 1/2 cannot run MRC at all — a fleet-wide upgrade is on the order of years and many billions of dollars.
- **Last-hop incast.** NSCC is solid, but in practice MRC leans on packet trimming + selective retransmit + receiver-side backpressure to absorb receiver-side bursts. For workloads with very skewed receiver-side hot spots (MoE serving with hot experts, PD disaggregation, irregular all-to-all), a receiver-driven scheduler (EQDS-style) is a strictly better answer — and it's unclear whether MRC can support this.

## 3. Tradeoff summary: MRC vs. AWS SRD vs. UCCL-Tran / UCCL-P2P

MRC, AWS SRD, and UCCL-Tran / UCCL-P2P all answer the same question — "how do we move ML traffic across a large GPU fabric reliably, with multipath, no PFC, and graceful loss recovery?" — but they make very different bets on *where* the transport lives and *how open* it is. MRC pushes the answer into a new generation of merchant silicon under an open OCP spec; AWS SRD bakes a similar answer into the closed Nitro / EFA data plane, tightly co-designed with the AWS VPC fabric [^5]; UCCL-Tran / UCCL-P2P keeps the answer in CPU software, so it can ride on the legacy RDMA NICs already deployed in the field. The table below lines up the design choices side by side so the tradeoffs — performance ceiling, hardware dependency, openness, programmability, observability, and time-to-ship — are easy to compare.

<p align="center"><em>Table 2: Comparison of different transports</em></p>

|                        | **MRC (hardware)**                              | **AWS SRD (hardware)**                          | **UCCL-Tran / UCCL-P2P (software)**                |
|------------------------|-------------------------------------------------|-------------------------------------------------|----------------------------------------------------|
| **Where transport runs**   | NIC ASIC data plane                             | Nitro / EFA NIC data plane                       | Host CPU control path + RDMA UC/RC/UD or `AF_XDP` data path |
| **Hardware requirement**   | CX-8 / Pollara·Vulcano / Thor Ultra only        | AWS EFA-enabled instances only (Nitro)           | Runs on legacy NICs: CX-5/6/7, BlueField, EFA, Thor 1/2 |
| **Wire format**            | Open, OCP MRC 1.0 spec                      | Closed, AWS-proprietary                           | Open                  |
| **Spraying granularity**   | Per-packet                                      | Per-packet                                       | Per-chunk for UC/RC, per-packet for UD/AF_XDP   |
| **Ordering model**         | OOO packet delivery, OOO message delivery, `WRITE_WITH_IMM` enforces order| OOO packet delivery, OOO message delivery | OOO packet/chunk delivery, In-order/OOO message delivery are both supported |
| **Verb surface**           | `libibverbs`, `WRITE` + `WRITE_WITH_IMM` only                 | `libfabric`; `SEND/RECV` + `WRITE` + `READ`, no `ATOMIC` | All verbs the underlying NIC exposes (incl. `WRITE`, `READ`, `ATOMIC`, `SEND/RECV`) |
| **Congestion control**     | UET NSCC (ECN + RTT, window-based)              | AWS-proprietary, Cubic-like, designed for VPC fabric | RTT-based, pluggable |
| **Loss recovery**          | SACK + packet trimming, in-NIC                  | Selective retransmit, in-NIC    | SACK + selective retransmit, in software           |
| **PFC**                    | Off (lossy by design)                           | Off (lossy by design, VPC fabric)                 | Off (lossy by design)                              |
| **Packet spray**         | EV → ECMP hash, Structed EV, SRv6 uSID source-route      | Multi-path over AWS VPC (hash-based, fabric-managed) | EV → ECMP hash              |
| **Path selection**         | Programmable      | Unprogrammable | Programmable              |
| **Topology assumption**    | Co-designed with multi-plane port-breakout fabric | Co-designed with AWS VPC / Nitro fabric           | Works on whatever fabric you already have (single-plane, rail, multi-plane) |
| **Time to ship a new idea** | New silicon tape-out + spec revision + limited programmability          | New Nitro firmware/silicon under AWS control     | Code change                                        |
| **Openness / portability** | Open spec (OCP), multiple vendors                | AWS-only, not portable off AWS                    | Open, runs on any commodity RDMA/Ethernet NIC      |
| **Observation**  | Limited by hardware interface       | Limited by hardware interface, AWS-controlled telemetry only   | Highly observable in software    |


## References

[^1]: Ultra Ethernet Consortium. *Ultra Ethernet Specification v1.0.* 2025. <https://ultraethernet.org/wp-content/uploads/sites/20/2025/06/UE-Specification-6.11.25.pdf>
[^2]: DeepSeek-AI. *DeepEP — An efficient expert-parallel communication library.* GitHub, 2025–2026. <https://github.com/deepseek-ai/DeepEP>
[^3]: Y. Zhou, Z. Chen, Z. Mao, C. Lao, S. Yang, P. G. Kannan, J. Gao, Y. Zhao, Y. Wu, K. You, F. Ren, Z. Xu, C. Raiciu, I. Stoica. *UCCL-Tran: An Extensible Software Transport Layer for Machine Learning Workloads.* USENIX OSDI, 2026. <https://arxiv.org/pdf/2504.17307>
[^4]: NVIDIA. *ConnectX-8 SuperNIC.* Hot Chips 2025. <https://hc2025.hotchips.org/assets/program/conference/day1/CX8%20HotChips%20Aug25v2.pdf>
[^5]: L. Shalev, H. Ayoub, N. Bshara, E. Sabbag. *A Cloud-Optimized Transport Protocol for Elastic and Scalable HPC.* IEEE Micro, vol. 40, no. 6, 2020. <https://ieeexplore.ieee.org/document/9189994>
