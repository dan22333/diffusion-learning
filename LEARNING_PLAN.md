# Learning Plan: Diffusion + Training + Distributed Systems (→ Employable)

> **Goal:** understand the full stack end-to-end — training → diffusion → distributed training → the frameworks/orchestration around it — well enough to be **employable**. One anchor project (a diffusion world model), attacked in phases.

---

## What "employable" means here (the target skill set)

By the end you can walk into a job and:
- **Provision GPU compute on GCP** and get a training loop running from scratch.
- **Read a GPU like a dashboard** — know if you're compute-bound, dataloader-bound, or memory-bound, and fix it.
- **Explain + implement diffusion** from first principles (train + sample).
- **Scale a training job** across GPUs — DDP, then FSDP/DeepSpeed — and know *why* each exists.
- **Use the real frameworks**: PyTorch, Accelerate, torchrun, and one orchestrator (Ray or Vertex).
- **Take a model to deployment**: distill → quantize → serve in a container.

That combination — model understanding + systems literacy — is the rare, hireable profile.

---

## The dependency order (why this sequence)

| Topic | Layer | When |
|---|---|---|
| **Diffusion** | Foundation | First |
| **World model (Diamond)** | Application of diffusion | The anchor |
| **GPU systems literacy** | Cross-cutting lens | Day one, every phase |
| **Distributed training** (DDP/FSDP/DeepSpeed) | Scaling tool | Only once a model doesn't fit / trains too slow |
| **Distillation** | Post-training | After you have a trained model |
| **Quantization** | Deployment | Last — compress the best, already-fast model |

You **can't** distill or quantize first — they operate *on* a trained model. So: diffusion first.

---

## The anchor project: DIAMOND

**DIAMOND** — https://github.com/eloialonso/diamond — a diffusion world model.
Diffusion-based, clean public code, single-GPU trainable (Atari), and *fun* (playable). Real-time interaction makes slow sampling a genuine problem → distillation & quantization become things you actually need, not academic exercises. Match its existing conventions (config, raw PyTorch) rather than fighting them.

*(GameMatrix / MiniWM / LongBot = great **second** project once Diamond is understood.)*

---

## Cross-cutting: GPU systems literacy (from day one)

**Mental model:**
- **GPU-Util** (`nvidia-smi`) = % of recent time ≥1 kernel ran. Rough "is it doing anything" — can read 100% and still be inefficient.
- **Real question: compute-bound or stalled?** Stalls come from: small batches, slow dataloader, CPU↔GPU sync points, or memory-bandwidth limits.
- **Batch size** = main utilization knob. Too small → GPU idles between batches. Grow it until you fill memory or stop gaining samples/sec — that's "big enough."

**Tools:** `nvidia-smi`/`nvitop` (glance) → `watch -n1 nvidia-smi` (spot idle gaps) → **PyTorch Profiler + TensorBoard** (the real tool: dataloader-bound vs compute-bound) → DCGM / GCP Cloud Monitoring (over time).

---

## The phases (GCP)

### Phase 0 — Prove the pipeline on a toy model *(START HERE)*
Cheap single-GPU **GCE VM** (L4 or T4). Raw PyTorch, no Accelerate yet — you want to *see* everything.
1. **Overfit one batch to ~zero loss.** If it doesn't, the pipeline is broken. Fastest bug-catcher in ML.
2. Instrument: watch `nvidia-smi`, run the profiler. Deliberately shrink batch size → *watch utilization drop* → fix it. Starve the dataloader (`num_workers=0`) → watch the GPU idle. **Answer your infra questions by experiment.**

### Phase 0.5 — Diffusion fundamentals
Same cheap VM: tiny DDPM on MNIST. Overfit, understand forward/reverse process, noise schedule, why sampling is multi-step (DDPM vs DDIM).

### Phase 1 — Diamond end-to-end (train → play)
Get it training, get it sampling, **play** the world model. Read the code until the sampler → rollout loop is clear. Move to an **A100** only now. Output: a real trained checkpoint.

### Phase 2 — Distillation
The multi-step sampler is the bottleneck. Progressive distillation → consistency models → DMD, applied to *your* checkpoint. Cut sampling from N steps to a few. Feel the speedup in the playable game.
- *Sweeps here:* a few configs → a plain loop is fine. Many configs → **W&B Sweeps** (simplest) or Ray Tune (parallel scheduling + early-stopping).

### Phase 3 — Quantization + deploy
Quantize the distilled model, then serve it. **Diffusion quantization is harder than LLM quantization** (error accumulates across denoising steps; activation ranges swing across timesteps; artifacts are *visible*) — so we do it empirically, watching it break:
1. **Weight-only int8/fp8** — the safe baseline; should just work.
2. **Push low-bit / activation quantization** — *expect* artifacts. Observe the timestep-varying-activation problem firsthand → understand why timestep-aware methods (Q-Diffusion, PTQD, SVDQuant 4-bit) exist. This "quantization that fights back" is the real lesson.
- Reminder on ordering: distillation (fewer steps) is the *big* diffusion inference win; quantization is the smaller, riskier secondary squeeze — which is why it comes after.
- Then build **one custom inference container** (FastAPI / vLLM / Triton), serve on **Cloud Run or Vertex Endpoint** — no K8s needed. Measure latency/quality live.

### Phase 4 — Distributed training (the employability multiplier)
- **4a — one node, multiple GPUs** (`a3-highgpu-8g`). Do **DDP by hand once**: `torchrun --nproc_per_node=8`, raw `DistributedDataParallel` — learn ranks, `world_size`, NCCL. Then FSDP and DeepSpeed ZeRO on the same box. **No orchestrator needed.**
- **4b — multi-node.** *Then* adopt **Accelerate** (same script → DDP/FSDP/DeepSpeed via one config). Pick **one** orchestrator: **Vertex** (managed, easiest) or **Ray on GCE** (`ray up`, Python-native, no K8s).

---

## Tooling, orchestration & deployment (reference)

### The stack (what sits on what)
```
PyTorch DDP / FSDP / DeepSpeed  + NCCL     ← does the ACTUAL distributed training
Launcher: torchrun / Accelerate / Ray Train ← starts 1 process/GPU, forms the group
Cluster scheduler:  Slurm  OR  K8s          ← owns machines, decides what runs where (PEERS, pick one)
Hardware: GPU nodes + network (NVLink/net)
```
None of the orchestrators *do* the training — **PyTorch + NCCL does.** They allocate machines and launch processes.

### Accelerate — no in Phase 0, yes in Phase 4
Raw PyTorch first (see the mechanics). Adopt Accelerate at multi-node: same script runs single-GPU → DDP → FSDP → DeepSpeed by changing **one config**. It parallelizes **one run**; it does *not* do sweeps.

### The 4 parallelisms
- **Data parallel (DDP):** full model **copied** per GPU, different data. Limit = model must fit on 1 GPU.
- **Sharding (FSDP / ZeRO):** one model **split by parameter** — raises the model-size ceiling.
- **Tensor parallel:** split individual **layers/matrices**.
- **Pipeline parallel:** different **layers** on different GPUs.
- **Combined at scale:** FSDP `HYBRID_SHARD` (shard within a node over NVLink, replicate across nodes) and 3D parallelism (data × tensor × pipeline).

### How many GPUs? No framework cap — thousands in production.
Real limits: **DDP** = model must fit on one GPU (memory, not count); **FSDP/DeepSpeed** = communication bandwidth. For you, the limit is **GCP quota + budget**: 1 → 8 (one node) → ~16 (two nodes). File a quota-increase request.

### Sweeps
Accelerate can't sweep. A loop runs configs **sequentially**; parallel scheduling + early-stop + aggregation = **W&B Sweeps / Ray Tune / Optuna / Vertex HP Tuning**. They compose with Accelerate (sweep tool schedules trials; each trial trains via Accelerate/torch).

### What is GCE?
**Google Compute Engine** = raw GCP VMs. Create instance (pick GPU) → SSH in → train → **stop** (halts compute billing, still pay disk) or **delete** (removes all). Most transparent; best for learning. Contrast: **Vertex AI** = managed jobs, **GKE** = managed Kubernetes.

### Training vs inference → when you need K8s
The real axis is **predictable bounded job vs variable always-on service**:
- **Training** = runs to completion, resources known up front → **GCE VM, no K8s.** (K8s only enters at 100s-of-GPUs / shared clusters, for fault-tolerance & scheduling — not traffic autoscaling.)
- **Inference** = long-running service, variable traffic, must stay up → K8s's whole reason to exist. But you can also serve on **Cloud Run / Vertex Endpoint** without K8s. Inference = a **custom container** (model + serving code).

### Slurm vs K8s vs Ray
- **Slurm** and **K8s** are **peer** cluster schedulers from different worlds (HPC/academia vs cloud). Pick one — you don't stack Slurm on K8s.
- You're in **K8s-world (GCP)** → skip Slurm unless you head to HPC/academia.
- **Ray** is the thing that *does* run "on top of" — Ray-on-K8s (KubeRay), Ray-on-Slurm, or Ray-on-plain-VMs.

---

## Next step
**Phase 0.** Spin up a cheap L4/T4 **GCE VM**, get an overfit-one-batch loop running in raw PyTorch, wire up `nvitop` + the profiler. Everything else builds on that muscle. Recommended over Vertex for Phase 0 because you *see* the systems layer.
