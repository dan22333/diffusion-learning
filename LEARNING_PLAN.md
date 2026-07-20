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

### Phase 0.5 — Diffusion fundamentals: build all four, compare head-to-head
Same cheap VM, same model + dataset (**CIFAR-10**, 32×32 — real natural images so quality gaps are visible, and *the* standard diffusion benchmark). Overfit one batch first, understand forward/reverse process, noise schedule, why sampling is multi-step. Then **implement all four formulations and race them on identical conditions.**

> **Why CIFAR-10, not MNIST or ImageNet:** MNIST digits are too easy — all four samplers look identical and FID is meaningless. Full ImageNet from scratch is a multi-week, multi-GPU research job that defeats the fast-iteration purpose of this phase. CIFAR-10 is the sweet spot: hours-to-a-day per model on the cheap VM (~16 runs across methods × step-budgets = a weekend), genuinely good-looking natural images, and **published FID targets to validate your implementations against** (EDM ≈ 1.9, DDPM ≈ 3.2). ImageNet is a Phase 1 stretch goal, where the A100 budget exists.

The four fall on two axes: **DDPM / DDIM / EDM** are the score/noise-prediction family (differ in sampling ODE/SDE + parameterization); **flow matching** learns a velocity field along straight-ish paths (what SD3/Flux use).

| Method | Core idea | Steps to good sample | Notes |
|---|---|---|---|
| **DDPM** | Reverse a fixed Markov noising chain; net predicts ε, small stochastic steps | ~1000 | The slow foundational baseline |
| **DDIM** | Same trained net, deterministic non-Markov ODE → skip timesteps | ~20–50 | Drop-in on DDPM weights, no retraining; deterministic (enables interpolation) |
| **EDM** (Karras 2022) | Continuous σ-space, preconditioning, 2nd-order Heun sampler | ~10–35 | SOTA quality-per-step. **This is what Diamond uses** → master it for Phase 1 |
| **Flow matching** | Velocity field along near-straight paths; integrate an ODE | ~10–30 (→1–4 rectified) | Simpler objective, straighter paths; the modern default |

**The deliverable is a comparison, not four separate models.** For each method, log and plot on shared axes:
- **Steps vs quality** (the money plot — quality on y, NFE/step-count on x, one curve per method)
- **Wall-clock runtime** per sample at matched quality (each step = one net forward → NFE is the real cost)
- **Deterministic vs stochastic** behavior (fix the seed; DDIM/EDM-Euler/FM are deterministic, DDPM/EDM-churn are not)
- Sample grids at 1000 / 100 / 50 / 20 / 10 / 4 steps so you *see* where each degrades

> **Validate against published numbers:** because CIFAR-10 is the standard benchmark, every one of your implementations has a known-good FID target (EDM ~1.9, DDPM ~3.2, DDIM/FM in between depending on steps). If yours is wildly off, the implementation is buggy — same "overfit one batch" spirit, applied to the whole model.

> **Diamond alignment:** Diamond's denoiser + `diffusion_sampler.py` is EDM (Karras σ-schedule with `rho`, `c_skip`/`c_in`/`sigma_data` preconditioning, Heun order-2, `s_churn`). Reproducing EDM here on CIFAR-10 is a direct on-ramp to reading Diamond's code in Phase 1.

### Phase 0.75 — Measuring quality: the metrics, run the standard test suite
You just made "quality" claims in the comparison above — now learn to measure it properly instead of eyeballing. Run **all the well-known best-practice generative-quality metrics** on the four models' outputs, see how they (dis)agree, and understand what each actually captures.

- **FID** (Fréchet Inception Distance) — the standard headline number; distance between real/generated feature distributions. Learn its sharp edges: sample-count sensitivity, backbone dependence, that lower≠always-better.
- **Inception Score (IS)** — older, quality × diversity from a classifier; know why FID largely replaced it.
- **sFID, KID** — variants that fix specific FID weaknesses (spatial features; unbiased small-sample estimator).
- **Precision & Recall / Density & Coverage** — *split* fidelity from diversity (FID conflates them) — catches mode collapse a single scalar hides.
- **CLIP score** — text–image alignment (relevant later for conditional/text models; on plain CIFAR-10 you can still probe class-conditional alignment).
- **LPIPS / PSNR / SSIM** — perceptual & pixel similarity, for reconstruction/interpolation quality.

**Deliverable:** one metrics table — rows = {DDPM, DDIM, EDM, FM} × step-budget, columns = every metric above. Then the lesson: *where do the metrics disagree, and why?* (e.g. a model that scores great on FID but poor on Recall = mode collapse). This is exactly the muscle you'll reuse to judge distillation (Phase 2) and quantization (Phase 3) — both are "did quality drop, and by how much?" questions.
> Tooling: `torch-fidelity` / `clean-fid` (FID/KID/IS done right), `torchmetrics` (LPIPS/SSIM/PSNR), `prdc` (precision/recall/density/coverage).

### Phase 1 — Diamond end-to-end (train → play)
Get it training, get it sampling, **play** the world model. Read the code until the sampler → rollout loop is clear. Move to an **A100** only now. Output: a real trained checkpoint.
- *Optional stretch (scale the images up):* now that the A100 budget exists, re-run your best sampler from 0.5 (**EDM**, which is also what Diamond uses) on **class-conditional ImageNet-64** or a latent-diffusion setup — see how the fundamentals + metrics from 0.5/0.75 hold up at bigger resolution and harder data.

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
