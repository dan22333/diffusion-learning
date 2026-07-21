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

## Cross-cutting: Production-grade code (frontier-lab discipline)

> **Principle: every line is written as if it ships at a frontier lab.** No notebooks-as-source, no magic numbers, no "I'll clean it up later." The toy batch and the 50k run share the **same codebase** — only the config and the hardware change. This is a deliberate skill: hireable ML engineers write *runs that survive a preemption and are reproducible by someone else*, not scripts.

**Repo layout** — one installable package, configs separate from code:
```
diffusion/
  src/diffusion/
    models/        # UNet, EDM preconditioning, EMA
    diffusion/     # schedules, forward/reverse, samplers (ddpm/ddim/edm/fm)
    data/          # datasets, transforms, dataloaders
    train/         # training loop, optimizer, checkpointing
    eval/          # FID/IS/… metrics, sample grids
    utils/         # seeding, logging, distributed, profiling
  configs/         # typed configs (dataclass/Hydra/OmegaConf), one per experiment
  tests/           # unit tests (schedules, shapes, forward/reverse)
  scripts/         # train.py / sample.py / eval.py entrypoints
```

**Config & reproducibility:** typed configs — *no hardcoded hyperparameters*. Global seed control; log the full resolved config + git SHA with every run; deterministic where it matters.

**Experiment tracking & checkpointing:** W&B (or TensorBoard) from run 1 — loss, LR, grad-norm, samples/sec, GPU-util, sample grids. Checkpoint **and resume** (model, optimizer, EMA, step, RNG state). A run must survive a preemption.

**Code hygiene:** type hints, docstrings, `ruff` + `black`, a pre-commit hook, and unit tests for the diffusion math (a wrong noise schedule is a *silent* bug FID won't localize for you).

### The "right checks" — how a lab actually picks hyperparameters
> **These matter on the full 50k run (Phase 0.5+), not the toy batch.** The toy batch only answers "is the code correct?" — you can't meaningfully tune an LR against 8 memorized images. These answer "is the run configured well?" and only make sense on real data:

- **Batch size** — grow it until GPU memory is full or samples/sec stops improving; that's "big enough." Use **gradient accumulation** to hit a large *effective* batch when memory caps you.
- **Learning rate** — don't guess: run an **LR range test** (sweep LR, watch loss). Apply the **linear scaling rule** (LR ∝ effective batch size) + **warmup** + a schedule (cosine/constant). LR and batch size are coupled — never tune one blind to the other.
- **Optimizer & regularization** — AdamW, tuned betas (diffusion often uses β₂≈0.999), **weight decay**, **gradient clipping** (guards against loss spikes).
- **EMA of weights** — *non-negotiable for diffusion.* Samples come from the EMA copy, not the raw weights; getting this wrong tanks FID even with perfect training.
- **Mixed precision** — bf16/fp16 AMP: the single biggest throughput + memory win. Watch for NaN/overflow.
- **Health monitoring** — log grad-norm, param-norm, loss scale; **NaN/inf guards**; alert on divergence. A frontier run is *observable*.

> **Where this lives:** production **structure** (layout, configs, tests, tracking) starts at **Phase 0−** on the Mac. The **tuning checks** above become real at **Phase 0.5**, the first full-data run on a GPU.

---

## The phases (GCP)

### Phase 0− — Build the codebase & prove correctness *(LOCAL — on your Mac)*
No cloud, no GPU. Stand up the production repo layout above and write the diffusion code, then prove it **works** — not that it **learns**.
1. Implement the core: noise schedule, forward (add-noise), a reverse/denoise step, a small UNet, the training loop.
2. **Unit-test the math:** forward/reverse tensor shapes, schedule endpoints (σ_min/σ_max), that q(xₜ|x₀) has the right mean/variance.
3. **Overfit one batch (~8 images) to ~0 loss** on CPU/MPS. It will *memorize* — sample and you get those 8 images back. That is success: it proves the pipeline can learn *something*. It will **not** generalize — generalization needs the full 50k + a real GPU (Phase 0.5).

**Deliverable:** a clean, tested repo that overfits one batch locally — ready to move to the VM *unchanged* (only config + device change). This is your "play locally first" stage.

### Phase 0 — Move to the GPU & learn the systems layer *(first cloud step)*
Take the **exact same repo** from Phase 0− and run it on a cheap single-GPU **GCE VM** (L4 or T4). Raw PyTorch, no Accelerate yet — you want to *see* everything. Nothing about the code changes; only config + device.
1. **Re-run overfit-one-batch on the GPU** — confirm the port is clean and the loss still collapses.
2. Instrument: watch `nvidia-smi`/`nvitop`, run the PyTorch profiler. Deliberately shrink batch size → *watch utilization drop* → fix it. Starve the dataloader (`num_workers=0`) → watch the GPU idle. **Answer your infra questions by experiment.** This is the muscle you can't build on the Mac (no CUDA, no `nvidia-smi`).

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
- *Sweeps here:* a few configs → a plain loop is fine. Many configs → do it properly in Phase 2.5.

### Phase 2.5 — Hyperparameter sweeps (a search problem, not a scaling problem)
Distillation gives you real knobs worth tuning (step count, LR, loss weights, EMA), so this is the natural place to learn sweeps as a *first-class skill*. **Key distinction to internalize:** a sweep is **many runs of the same model with different configs to find the best** — orthogonal to multi-node, which is *one* run split for speed/size. (Each trial in a sweep might itself be single- or multi-GPU.)

1. **Start dumb:** a plain Python `for` loop over configs, sequential. Feel why it's slow and why you need scheduling.
2. **Adopt a sweep tool** — pick one and learn its model:
   - **W&B Sweeps** — simplest, great dashboards; grid/random/Bayesian. Best default.
   - **Optuna** — Pythonic, strong Bayesian/TPE search, pruning.
   - **Ray Tune** — parallel trial scheduling + early-stopping (ASHA/PBT); scales trials across a cluster.
   - **Vertex HP Tuning** — managed, if you're already on Vertex.
3. **Learn the ideas that matter more than the tool:** search space design, random vs Bayesian vs grid, **early-stopping/pruning** (ASHA — kill bad trials early), and how trials get scheduled in parallel.
- *Composes with everything:* the sweep tool schedules trials; each trial trains via plain torch / Accelerate. Sweeps ≠ parallelism — they sit *above* it.
- *This is the same "compare many configs" muscle you already used* in the Phase 0.5 four-way race and 0.75 metrics table — now formalized.

### Phase 3 — Quantization + deploy
Quantize the distilled model, then serve it. **Diffusion quantization is harder than LLM quantization** (error accumulates across denoising steps; activation ranges swing across timesteps; artifacts are *visible*) — so we do it empirically, watching it break:
1. **Weight-only int8/fp8** — the safe baseline; should just work.
2. **Push low-bit / activation quantization** — *expect* artifacts. Observe the timestep-varying-activation problem firsthand → understand why timestep-aware methods (Q-Diffusion, PTQD, SVDQuant 4-bit) exist. This "quantization that fights back" is the real lesson.
- Reminder on ordering: distillation (fewer steps) is the *big* diffusion inference win; quantization is the smaller, riskier secondary squeeze — which is why it comes after.
- Then build **one custom inference container** (FastAPI / vLLM / Triton), serve on **Cloud Run or Vertex Endpoint** — no K8s needed. Measure latency/quality live.

### Phase 4 — Distributed training (the employability multiplier)
> **4a and 4b are genuinely different, not the same thing at bigger scale.** 4a = GPUs in *one box* talking over NVLink (no networking). 4b = *multiple machines* talking over the network via NCCL (rendezvous, slower interconnect, coordination) — this is the real "multi-node" jump.

- **4a — single node, multiple GPUs** (`a3-highgpu-8g`, up to 8 GPUs). Do **DDP by hand once**: `torchrun --nproc_per_node=8`, raw `DistributedDataParallel` — learn ranks, `world_size`, NCCL. Then FSDP and DeepSpeed ZeRO on the same box. **No orchestrator needed** — one machine, GPUs over NVLink.
- **4b — multi-node (multiple machines).** *Then* adopt **Accelerate** (same script → DDP/FSDP/DeepSpeed via one config). Now you need something to *provision the machines + launch processes across them*. Pick **one** orchestrator: **Vertex** (managed, easiest — submit a multi-replica job) or **Ray on GCE** (`ray up` provisions a cluster of GCE VMs, Python-native, no K8s). K8s only if joining a shared-fleet org.
  - *What decides what:* **model size → the parallelism** (fits 1 GPU = DDP; too big = FSDP/ZeRO; enormous = +tensor/pipeline). **Org context → the orchestrator** (academia/HPC = Slurm; cloud startup = Vertex/managed; big shared fleet = K8s±Ray). Not the same axis.

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

### The launcher trio: torchrun vs Accelerate vs Ray Train
All three form the process group (Layer 3), but at different heights:
- **torchrun** — raw launcher: spawns 1 process/GPU, sets `RANK`/`LOCAL_RANK`/`WORLD_SIZE`, sets up rendezvous. You write the DDP/FSDP wrapping. *Assumes machines already exist.* Most transparent — use it by hand in 4a.
- **Accelerate** — a convenience wrapper *over* torchrun: one config switches your script single-GPU ↔ DDP ↔ FSDP ↔ DeepSpeed with no code change. *Still assumes machines exist; does not sweep.*
- **Ray Train** — launcher **+ its own cluster manager**: coordinates workers across a Ray cluster with fault tolerance/elasticity, integrated with Ray Data/Tune/Serve. *Can also provision the machines.*

### VM ↔ Vertex dev loop
- **VM** = interactive: edit, `python train.py`, watch `nvitop` live, re-run in seconds. The workshop.
- **Vertex** = submit a job → cold-provision (minutes) → runs elsewhere → you watch logs/metrics live but **can't interactively poke** (no persistent shell; "rerun" = cancel + resubmit). The factory.
- Standard workflow: **prototype on VM/notebook → submit the same code as a Vertex job to scale.** A notebook runs on *one* machine, so it maxes at that box's GPUs (≤8); beyond that the notebook becomes a *launcher* that submits to a cluster.

### K8s, Ray, Kubeflow — who sits where (the part that confuses everyone)
- **K8s = infrastructure manager (Layer 2):** "run these containers, restart crashed ones, share machines across teams, quotas/RBAC/networking." Knows nothing about your Python.
- **Ray = distributed-application engine (Layer 3+):** "run this Python program across the cluster" — actors, elastic fault-tolerant training, Train/Tune/Data/Serve libraries. K8s alone can't do elastic training or coordinate a Python compute graph — **that's what Ray adds on top.**
- **Ray-on-VMs** (`ray up`) vs **Ray-on-K8s** (KubeRay): same Ray, different provisioner. Use Ray-on-VMs when you have no platform and want a self-contained cluster (simplest, our choice). Use Ray-on-K8s when the org already standardized on K8s and wants Ray to live in that one shared platform (same quotas/monitoring/ops) instead of a parallel path.
- **Kubeflow = an ML toolkit *for* K8s** — a *suite of separate components*: Training Operator (`PyTorchJob` — distributed training), Katib (tuning), KServe (serving), Pipelines (DAGs). Contrast with Ray, which offers Train/Tune/Data/Serve as *one unified Python library* rather than assembled K8s CRDs.

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
**Phase 0− (local).** On your Mac, stand up the production repo layout, write the diffusion core + unit tests, and **overfit one batch to ~0 loss on CPU/MPS** — proving the code is correct before spending a cent on cloud. Then **Phase 0**: move that same repo to a cheap L4/T4 **GCE VM**, wire up `nvitop` + the profiler, and learn to read the GPU. Everything else builds on those two muscles — correct code, then systems literacy.
