# Learning Plan: Diffusion → World Models → Systems (→ Employable)

> **Companion file:** [`LEARNINGS.md`](LEARNINGS.md) — everything we've *concluded*, organized by topic (formulations, latent spaces, backbones, conditioning, rollout/drift, memory, distillation, metrics, the world-model landscape, systems, costs, and a corrections log). This file is the roadmap; that one is the knowledge base.

> **Goal:** understand the full stack end-to-end — diffusion → world models → distributed training → performance engineering → deployment — well enough to be **employable**. Two anchor projects: **Diamond** (to understand) and **MIRA** (to be current), attacked in a strict dependency order.

---

## What "employable" means here (the target skill set)

By the end you can walk into a job and:
- **Provision GPU compute on GCP** and get a training loop running from scratch.
- **Read a GPU like a dashboard** — know if you're compute-bound, memory-bandwidth-bound, or dataloader-bound, and fix it.
- **Explain + implement diffusion** from first principles, all four formulations (train + sample).
- **Build both backbones** — UNet *and* DiT — and say why each exists.
- **Explain what space diffusion runs in** — pixels vs VAE vs representation autoencoder — and the tradeoff.
- **Build an interactive world model** — action conditioning, causal/AR rollout, and why it drifts.
- **Use the real ecosystem**: PyTorch, HuggingFace (`diffusers`/`transformers`/Hub), Accelerate, torchrun.
- **Scale a training job** across GPUs — DDP, then FSDP/DeepSpeed — and know *why* each exists.
- **Make it fast**: distill few-step samplers, then **write a fused kernel** when the profiler says the framework is the problem.
- **Take a model to deployment**: quantize → serve in a container → measure latency live.

That combination — model understanding + systems literacy + performance engineering — is the rare, hireable profile.

---

## The dependency order (why this sequence)

| Topic | Layer | When |
|---|---|---|
| **Diffusion fundamentals** | Foundation | First — everything is built on it |
| **GPU systems literacy** | Cross-cutting lens | Day one, every phase |
| **Latents + DiT** | Modern architecture | Before any modern world model |
| **HuggingFace ecosystem** | How pretrained parts ship | Before you build on anything pretrained |
| **World model (Diamond)** | Application of diffusion | The teaching anchor |
| **Causal/AR rollout** | What makes a world model playable | After you can generate images |
| **Distributed training** | Scaling tool | Only once a model doesn't fit / trains too slow — that's MIRA |
| **World model (MIRA)** | The current stack | After latents + DiT + forcing + multi-GPU |
| **Distillation** | Post-training | After you have a trained model |
| **Kernels / perf engineering** | Below the framework | Once the profiler proves the framework is the limit |
| **Quantization + serving** | Deployment | Last — compress the best, already-fast model |

You **can't** distill or quantize first — they operate *on* a trained model. So: diffusion first.

---

## The two anchor projects

**1. DIAMOND** — https://github.com/eloialonso/diamond — the *teaching* anchor.
Pixel-space UNet + EDM. Clean public code, single-GPU trainable (Atari), and *playable*. It is also an **RL world model** — you train an agent *inside* it (`rew_end_model.py`, `actor_critic.py`), which no other project on this list does. Match its conventions (config, raw PyTorch) rather than fighting them.

**2. MIRA** — https://github.com/mira-wm/mira — the *currency* anchor.
Latent DiT + representation autoencoder + diffusion forcing. Apache-2.0 training code **and** a released dataset. This is where the field actually is in 2026.

*Read, don't reproduce:* **GameNGen** (the origin of the genre), **MultiGen** (external memory), **Matrix-Game 2.0/3.0** (few-step + retrieval memory), **LingBot-World** (the largest open one), **minWM** (a pluggable framework for all of the above).

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
    models/        # UNet, DiT, EDM preconditioning, EMA — selectable by config
    diffusion/     # schedules, forward/reverse, samplers (ddpm/ddim/edm/fm)
    data/          # datasets, transforms, dataloaders
    train/         # training loop, optimizer, checkpointing
    eval/          # FID/IS/… metrics, sample grids
    utils/         # seeding, logging, distributed, profiling
  configs/         # typed configs (dataclass/Hydra/OmegaConf), one per experiment
  tests/           # unit tests (schedules, shapes, forward/reverse)
  scripts/         # train.py / sample.py / eval.py entrypoints
```

> **Validation that this layout is right:** MIRA — a 5B-parameter frontier world model — ships exactly this shape: `configs/ src/mira/ tests/ scripts/`, Hydra configs, `torchrun`, `wandb`, `ruff` + `pytest`. Same structure, 1000× the compute.

**Config & reproducibility:** typed configs — *no hardcoded hyperparameters*. **Models are selected by config** (`model.name: unet | dit`), never hardcoded in the trainer. Global seed control; log the full resolved config + git SHA with every run; deterministic where it matters.

**Experiment tracking & checkpointing:** W&B (or TensorBoard) from run 1 — loss, LR, grad-norm, samples/sec, GPU-util, sample grids. Checkpoint **and resume** (model, optimizer, EMA, step, RNG state). A run must survive a preemption.

**Code hygiene:** type hints, docstrings, `ruff` + `black`, a pre-commit hook, and unit tests for the diffusion math (a wrong noise schedule is a *silent* bug FID won't localize for you).

### The "right checks" — how a lab actually picks hyperparameters
> **These matter on the full 50k run (Phase 3+), not the toy batch.** The toy batch only answers "is the code correct?" — you can't meaningfully tune an LR against 8 memorized images. These answer "is the run configured well?" and only make sense on real data:

- **Batch size** — grow it until GPU memory is full or samples/sec stops improving; that's "big enough." Use **gradient accumulation** to hit a large *effective* batch when memory caps you.
- **Learning rate** — don't guess: run an **LR range test** (sweep LR, watch loss). Apply the **linear scaling rule** (LR ∝ effective batch size) + **warmup** + a schedule (cosine/constant). LR and batch size are coupled — never tune one blind to the other.
- **Optimizer & regularization** — AdamW, tuned betas (diffusion often uses β₂≈0.999), **weight decay**, **gradient clipping** (guards against loss spikes).
- **EMA of weights** — *non-negotiable for diffusion.* Samples come from the EMA copy, not the raw weights; getting this wrong tanks FID even with perfect training.
- **Mixed precision** — bf16/fp16 AMP: the single biggest throughput + memory win. Watch for NaN/overflow.
- **Health monitoring** — log grad-norm, param-norm, loss scale; **NaN/inf guards**; alert on divergence. A frontier run is *observable*.

> **Where this lives:** production **structure** (layout, configs, tests, tracking) starts at **Phase 1** on the Mac. The **tuning checks** above become real at **Phase 3**, the first full-data run on a GPU.

---

# The path

**Numbering note:** phases are integers, in strict execution order. *(Renumbered 2026-07-28 from the old fractional scheme: 0−→1, 0→2, 0.5→3, 0.75→4, 0.9→5, 0.95→6, 1→8, 1.5→9, 2→12, 2.5→13, 3→15, 4→10.)*

| Part | Phases | Where |
|---|---|---|
| **A — Foundations** | **1** build the repo, prove correctness | Mac |
| **B — Diffusion fundamentals + GPU literacy** | **2** GPU + profiler · **3** DDPM/DDIM/EDM/rectified-flow race · **4** metrics suite | cheap L4/T4 |
| **C — Modern architecture** | **5** what space diffusion runs in · **6** DiT backbone | cheap VM |
| **D — The ecosystem** | **7** HuggingFace, CFG, ControlNet/LoRA, open weights | cheap VM |
| **E — World models** | **8** Diamond (train → play) **+ IRIS paired read/run** · **9** the forcing family | A100 |
| **F — Scale (needed *before* MIRA)** | **10** DDP → FSDP, single-node → multi-node, **+ one Vertex job** | 8×GPU → cluster |
| **G — Reproduce the current stack** | **11** MIRA end-to-end | multi-GPU |
| **H — Make it fast, then ship it** | **12** distillation · **13** sweeps · **14** kernels · **15** quantize + serve | — |
| **I — Frontier** | **16** memory & multiplayer (MultiGen, Matrix-Game, LingBot) · **17** latent actions (Genie read, **LAPO run**) | — |

---

## Datasets & budget (decide this once, up front)

**You train your own models through Phase 11. You use open weights only in Phase 7** (to learn the ecosystem and to have a strong reference), **and in Phase 16** (to *play* a SOTA world model without paying to train one). Everywhere else, training it yourself is the point — you cannot learn "how it trains and what goes wrong" from a checkpoint.

**Which dataset, and why it changes at Phase 5:**

| Phases | Dataset | Why |
|---|---|---|
| **1** (local) | 8 CIFAR-10 images | Overfit test only |
| **2–4** | **CIFAR-10, 32×32** | *The* diffusion benchmark → published FID targets to sanity-check against. Cheap: hours per model |
| **5–7** | **64×64 or 128×128** — Imagenette (10-class ImageNet subset), FFHQ-64, or downsampled ImageNet-64 | ⚠️ **CIFAR-10 breaks here.** A VAE at 8× downsampling turns 32×32 into a **4×4 latent** — patchify that and a DiT sees ~4 tokens. Meaningless. You need ≥64×64 so the latent is ≥8×8. Latent diffusion at 8×8×4 is *cheaper* per step than 32×32 pixels, so this costs you nothing |
| **8** | Atari (Diamond's own collector) | Comes with the repo |
| **9** | Any short-video set, or Diamond rollouts | You're measuring drift, not image quality |
| **11** | **Rocket Science** (MIRA's, 4k view-hours @720p) | Ships with the code; downsample hard |

**ImageNet from scratch is off the table** and should be stated plainly: class-conditional ImageNet-64 in the EDM paper took tens of GPUs for over a week; DiT-XL/2 on ImageNet-256 was 7M steps on 8×A100. That's thousands of GPU-hours — $2k–10k+. If you want ImageNet-quality samples, **load open weights in Phase 7**. Training it is not a learning exercise, it's a budget.

**Rough cost per phase** — GCP on-demand ≈ T4 $0.35/hr, L4 $0.70/hr, A100-40GB $3.67/hr, H100 $9–11.50/hr. **Use Spot for everything** (60–91% off) — your Phase 1 checkpoint/resume discipline is exactly what makes Spot safe:

| Phase | Hardware | Rough time | Rough cost (Spot) |
|---|---|---|---|
| **1** | Mac | — | $0 |
| **2** | T4/L4 | a few hours | **< $5** |
| **3** | L4 | 3 models × ~6–10 h *(DDIM is free — it reuses DDPM's weights)* | **$10–25** |
| **4** | L4 | metrics only, no training | **< $5** |
| **5** | L4 | pretrained VAE + your small RAE + 2 diffusion runs | **$20–40** |
| **6** | L4 | UNet-vs-DiT A/B + a patch-size sweep | **$25–50** |
| **7** | L4 | mostly inference; one tiny ControlNet | **$10–20** |
| **8** | A100 | Diamond Atari, days | **$100–250** |
| **9** | A100 | small causal video model + drift ablations | **$100–200** |
| **10** | 8×A100 (hours, not days) | DDP/FSDP scaling study | **$50–150** |
| **11** | multi-GPU | ⚠️ **the real spend** — codec + single-player WM, scaled down | **$500–2000+** |
| **12–15** | A100 | distillation + kernels + serving | **$150–400** |
| **16** | 1×A100 to *run* LingBot/Oasis | inference only | **< $20** |

> **Honest note on FID targets.** Published CIFAR-10 numbers (EDM ≈1.9, DDPM ≈3.2) come from near-full training runs — EDM's own was ~400 V100-hours *per model*. At this budget you'll land in the **FID 10–30** range, which is plenty to see real differences. So use published numbers as a **ceiling and a smell test**, not a pass/fail gate. What you actually validate is: (a) unit tests + the overfit-one-batch test prove correctness, and (b) the **relative ordering** is right — EDM should beat DDPM badly at low NFE, DDIM should track DDPM at 1000 steps and win at 50. If the *ordering* is wrong, you have a bug. If only the absolute number is high, you just trained less.

---

## Part A — Foundations

### Phase 1 — Build the codebase & prove correctness *(LOCAL — on your Mac)*
No cloud, no GPU. Stand up the production repo layout above and write the diffusion code, then prove it **works** — not that it **learns**.
1. Implement the core: noise schedule, forward (add-noise), a reverse/denoise step, a small UNet, the training loop.
2. **Unit-test the math:** forward/reverse tensor shapes, schedule endpoints (σ_min/σ_max), that q(xₜ|x₀) has the right mean/variance.
3. **Overfit one batch (~8 images) to ~0 loss** on CPU/MPS. It will *memorize* — sample and you get those 8 images back. That is success: it proves the pipeline can learn *something*. It will **not** generalize — generalization needs the full 50k + a real GPU (Phase 3).
4. **Cut the model seam now:** `build_model(cfg.model)` selected by `model.name`, so Phase 6's DiT is a config line and not a refactor. Never construct a model class directly in the trainer.

**Deliverable:** a clean, tested repo that overfits one batch locally — ready to move to the VM *unchanged* (only config + device change).

---

## Part B — Diffusion fundamentals + GPU literacy

> **One deliberate deviation from "fundamentals first":** Phase 2 (get on the GPU) comes *before* Phase 3 (the four-way race) for the simple reason that the race needs a GPU to run on. But the *deep* profiling work naturally lands during Phase 3 — that's when you first have real, long, varied runs worth profiling. Phase 2 is "get on the box and learn to look"; Phase 3 is where looking pays.

### Phase 2 — Move to the GPU & learn the systems layer *(first cloud step)*
Take the **exact same repo** from Phase 1 and run it on a cheap single-GPU **GCE VM** (L4 or T4). Raw PyTorch, no Accelerate yet — you want to *see* everything. Nothing about the code changes; only config + device.
1. **Re-run overfit-one-batch on the GPU** — confirm the port is clean and the loss still collapses.
2. Instrument: watch `nvidia-smi`/`nvitop`, run the PyTorch profiler. Deliberately shrink batch size → *watch utilization drop* → fix it. Starve the dataloader (`num_workers=0`) → watch the GPU idle. **Answer your infra questions by experiment.** This is the muscle you can't build on the Mac (no CUDA, no `nvidia-smi`).
3. Learn to name the three bottlenecks and tell them apart in a profiler trace: **compute-bound** (kernels back-to-back, high SM occupancy), **memory-bandwidth-bound** (kernels running but arithmetic intensity low — the usual state for diffusion at small batch), **dataloader/host-bound** (gaps between kernels). This vocabulary is what Phase 14 acts on.

### Phase 3 — Diffusion fundamentals: build all four, race them head-to-head
Same cheap VM, same model + dataset (**CIFAR-10**, 32×32 — real natural images so quality gaps are visible, and *the* standard diffusion benchmark). Overfit one batch first, understand forward/reverse process, noise schedule, why sampling is multi-step. Then **implement all four formulations and race them on identical conditions.**

> **Why CIFAR-10, not MNIST or ImageNet:** MNIST digits are too easy — all four samplers look identical and FID is meaningless. Full ImageNet from scratch is a multi-week, multi-GPU research job that defeats the fast-iteration purpose of this phase. CIFAR-10 is the sweet spot: hours-to-a-day per model on the cheap VM (~16 runs across methods × step-budgets = a weekend), genuinely good-looking natural images, and **published FID targets to validate your implementations against** (EDM ≈ 1.9, DDPM ≈ 3.2).

The four fall on two axes: **DDPM / DDIM / EDM** are the score/noise-prediction family (differ in sampling ODE/SDE + parameterization); **flow matching** learns a velocity field along straight-ish paths (what SD3/Flux use).

| Method | Core idea | Steps to good sample | Notes |
|---|---|---|---|
| **DDPM** | Reverse a fixed Markov noising chain; net predicts ε, small stochastic steps | ~1000 | The slow foundational baseline |
| **DDIM** | Same trained net, deterministic non-Markov ODE → skip timesteps | ~20–50 | Drop-in on DDPM weights, no retraining; deterministic (enables interpolation) |
| **EDM** (Karras 2022) | Continuous σ-space, preconditioning, 2nd-order Heun sampler | ~10–35 | SOTA quality-per-step. **This is what Diamond uses** → master it for Phase 8 |
| **Flow matching** | Velocity field along near-straight paths; integrate an ODE | ~10–30 | Simpler objective, straighter paths; the modern default |

> **Do flow matching as *rectified flow* specifically** — that's the variant SD3 and Flux ship. Three things people conflate under one heading:
> - **Rectified flow** = the *formulation*: straight-line interpolation between noise and data, network predicts **velocity**. Standard and alive — **learn it.**
> - **Logit-normal / shifted timestep sampling** = a *training detail*: which t you draw during training. SD3 found logit-normal beats uniform, plus a resolution-dependent shift. Cheap, materially affects quality — **learn it.**
> - **Reflow** = an *iterative procedure*: generate (noise, sample) pairs with your own model, retrain on them to straighten paths, repeat. This is what used to buy 1–2 steps. **Superseded — skip it.** DMD2 / sCM / MeanFlow do that job better, and reflow costs a full synthetic dataset per round.

**The deliverable is a comparison, not four separate models.** For each method, log and plot on shared axes:
- **Steps vs quality** (the money plot — quality on y, NFE/step-count on x, one curve per method)
- **Wall-clock runtime** per sample at matched quality (each step = one net forward → NFE is the real cost)
- **Deterministic vs stochastic** behavior (fix the seed; DDIM/EDM-Euler/FM are deterministic, DDPM/EDM-churn are not)
- Sample grids at 1000 / 100 / 50 / 20 / 10 / 4 steps so you *see* where each degrades

> **Validate by relative ordering, not by hitting published FID** (see the budget note above — those numbers cost ~400 V100-hours *per model*). Your gates: unit tests pass, overfit-one-batch still collapses, and **EDM ≫ DDPM at low NFE while DDIM ≈ DDPM at 1000 steps**. Wrong *ordering* = bug. Merely high absolute FID = you trained less, which is fine.
> **Cost note:** you train **three** models, not four — DDIM is a *sampler*, so it reuses DDPM's weights with no retraining. That's the cheapest lesson in the phase: the same checkpoint, a different ODE, 20× fewer steps.

> **Diamond alignment:** Diamond's denoiser + `diffusion_sampler.py` is EDM (Karras σ-schedule with `rho`, `c_skip`/`c_in`/`sigma_data` preconditioning, Heun order-2, `s_churn`). Reproducing EDM here is a direct on-ramp to reading Diamond's code in Phase 8.

### Phase 4 — Measuring quality: the metrics, run the standard test suite
You just made "quality" claims — now learn to measure it properly instead of eyeballing. Run **all the standard generative-quality metrics** on the four models' outputs, see how they (dis)agree, and understand what each actually captures.

- **FID** — the standard headline number; distance between real/generated feature distributions. Learn its sharp edges: sample-count sensitivity, backbone dependence, lower≠always-better.
- **Inception Score (IS)** — older, quality × diversity from a classifier; know why FID largely replaced it.
- **sFID, KID** — variants fixing specific FID weaknesses (spatial features; unbiased small-sample estimator).
- **Precision & Recall / Density & Coverage** — *split* fidelity from diversity (FID conflates them) — catches mode collapse a single scalar hides. **This is the instrument that exposes distillation damage in Phase 12.**
- **CLIP score** — text–image alignment (relevant once you hit conditional/text models in Phase 7).
- **LPIPS / PSNR / SSIM** — perceptual & pixel similarity, for reconstruction/interpolation quality. **These become your primary metrics in Phase 5**, where you're judging an autoencoder's reconstruction.

**Deliverable:** one metrics table — rows = {DDPM, DDIM, EDM, RF} × step-budget, columns = every metric above. Then the lesson: *where do the metrics disagree, and why?* (great FID + poor Recall = mode collapse). You will reuse this table in Phases 5, 6, 12 and 15 — every one of them is a "did quality drop, and by how much?" question.
> Tooling: `torch-fidelity` / `clean-fid` (FID/KID/IS done right), `torchmetrics` (LPIPS/SSIM/PSNR), `prdc` (precision/recall/density/coverage). *MIRA itself uses `torchmetrics`, `lpips` and `pytorch-fid` — you're learning the real toolchain.*

---

## Part C — Modern architecture (latents + transformers)

### Phase 5 — What space does diffusion run in? *(pixels → VAE → VQ → representation latents)*
Everything so far runs in **pixel space** (what Diamond uses). The question this phase answers is **what representation diffusion operates on** — the biggest architectural fork in the field, and a hard prerequisite for Phase 6, because attention cost makes pixel-space DiTs impractical.

> **Four ways to handle images — don't conflate them:**
> 1. **Pixel space** — Diamond. Maximum detail fidelity, maximum compute per pixel.
> 2. **Continuous latent / VAE** — Stable Diffusion, GameNGen, Matrix-Game's 3D causal VAE. Conv encoder+decoder trained *together for reconstruction*, aggressive down/upsampling.
> 3. **Discrete latent / VQ-VAE + transformer** — IRIS-style tokenizers. Quantized codebook tokens; loses detail.
> 4. **Representation latent / representation autoencoder (RAE)** — a **frozen self-supervised encoder** (DINOv2/v3, SigLIP2) with only a **ViT decoder** trained on top. MIRA.
>
> A VAE is *not* a discrete tokenizer: continuous latents lose far less detail than quantized tokens.

> **Why (4) is the new idea — and it's a strong result.** A VAE encoder is trained for *reconstruction*, so its latents are packed with high-frequency detail: easy to decode, hard to *predict*. A DINO encoder is trained for *semantics*, so its space is smoother and more linearly structured — a much easier target for a diffusion transformer. [RAE (ICLR 2026)](https://arxiv.org/pdf/2510.11690) shows a frozen DINOv2 + trained **ViT** decoder gives reconstructions **on par with or better than SD-VAE** while using **~6× fewer encoder GFLOPs and ~3× fewer decoder GFLOPs**. Note the architectural difference: VAE = conv with aggressive compression; RAE = ViT **without** spatial compression. Same insight as REPA (aligning DiT internals to DINO features), pushed to its conclusion: don't *align* to representations — **diffuse in them**. MIRA reports this swap "gave us the highest leap in modeling performance."

1. **Use a *pretrained* VAE first** (SD's) — encode CIFAR → latents, train your EDM diffusion *in latent space*, decode to compare. This isolates "diffusion-in-latents" from "training an autoencoder."
2. **Then build a small RAE** — frozen DINOv2 encoder + your own ViT decoder, trained with LPIPS + L1. Compare reconstruction (LPIPS/PSNR) *and* downstream diffusion quality (FID) against the VAE. This is the single highest-leverage experiment in Part C.
3. **Reuse the Phase 4 metrics table** to quantify pixel vs VAE vs RAE: quality vs speed vs compute.
4. *Optional stretch:* train your own small VAE to feel the reconstruction/rate tradeoff and why latent **scale-normalization** matters.

**Deliverable:** the same pipeline running in three spaces, with one comparison table. You can now articulate the choice separating Diamond (pixels) from Stable Diffusion (VAE) from MIRA (RAE).
> **Note:** Diamond deliberately stays in **pixel space** ("visual details matter" — arguing pixels beat *discrete* latents like IRIS on game-critical detail). For high-res CS:GO it avoids an autoencoder entirely via a **diffusion cascade** — a low-res (56×30) world model + a second diffusion model upsampling to 280×150. So this phase is a *fundamental you own*, not a Diamond prerequisite — but it **is** a MIRA prerequisite.

### Phase 6 — Swap the backbone: UNet → DiT *(the transformer half of modern diffusion)*
Every model you'd be hired to work on — SD3, Flux, Sora, Wan, Oasis, MIRA, Matrix-Game — is a **diffusion transformer**. This is **not** a one-line model swap; it's six concepts with no UNet analogue:

1. **Patchification** — patch size *p* becomes the primary compute↔quality knob. DiT's headline result: shrinking *p* improves FID at **fixed parameter count** (pure FLOPs). Sweep *p*, not just width/depth.
2. **Conditioning becomes a design choice** — in-context tokens vs cross-attention vs **adaLN-Zero** (adaLN-Zero won in the DiT paper); SD3's **MMDiT** is a fourth option (two-stream joint attention). Diamond's action conditioning is FiLM-style AdaGroupNorm; in a DiT you must *pick*. Note Diamond already uses adaLN-Zero's identity-init trick — `inner_model.py:42` zero-inits the output conv.
3. **No multi-scale hierarchy or skips** — the UNet's locality + multi-resolution bias is gone → slower convergence, more data-hungry. (U-ViT's long skips exist for this reason.)
4. **Positional encoding decides resolution generalization** — absolute sincos vs 2D/3D **RoPE**. Flux uses RoPE so it can sample at resolutions it wasn't trained at.
5. **Attention is O(N²) in tokens, N ∝ (res/p)²** — pixel-space DiTs are quadratically punished. *This is why DiT presupposes Phase 5, and why Diamond (56×30 pixels) has no reason to switch.*
6. **Scaling-law literacy** — the DiT paper's real contribution is a clean FLOPs→FID curve. Reproducing that plot is the lesson, and it doubles as GPU-utilization practice.

**On attention — yes, it's ordinary attention.** The *operator* is standard scaled-dot-product self-attention everywhere (FlashAttention in practice). Nobody invents exotic attention for diffusion. What varies is only the **mask** and the **token set** — that's the whole game, and seeing it this way collapses a dozen "architectures" into one:

| Setting | Mask | Tokens attended |
|---|---|---|
| Image DiT | bidirectional (none) | all patches of one image |
| Text-to-image | bidirectional | patches + text (cross-attn, or MMDiT joint) |
| Video DiT, cheap | bidirectional, **factorized** | spatial-within-frame, then temporal-across-frames (Latte, Open-Sora ST-DiT) |
| Video DiT, modern | bidirectional, **full 3D** | all space-time tokens at once (Wan, HunyuanVideo) — better, costlier |
| **Interactive world model** | **block-causal** + KV cache | causal across time chunks, bidirectional within a chunk |
| + retrieval memory | block-causal, joint | memory latents + past latents + noisy-current latents in one attention space (Matrix-Game 3.0) |

> Also note: in the original DiT, **conditioning does not go through attention at all** — adaLN modulates the norm layers. Attention is for *tokens*; adaLN is for *scalars/vectors* like timestep and class. This trips people up constantly.

**Do it in two steps:** (a) the *controlled A/B* — same EDM, same CIFAR-10, same Phase 4 metrics table, **one variable changed** — proves your implementation is correct; (b) the *real* DiT in Phase 5's latent space, which is the regime it was designed for.

**Deliverable:** `models/dit.py` selectable by config alongside `unet.py`, the Phase 4 metrics table re-run for both, and a FLOPs→FID scaling plot.

---

## Part D — The ecosystem

### Phase 7 — HuggingFace, conditioning, and the open-weights landscape
**Do people actually use HF for diffusion? It splits cleanly, and the split is the lesson:**

| If you are… | You use | Evidence |
|---|---|---|
| **Training from scratch** (research, novel architecture) | **raw PyTorch.** No `diffusers`. | MIRA's deps: `torch`, `einops`, `pydantic`, `hydra`, `wandb`, `torchmetrics`, `lpips`, `pytorch-fid`. **No diffusers, no transformers, no accelerate.** Diamond: same, raw PyTorch |
| **Building on a pretrained foundation model** | **HF everywhere** — `diffusers` + `transformers` + `accelerate` + Hub | Self-Forcing's `requirements.txt` pins `diffusers==0.31.0`, `transformers>=4.49`, `accelerate>=1.1.1`, `huggingface_hub` — because it builds on Wan2.1. Matrix-Game 2.0 and minWM init from Wan2.1/SkyReels, so same story |
| **Anyone touching weights or datasets** | **HF Hub**, universally | Even MIRA — no `diffusers`, but `huggingface_hub` + `datasets` to ship Rocket Science |

**So: yes, learn it — but as the *ecosystem*, not as your training framework.** You need `diffusers` to *consume* pretrained components (VAEs, DINO encoders, text encoders, DiT backbones), to read reference implementations, and because half the field's code assumes it. You do **not** want it as the abstraction you write your own research inside — that's why the plan stays raw-PyTorch for the parts you build.

**7.1 — `diffusers` mental model: three separable layers.** The whole library is *pipeline* (orchestration) / *scheduler* (the sampling math) / *model* (the network). Learn to bypass the top layer:
```python
# the four component types you will actually reach for
from diffusers import AutoencoderKL, UNet2DConditionModel, DiTTransformer2DModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, AutoModel, AutoImageProcessor

vae  = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")          # Phase 5: the VAE baseline
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
txt  = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")      # text encoder
t5   = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")                # SD3/Flux-style text encoder
dino = AutoModel.from_pretrained("facebook/dinov2-large")                  # Phase 5: the RAE encoder
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")         # its matching preprocessing
```
Key habits: `subfolder=` pulls one component out of a full pipeline repo; `torch_dtype=torch.bfloat16` and `.requires_grad_(False)` for anything frozen; a VAE needs its **scaling factor** applied (`latents * vae.config.scaling_factor`) or your diffusion sees the wrong variance — a classic silent bug; DINO needs *its* normalization, not SD's.

**The exercise that makes it click:** load SD's UNet, throw away the pipeline, and **run your own Phase 3 DDIM/EDM sampler against their weights.** If your sampler produces good images from someone else's network, both are correct. This is a cross-implementation unit test.

**7.2 — Build DiT yourself first, read HF's second.** You already built `models/dit.py` in Phase 6 — that was the point, because patchification, adaLN-Zero and RoPE cannot be learned by calling a pipeline. *Now* open `diffusers`' DiT/SD3 transformer and **diff it against yours**, line by line: where do they put the modulation, how do they handle the final layer, what do they do about attention masks and dtype. Every difference is either a bug in yours or a production concern you hadn't met. Build first, compare second.

**7.3 — Classifier-free guidance (CFG), properly.** The single most important conditioning technique in diffusion.
- **Training:** randomly replace the condition with a null/learned-empty embedding, ~10% of the time. One model, two behaviours.
- **Sampling:** `ε = ε_uncond + w·(ε_cond − ε_uncond)` — extrapolate *away* from the unconditional prediction.
- **What it costs:** two forward passes per step → **NFE doubles**. (That's why guidance distillation in Phase 12 is a free 2×.)
- **What `w` trades:** fidelity/prompt-adherence ↑, diversity ↓. **Don't take that on faith — measure it** with Phase 4's Precision/Recall across a `w` sweep and watch Recall fall. This is the cleanest demonstration in the whole plan that one scalar metric lies.
- Also learn **guidance intervals/schedules** (apply guidance only in a middle band of σ — often strictly better than a constant `w`).

**7.4 — ControlNet: adding a conditioning signal to a *frozen* model.** ⚠️ **This is not "guided inference" — do not conflate them:**

| | **Trained conditioning** (ControlNet family) | **Inference-time guidance** (CFG, classifier guidance) |
|---|---|---|
| Adds parameters? | **Yes** — a new branch you train | **No** |
| Needs training data? | Yes — (condition, image) pairs | No |
| Mechanism | New network injects features into the frozen base | Reweights/combines score predictions at sample time |
| When | Once, offline | Every sampling step |

Three canonical forms of the trained kind: **ControlNet** (clone the base encoder, **zero-init the joins** so it starts as a no-op, freeze the base), **T2I-Adapter** (lighter side network), **IP-Adapter** (decoupled cross-attention for image prompts). Learn ControlNet properly because **Matrix-Game 2.0 does exactly this pattern** — pretrained DiT, plus *action modules injected into each DiT block*. ControlNet is the cheap image-domain rehearsal for action conditioning.
> Note the zero-init trick recurs everywhere: ControlNet's joins, adaLN-**Zero**, and Diamond's `nn.init.zeros_(self.conv_out.weight)` (`inner_model.py:42`). Same idea each time — *start as identity/no-op so adding capacity can't hurt.*

**7.5 — LoRA / PEFT.** ⚠️ **Unrelated to CFG** — I listed them together earlier and that was misleading. LoRA is *parameter-efficient fine-tuning*: freeze the base, learn low-rank `ΔW = BA` on the attention projections. Learn it for three independent reasons: it's how the field ships fine-tunes; **LCM-LoRA** (Phase 12) made *distillation itself* a swappable adapter, which is a genuinely surprising result; and it's how you'd adapt a big pretrained model on your budget.

**7.6 — The open-weights landscape** — know what to reach for, and *why*, rather than memorizing:

| Need | Reach for |
|---|---|
| Image gen, UNet-era baseline | SD 1.5 / SDXL |
| Image gen, DiT + rectified flow | SD3.5, **Flux** |
| Video gen backbone to build on | **Wan 2.1/2.2** (1.3B is laptop-scale), HunyuanVideo, LTX-Video |
| VAE off the shelf | SD-VAE, or Wan's 3D causal VAE for video |
| Representation encoder | **DINOv2 / DINOv3**, SigLIP2 |
| Playable world model, open weights | **LingBot-World**, Oasis 500M |
| World model with training code + data | **MIRA** (Phase 11) |

**Deliverable:** a notebook-free script that loads a pretrained DiT from `diffusers`, runs *your own* Phase 3 sampler against it, sweeps CFG scale, and plots **FID *and* Recall vs `w`** (watch them diverge). Plus one **tiny ControlNet on your own Phase 6 DiT** — Imagenette + Canny edges as the conditioning signal. Training a ControlNet on SDXL is expensive; training one on your own small model is cheap and teaches the identical lesson: freeze the base, add a branch, zero-init the join.

---

## Part E — World models

### Phase 8 — Diamond end-to-end (train → play)
Get it training, get it sampling, **play** the world model. Read the code until the sampler → rollout loop is clear. Move to an **A100** only now. Output: a real trained checkpoint.

> **The two lineages — know which one you're in.**
> **(a) Pixel / UNet / EDM:** **Diamond** and **GameNGen**. Same family, different goals. Diamond is an **RL world model** you train an agent *inside* (hence `rew_end_model.py` + `actor_critic.py`, Atari-100k), from scratch and tiny, pixel-space, EDM+Heun. GameNGen is a **playable engine demo** — no agent trained inside it — fine-tuning **pretrained SD 1.4** in *VAE latents*, ~4-step DDIM, fighting drift with noise-augmented context. Diamond is open; GameNGen is not (unofficial reimplementations only).
> **(b) Latent / DiT / diffusion-forcing:** **Oasis**, **MIRA**, **Matrix-Game 2.0/3.0**, **minWM**, **LingBot-World**. Where the field is, and where Phases 5 → 6 → 9 → 11 take you.
> Diamond stays the teaching anchor because it is cheap, complete, single-GPU and *playable*. Lineage (b) is the employable one. Do (a) to understand, (b) to be current.

### What Diamond actually is: Dreamer's programme, executed with diffusion

Diamond is not "a playable demo" — it is an **RL world model**, and that framing explains the repo (`actor_critic.py`, `rew_end_model.py` — no video-generation world model ships those). The *goal* is Dreamer's: **train an agent entirely inside a learned world model.** Every *mechanism* differs.

| Model | State representation | Sequence model | Policy trains on | Params | Atari-100k HNS |
|---|---|---|---|---|---|
| **DreamerV3** | compact **recurrent latent** (RSSM — GRU-based, **not** a transformer) | GRU recurrence | **latents** — never needs pixels | 18M | 1.097 |
| **IRIS** | **discrete VQ image tokens** | autoregressive **transformer** | decoded frames | 30M | 1.046 |
| **STORM** | discrete latents, DreamerV3-style | **transformer** | latents | — | 1.266 |
| **DIAMOND** | **raw pixels**, no compression | **diffusion** (conv UNet + EDM) | rendered frames | **13M** | **1.46** ← SOTA |

> **The striking number is 13M beating 18M and 30M.** Not a scale win — a *representation* win, which is exactly the paper's title, *"Visual Details Matter."* The argument targets IRIS's **discrete** tokenizer: quantisation drops small but game-critical detail (a distant enemy, a bullet, a HUD digit), and **an agent cannot act on what the world model failed to render.**
> ⚠️ Common mis-statement to avoid: *"Diamond is DreamerV3 with a transformer instead of an RNN."* DreamerV3 is an **RSSM (recurrent)**; the transformer variants are **TWM** and **STORM**; and Diamond is neither — it is diffusion over pixels.

### Read *and run* IRIS alongside Diamond — the cleanest controlled comparison in this literature

[`eloialonso/iris`](https://github.com/eloialonso/iris) (*Transformers are Sample-Efficient World Models*, ICLR 2023 notable top-5%; Micheli, **Alonso**, **Fleuret**) ships **code and pretrained checkpoints**, so you can run it without training anything.

**Why the pairing is unusually valuable:** Diamond's thesis is an argument *against* IRIS, and **both come from the same lab, on the same benchmark, with the same code conventions** — Alonso and Fleuret authored both. The comparison is therefore genuinely **controlled**: *discrete tokens + transformer* vs *pixels + diffusion*, with team, benchmark and engineering held roughly constant. That almost never happens in this field.

It also closes a loop opened in Phase 5, which cites IRIS as the "discrete latent / VQ-VAE + transformer" option without ever having you look at it.

**Deliverable:** run IRIS from a checkpoint, run your Diamond checkpoint, and write the paragraph explaining **why Diamond went to pixels**. Being able to argue *both* sides is what separates having read the papers from having understood them.

### Know that you are only learning one of two paradigms

| Paradigm | Predicts | Examples | In this plan |
|---|---|---|---|
| **Generative / pixel-rendering** | the next **frame** | Sora, Genie, MIRA, Diamond, Matrix-Game | ✅ covered end-to-end |
| **Non-generative / latent prediction** | the next **embedding** — never renders | **V-JEPA 2** (Meta), **DreamerV3** | ⚠️ read only |

V-JEPA 2 learns physics from ~1M hours of video and transfers to robot control with ~62 h of robot data; DreamerV3 trains a policy by imagining rollouts in a latent world model. **Both open.** Don't reproduce either — but you should be able to say why the JEPA camp argues rendering pixels is wasted capacity, and why the generative camp pays that cost to get a *usable simulator* out of it. **Diamond is the bridge:** a generative world model serving Dreamer's purpose.

- **Read GameNGen's paper here**, alongside Diamond's, as a same-lineage contrast. It costs an evening and it's where the drift problem is first named.
- *Optional stretch:* re-run EDM on **class-conditional ImageNet-64**, or apply Phase 5's latent setup at higher resolution.
- *Optional capstone (CS:GO):* train the pixel-space CS:GO world model (`git checkout csgo`) on the public [CounterStrike_Deathmatch dataset](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch) (5.5M frames / 95h). The authors did it in ~12 days on one RTX 4090 (≈$100–200 rented). Mind the tens-to-hundreds of GB hdf5 download.

### Phase 9 — Causal / autoregressive rollout: **the forcing family** *(the image→world-model bridge)*
This is the concept that turns an image diffusion model into a playable world model, and it's missing from most curricula because it's neither a *formulation* topic (Phase 3) nor a *backbone* topic (Phase 6).

**The problem is exposure bias.** At train time you condition on *real* past frames; at inference you condition on *your own* generated frames, whose errors compound until the world melts. That compounding is **drift**.

**Yes — all five rows below are ways to prevent drift, and they do build on each other.** Read the table as a single argument that gets progressively less ad-hoc:

| Technique | What it does | What it fixes / gives | Cost |
|---|---|---|---|
| **Teacher forcing** | Condition on ground-truth past | Nothing — this *is* the problem. Simple + fully parallel training | Maximum train/test mismatch → drift |
| **Noise-augmented context** (GameNGen, 2024) | Add Gaussian noise to the conditioning frames during training; tell the model the level | **First** fix: the model learns to work from a *corrupted* history, so its own flawed outputs are in-distribution | Ad-hoc — one global corruption scheme, no theory |
| **Diffusion forcing** (NeurIPS 2024) | Give **every frame its own independent noise level** | Same insight, now **principled** (a variational bound on all subsequences). Unifies next-token prediction with full-sequence diffusion → **rollouts extend past the training horizon where baselines diverge**, variable-length generation, and new guidance schemes | Still never literally simulates inference |
| **Self-forcing** (NeurIPS 2025) | **Actually run the AR rollout during training** with KV caching, then a *holistic* distribution-matching loss over the whole sequence | Closes the train/test gap **directly** instead of approximating robustness. Real-time streaming: ~16 FPS on H100, ~10 FPS on a **single 4090**; beats CausVid at equal speed without over-saturation | Sequential → slower training steps; quality degrades past the ~5 s training horizon |
| **Causal forcing / ++** (ICML 2026) | "AR diffusion distillation done right" — causal ODE / consistency distillation + asymmetric DMD | Fixes Self-Forcing's remaining issues and **scales**; the current recipe | Needs a teacher model; most machinery |

> **Vocabulary, because these get conflated:** "causal / autoregressive diffusion" is the **category**. Diffusion forcing, self-forcing and causal forcing are **different members**, not synonyms. Diffusion forcing changes the **noising scheme**; self-forcing changes **what the model conditions on during training**. They're orthogonal — modern systems do **both**.

> **Diffusion forcing is not "just add noise" — the *independence* is the whole idea.** Ordinary video diffusion draws **one** σ and applies it to the whole clip. Diffusion forcing draws a **separate, independent σ per frame**. That sounds minor; it changes what the network learns:
> - Standard training covers a 1-D family: *(one noise level)*. Diffusion forcing covers the full **2-D grid of (frame position × noise level)** — including the asymmetric corner that matters: **clean-ish past, very noisy future.** That corner *is* causal generation, and standard training never visits it.
> - So at inference you can **choose the noise schedule across time**, not just across steps. Fully denoise the past for maximum sharpness, or hold the past at low-but-nonzero noise to stay robust to your own errors — a dial you simply don't have otherwise.
> - It also subsumes the special cases: all-σ-equal = full-sequence diffusion; past-at-σ=0, future-at-σ=max = next-frame prediction. One objective, both regimes, with a variational bound over all subsequences.
> - **Implementation-wise it really is small** — sample a σ vector of shape `[B, T]` instead of `[B]`, broadcast per frame, and feed each frame's σ into the conditioning. That's why it's the cheapest thing on this list to add to Diamond.

**How to actually measure rollout stability** *(you'll claim "less drift" — here's how to prove it):*
1. **Drift curve** — the headline plot. Roll out N steps from a real seed frame; compute FID (or LPIPS vs ground truth, if you have the true continuation) **per rollout position**, then plot quality against step index. A stable model's curve flattens; an unstable one bends down and then falls off a cliff. Run every forcing variant on the same seeds and overlay the curves.
2. **Latent/pixel statistics over time** — track mean, std and norm of the generated frames per step. Drift almost always shows up as a slow statistical march (contrast creeping up, saturation growing) *before* it's visible. This is your cheap early-warning signal.
3. **Survival time** — the blunt one: how many steps until a human says "that's broken." Report it, it's what users feel.
4. **Action-response fidelity** — press left, does the world go left? Controllability degrades before appearance does.
5. **Ground-truth-state comparison, if you have it** — MIRA's trick: they withheld the true physics state from training and used it **only for evaluation**, so they can ask "is the ball where real physics says it should be?" rather than "does it look plausible?" If your sim exposes state, do this; it's the strongest signal available.
6. **Return-to-viewpoint test** — a *different axis*, covered in Phase 16. Don't confuse it with drift.

> **Is causal forcing "the best"?** For *shipping a real-time model*, yes — it's the current state of the art and what `minWM` packages. But it needs a teacher and the most machinery, so it's the wrong place to *start*. **Learn two, skim one:** implement **diffusion forcing** (cheapest, and what MIRA uses in Phase 11), implement **self-forcing** (where the real insight lives, and it fits on a 4090), then read **causal forcing** to see what production hardening looks like.

> **Diamond is already halfway here — read the code before anything else.** `diamond/src/models/diffusion/denoiser.py:100-119` loops over the sequence and writes each *denoised prediction* back into the conditioning buffer (`all_obs[:, n+i] = denoised`), so later steps condition on the model's **own generations**. That is **self-rollout, not teacher forcing**. What it lacks vs Self-Forcing: a distribution-matching loss (it uses per-step MSE), a KV cache (it's a conv UNet — nothing to cache), and a few-step student. **Diffing those two training loops is the single most instructive exercise in this phase.**

> **Cheap upgrades you can apply to Diamond directly** — this is the phase where Phase 8's checkpoint becomes an experiment bench:
> 1. **Add noise-augmented context** (GameNGen's trick) — a handful of lines, and you can measure the drift curve before/after.
> 2. **Add per-frame independent noise levels** (diffusion forcing) — Diamond already samples one σ per step; sampling a σ *per context frame* is a small change to `apply_noise`/`compute_conditioners`.
> 3. **Swap per-step MSE for a distribution-matching loss** — the real jump, and much more work. This is where you'd stop and move to a DiT instead.
> Items 1–2 are genuinely cheap and give you a publishable-shaped ablation on a model you already trained.

**Deliverable:** your own small causal video diffusion model with per-frame noise levels + KV-cached streaming rollout, plus a **drift curve** (quality vs rollout length) comparing teacher-forcing / noise-augmented / diffusion-forcing / self-forcing on identical data.
> Reference code: [`diffusion-forcing-transformer`](https://github.com/kwsong0113/diffusion-forcing-transformer) (small, academic scale — the tractable one), [`Self-Forcing`](https://github.com/guandeh17/Self-Forcing) (single 4090; note it *does* use `diffusers`), [`Causal-Forcing`](https://github.com/thu-ml/Causal-Forcing).

---

## Part F — Scale *(moved here on purpose)*

### Phase 10 — Distributed training (the employability multiplier)
> **Why here and not at the end:** the plan's own rule is "distributed training only once a model doesn't fit or trains too slow." Phase 11 (MIRA) is the **first** thing on this path that genuinely requires multiple GPUs — MIRA's own repo launches with `torchrun`. Learning DDP/FSDP immediately before you need it means you learn it against a real constraint instead of a toy.

> **10a and 10b are genuinely different, not the same thing at bigger scale.** 10a = GPUs in *one box* over NVLink (no networking). 10b = *multiple machines* over the network via NCCL (rendezvous, slower interconnect, coordination) — the real "multi-node" jump.

> **You train a real FSDP model here — and it does *not* need to be a world model.** Take the **DiT from Phase 6** and scale it up (width/depth/patch size) until it genuinely will not fit on one GPU with a useful batch size. That single model carries the entire phase, and it's a far better teacher than a toy MLP because you already know what its loss curve should look like.

**The concrete progression, one model, five runs:**
1. **1 GPU baseline** — record samples/sec, peak memory, loss curve. This is your reference; everything is measured against it.
2. **DDP by hand** — `torchrun --nproc_per_node=8`, raw `DistributedDataParallel`. Learn `RANK`/`LOCAL_RANK`/`WORLD_SIZE`, `init_process_group`, `DistributedSampler` (and the classic bug: forgetting `set_epoch`, so every epoch shuffles identically), and which NCCL collective is actually running (`all_reduce` on gradients). Confirm loss matches the 1-GPU run at the same *effective* batch.
3. **Break DDP on purpose** — grow the model until it OOMs on one GPU. Now you have *felt* DDP's ceiling: it replicates the full model per GPU, so the model must fit on one. This is the motivation for step 4, and it's much stickier than being told.
4. **FSDP** — shard parameters, gradients and optimizer state. Learn the auto-wrap policy (wrap per transformer block), `ShardingStrategy` (`FULL_SHARD` vs `SHARD_GRAD_OP` vs `HYBRID_SHARD`), mixed precision, and **activation checkpointing**. The model that OOM'd now trains. Then **DeepSpeed ZeRO** on the same box for comparison — ZeRO-2 vs ZeRO-3 maps onto FSDP's strategies, and seeing them as the same idea in two libraries is the lesson.
5. **The scaling study** — profile at 1, 2, 4, 8 GPUs and plot **scaling efficiency** (samples/sec per GPU ÷ 1-GPU baseline). If it isn't near-linear on one node, find out why *before* adding a second node: usually the dataloader, an unnecessary host sync, or comms not overlapping compute.

**Deliverable:** one DiT config that requires FSDP, plus a scaling-efficiency plot from 1→8 GPUs and a written explanation of where the efficiency went. This is a portfolio artifact — it's exactly what a systems interview asks you to describe.

### 10c — Submit one Vertex AI custom job *(half a day, ~$20)*

Take the **exact same script** from 10a and submit it as a **Vertex AI custom training job**. Nothing about the code changes; only how it is launched.

The value is not Vertex-specific — it is feeling the **submit-vs-interactive shift**:

| | **VM** (10a–10b) — the workshop | **Vertex** — the factory |
|---|---|---|
| Start-up | already warm, `python train.py` | **cold provision, minutes** |
| Debugging | SSH in, poke live, `nvitop` | **no persistent shell** — logs and metrics only |
| Re-run | edit, re-run in seconds | **cancel + resubmit** |
| Good for | iteration | unattended, reproducible, queued runs |

Every cloud ML job posting assumes you have lived that once, and it makes the "VM = workshop, Vertex = factory" distinction real instead of theoretical. Package the container, submit, watch the logs, retrieve the checkpoint from GCS. Then stop.

**Deliverable:** one completed Vertex job producing a checkpoint in GCS, launched from unmodified 10a code.

### Kubernetes — deliberately out of scope for *training*, and here is the reasoning

Not an omission, a decision. **Training is a bounded job with resources known up front → a VM or a Vertex job is the right tool.** K8s earns its keep for (a) always-on inference with variable traffic and (b) shared multi-team fleets with quotas. Neither describes you, and K8s-for-training is ops trivia that will not make you better at diffusion.

**⚠️ The one case where this reverses: reinforcement learning.** RL is genuinely the workload K8s + Ray were built for, because it is **heterogeneous and dynamic** in a way supervised training is not:

- Many **CPU rollout workers** + fewer **GPU learners** + separate **inference/verifier** models, all live at once
- Workers come and go → **elastic scaling** matters
- Ray originated at Berkeley *for RL* (RLlib came out of that project), and modern RL-for-LLM stacks (verl, OpenRLHF, NeMo-RL) are built on Ray
- The 2026 GKE pattern for this is explicit: **Kueue + JobSet** for admission and gang scheduling, **Ray on top** as the orchestrator, and **dedicated nodes for inference** (vLLM) so memory-heavy training buffers don't contend with compute-heavy rollout generation

**This is relevant to you eventually** — Diamond *is* RL (agent trained inside the world model), and the rollout-generation / policy-training / verification split is exactly that shape. It just isn't relevant at Diamond's single-GPU scale.

**On KubeRay specifically** — I should be precise, because it is easy to overstate. KubeRay is the **operator that runs Ray clusters as native K8s resources**. It is *one* good option on GKE, not the default for everything:

| Workload | Usual GKE choice |
|---|---|
| Plain PyTorch DDP/FSDP | **JobSet + Kueue** (simpler; Kueue admits and enforces quota) |
| A full ML platform (training + pipelines + serving) | **Kubeflow** Training Operator |
| **Ray-native programs — RL, tuning, heterogeneous pipelines** | **KubeRay** |
| LLM/model **inference** | **vLLM** + KServe, or Ray Serve via KubeRay |

Rule of thumb: **if your workload is already a Ray program, KubeRay is how you run it on K8s. If it's plain PyTorch, JobSet + Kueue is simpler and more common.** The 2026 consensus stack is roughly *vLLM (inference) + Kueue (GPU scheduling) + KServe (serving) + Ray (distributed training/RL)*.

**If you do want K8s:** treat it as a **separate track**, not part of this plan — GKE Autopilot plus one KubeRay cluster, motivated by an RL or serving workload rather than by diffusion training. And note it becomes a genuine requirement if you target ML **platform** roles rather than ML/research engineering.

- **10a — single node, multiple GPUs** (`a3-highgpu-8g`, up to 8) — the five runs above. **No orchestrator needed.**
- **10b — multi-node.** *Then* adopt **Accelerate** (same script → DDP/FSDP/DeepSpeed via one config). Now you need something to *provision machines + launch processes across them*. Pick **one**: **Vertex** (managed, submit a multi-replica job) or **Ray on GCE** (`ray up`, Python-native, no K8s). K8s only if joining a shared-fleet org.
- **Optimizing the run** — the other half of this phase, and the part interviews probe: **gradient accumulation**, **activation checkpointing**, **bf16/AMP**, `torch.compile`, **FlashAttention**, overlapping communication with compute, `DistributedSampler` correctness, and finding the point where scaling stops being linear. Profile at 1, 2, 4, 8 GPUs and plot **scaling efficiency** — if it's not ~linear on one node, find out why before adding a second node.
  - *What decides what:* **model size → the parallelism** (fits 1 GPU = DDP; too big = FSDP/ZeRO; enormous = +tensor/pipeline). **Org context → the orchestrator** (academia/HPC = Slurm; cloud startup = Vertex/managed; big shared fleet = K8s±Ray). Different axes.

---

## Part G — Reproduce the current stack

### Phase 11 — MIRA end-to-end *(where Phases 5 + 6 + 9 + 10 all cash in)*
[MIRA](https://mira-wm.com/blog-post/) (General Intuition + Kyutai + Epic Games) is the best-documented open target: **5B DiT + 600M representation autoencoder**, trained with **diffusion forcing in the codec's latent space**, simulating 4-player Rocket League at 576p/20 fps on one B200. It is the *only* release combining **Apache-2.0 training code** ([`mira-wm/mira`](https://github.com/mira-wm/mira)) with **a real dataset** ([Rocket Science](https://huggingface.co/datasets/kyutai/rocket-science) — 1,000 match-hours × 4 synchronized views at 720p, with action streams *and* physics states held back for evaluation).

**The cheap on-ramp is the whole reason this is feasible:** the repo trains a **single-player** world model first, then **warm-starts the 4-player model from that checkpoint**. Multiplayer is optional. Do single-player at reduced resolution and model size and you have learned the entire modern stack.

**What it teaches, all at once:**
- **Representation autoencoder** — frozen **DINOv3-L/16** encoder + trained decoder (Phase 5, for real)
- **DiT backbone** at scale (Phase 6)
- **Diffusion forcing** (Phase 9)
- **Action conditioning + action dropout** — 3-valued axis commands (steer/throttle/yaw/pitch/roll) + binary buttons (jump/boost/handbrake) at 15 Hz. Action dropout is CFG's trick from Phase 7, applied to actions, and it's what enables autopilot
- **Multi-view joint generation** (see Phase 16)
- **Multi-GPU training** via `torchrun` (Phase 10)

**What it does *not* teach:** few-step distillation (it needs a B200 for 20 fps precisely *because* it isn't distilled — that's Phase 12), explicit/retrieval memory (Phase 16), and text conditioning/MMDiT (Phase 7).

> **Precision on MIRA's stability claim** (it's easy to overstate): MIRA **does** use diffusion forcing — that's its training paradigm, not an optional extra. The claim is that it needed **no *additional* specialized anti-drift machinery** on top, and that the **representation codec** — not the forcing scheme — was the largest contributor to stability. So: diffusion forcing ✓, extra anti-drift tricks ✗, and the codec is the credited hero.

⚠️ **No pretrained weights released** as of the repo README — you train the codec **and** the world model yourself. **Scale down aggressively** and cost it out before starting. If you want to *play* a SOTA world model without training one, use **LingBot-World** (Phase 16) — it ships weights.

---

## Part H — Make it fast, then ship it

### Which optimization solves which problem *(read this before starting any of 12–15)*
Part H contains three different tools and they are **not interchangeable**. Diagnose first, then pick the phase:

| Your problem | Best first method | Phase |
|---|---|---|
| Model does not fit in GPU memory | Quantization or distillation | 15 / 12 |
| Need to run on a smaller GPU | Distillation + quantization | 12 → 15 |
| Diffusion model takes too many network calls | **Diffusion-step distillation** | **12** |
| GPU spends time launching many small operations | **Kernel fusion** | **14** |
| Model is memory-bandwidth limited | Quantization + fusion | 15 + 14 |
| Need higher serving throughput | Quantization, batching and fusion | 15 + 14 |
| Need lower single-request latency | Distillation and fusion | 12 + 14 |
| Need longer LLM context or larger batch | Weight / KV-cache quantization | 15 |
| Need **no** model-quality change | **Kernel fusion first** | **14** |

**Three things to read off that table:**
- **The last row is the ordering rule.** Fusion is the only lever here that is mathematically quality-neutral — it changes *how* the same arithmetic runs, not what it computes. Distillation and quantization both trade quality. So when in doubt: fusion first, and only spend quality when fusion has run out.
- **For an interactive world model, row 3 dominates everything.** Your bottleneck is NFE — number of network calls per frame — and no amount of fusion or quantization fixes a 1000-step sampler. Distillation is the *only* tool that attacks it. That's why Phase 12 comes before 14 and 15 despite fusion being safer.
- **Rows 5 and 9 are the same diagnosis with different constraints**, and both come straight out of Phase 2's vocabulary. If you can't yet say "I am memory-bandwidth-bound" from a profiler trace, you are not ready to pick a row — go back to Phase 2.

> Corollary worth stating plainly: **distillation is the big diffusion inference win; quantization is the smaller, riskier secondary squeeze.** Diffusion quantization is harder than LLM quantization (error compounds across denoising steps, activation ranges swing across timesteps, artifacts are *visible*). Hence 12 → 14 → 15, in that order.

### Phase 12 — Distillation
The multi-step sampler is the bottleneck. Distillation only makes sense once Phases 8–11 have produced a real model, and for interactive models it's **fused with Phase 9's causal training** (Self-Forcing *is* a rollout scheme and a distillation objective at once).

**There are ~7 families. You do not need all 7.** Triage:

| Family | Verdict |
|---|---|
| **Distribution matching — DMD → DMD2** | **Implement.** The workhorse. DMD2 drops the regression loss and adds a GAN term so it learns from real data. Every real-time interactive model uses a DMD-family objective |
| **Self-Forcing / Causal Forcing** | **Implement.** The interactive-specific one; not optional for real-time world models (Phase 9) |
| **Guidance distillation** (fold CFG into the model) | **Implement.** CFG doubles NFE, so this is a nearly free ~2×. Cheap, universal, and most curricula forget it |
| **Consistency — CM → LCM / LCM-LoRA, iCT/ECT, sCM, TCD** | **Understand, don't implement.** Explain the consistency objective and why LCM-LoRA mattered (distillation as a swappable adapter). Current data point: **sCM** wins at very low NFE, **MeanFlow** gives sharper detail at higher NFE |
| **Progressive distillation** | **Read only.** Historical; the idea takes ten minutes |
| **Adversarial — ADD (SDXL-Turbo), LADD (SD3-Turbo)** | **Read only.** Know the family exists and that it's GAN-flavored; strong at ~4 steps |
| **Teacher-free few-step — shortcut models, MeanFlow** | **Read the abstracts.** Frontier; notable for breaking the "you need a teacher" assumption |

> **Today's best practice for interactive models, in one sentence:** *a **few-step causal student**, distilled from a **bidirectional teacher** with a **DMD-style distribution-matching loss**, trained with **self-forcing rollout**, served with **KV-cache streaming**.* That is exactly what Matrix-Game 2.0 (3 steps, 25 FPS on one H100) and `minWM` implement. Reality check: 1–2 steps still visibly degrades.
> **How to judge the result:** never FID alone. Adversarial and distribution-matching methods buy sharpness by shedding mode coverage, so report **Recall / Coverage** from Phase 4 alongside wall-clock NFE. Also check whether the method preserves the deterministic noise→sample map — interpolation, editing and reproducibility depend on it.

### Phase 13 — Hyperparameter sweeps (a search problem, not a scaling problem)
Distillation gives you real knobs worth tuning (step count, LR, loss weights, EMA), so this is the natural place to learn sweeps as a *first-class skill*. **Key distinction:** a sweep is **many runs of the same model with different configs to find the best** — orthogonal to multi-node, which is *one* run split for speed/size. (Each trial might itself be single- or multi-GPU.)

1. **Start dumb:** a plain Python `for` loop over configs, sequential. Feel why it's slow and why you need scheduling.
2. **Adopt a sweep tool:** **W&B Sweeps** (simplest, best default) · **Optuna** (Pythonic, TPE, pruning) · **Ray Tune** (parallel scheduling + ASHA/PBT early-stop) · **Vertex HP Tuning** (managed).
3. **Learn the ideas over the tool:** search-space design, random vs Bayesian vs grid, **early-stopping/pruning** (ASHA), and how trials get scheduled in parallel.
- *Same muscle as* the Phase 3 four-way race and Phase 4 metrics table — now formalized.

### Phase 14 — Kernels & performance engineering *(below the framework)*
Phase 2 taught you to **read** the GPU. This phase teaches you to **change what it runs**. The trigger is specific: your profiler shows you're **memory-bandwidth-bound** or dominated by launch overhead — many small kernels, low arithmetic intensity — which is the normal state for real-time diffusion inference at batch size 1.

1. **Exhaust the free wins first, in order:** `torch.compile` (kernel fusion for free), **FlashAttention / SDPA backends**, channels-last, CUDA graphs (kills launch overhead), bf16, and removing host syncs. Measure after each — most "we need custom kernels" turns out to be a missing `torch.compile`.
2. **Then write kernels.** Learn **Triton** first (Python-like, most of the win, far less pain than CUDA C++): write a fused element-wise chain, then a fused norm+modulation (adaLN is a perfect target), then a simple attention variant. Compare against `torch.compile`'s output — sometimes you lose, and knowing when you lose is the skill.
3. **The concepts that matter more than the syntax:** arithmetic intensity and the **roofline model**, memory coalescing, tiling and shared memory, occupancy vs register pressure, why **fusion** is almost always a bandwidth argument rather than a FLOPs argument.
4. **The interactive-model target:** a **KV-cache attention kernel** for streaming rollout, and fused sampler steps. This is where Phase 12's few-step model becomes genuinely real-time.
- *Why here:* you need a real model, a real latency target, and profiler fluency before kernel work is anything but premature optimization.

### Phase 15 — Quantization + deploy
Quantize the distilled model, then serve it. **Diffusion quantization is harder than LLM quantization** (error accumulates across denoising steps; activation ranges swing across timesteps; artifacts are *visible*) — so do it empirically, watching it break:
1. **Weight-only int8/fp8** — the safe baseline; should just work.
2. **Push low-bit / activation quantization** — *expect* artifacts. Observe the timestep-varying-activation problem firsthand → understand why timestep-aware methods (Q-Diffusion, PTQD, SVDQuant 4-bit) exist. "Quantization that fights back" is the real lesson.
- Ordering reminder: **distillation** (fewer steps) is the *big* inference win; quantization is the smaller, riskier secondary squeeze — hence last.
- Then build **one custom inference container** (FastAPI / Triton), serve on **Cloud Run or Vertex Endpoint** — no K8s needed. Measure latency/quality live. For an interactive world model the real serving problem is **streaming KV-cache inference at a frame deadline**, not batch throughput.

---

## Part I — Frontier

### Phase 16 — Memory & multiplayer *(read + selectively reproduce)*
These are **three different problems**, not three solutions to one — the most common confusion in this literature.

**First, separate the two axes.** *Temporal stability* = "does the image degrade as I roll out?" (drift — Phase 9's problem). *Spatial memory* = "I turned around; is the room still the *same* room?" A model can be perfectly stable and have **zero** memory: it never diverges, it just quietly invents a different room. Solving one does not solve the other.

| Approach | Problem it actually solves | Mechanism |
|---|---|---|
| **MultiGen** (2026) — external memory | **Persistence + editability + shared world for N players.** "If I build a wall, walk away, come back — is it there? Could I have authored it?" | An **explicit data structure outside the network**: map geometry, poses, minimap, agent states. Independent of the context window, updated by actions, queried every step. Splits the engine into **Memory / Observation / Dynamics**. Each player runs their own Observation+Dynamics against **one shared memory** → arbitrary N players, ~20 FPS, stable 30-min 4-player sessions, levels authored as coarse 2D geometry |
| **MIRA** (2026) — synchronized views | **Several players must see the same event consistently, from different cameras, at the same instant.** | **Joint generation.** One model emits all 4 first-person views *together*, trained on 4 synchronized views per match. Consistency is implicit in the joint forward pass — **no external memory**. Fixed player count, baked into training |
| **Matrix-Game 3.0** (2026) — long-horizon memory | **Spatial consistency over minutes.** | **Retrieval.** Camera-aware selection picks only *view-relevant* past frames using camera pose + field-of-view overlap, then injects those memory latents into the **same joint self-attention space** as past and noisy-current latents, with relative **Plücker** (ray) encoding. 720p at up to 40 FPS with a 5B model |

**How MultiGen actually renders a point of view** *(the mechanism, because "external memory" is vague until you see it):*
- **Memory holds two things:** a **static level map** as a set of **2D vertices and line segments** defining walkable layout and walls; and **dynamic player poses** — `(x, y, yaw θ)` per player. That's it. Symbolic, tiny, human-editable — which is why you can *author* a level as coarse 2D geometry.
- **Observation** (the diffusion model — a **UNet** with cross-attention, not a DiT) generates the next first-person frame conditioned on: the last **L** frames, the **next action**, and — the clever part — a **geometric signal obtained by ray-tracing the 2D map from the current pose**: a 1-D depth vector across the field of view, converted to **disparity** (inverse depth) and concatenated with the context frames. So the model is *told* the wall layout in front of you; it only has to render it, not remember it.
- **Dynamics** is a **small transformer encoder** taking (current pose, action, geometry signal, intermediate UNet features) → predicts an **incremental pose update**, applied with angle wrapping. Poses are initialized as part of the input state.
- **How players see each other:** when rendering player A's frame, the system **queries the shared memory for other players visible from A's pose** and folds them into A's geometric conditioning. Nothing is synchronized *between* the generators — they all just read one authoritative state.
- **Cost reality check:** ~20 FPS using **one A100 per player**. Four players = four A100s. Elegant interface, linear hardware cost.

**Why MIRA can't take a 5th player.** Concretely: the network's input and output tensors carry exactly **4 view streams and 4 action streams**, and its cross-view attention pattern was learned over that fixed set. Adding a 5th changes the shape of the computation — it's the same class of problem as asking a model trained to emit 4 channels to emit 5. The dataset is 4-view too, so there's nothing to learn a 5th from. **MultiGen dodges this by never putting the player count in the network at all**: one player = one Observation+Dynamics instance, N players = N instances against one memory. The count lives in the *system*, not the *weights*. That's the whole architectural argument, and it's the reason to read the paper.

**The relationships, stated plainly:**
- **MIRA and MultiGen reach overlapping goals by opposite means.** MultiGen gets cross-player coherence from **shared explicit state**; MIRA gets it from **one joint model**. MultiGen's is the more general interface (any N, editable); MIRA's is the higher-fidelity single instance. Neither has the other's mechanism.
- **Matrix-Game 3.0 sits between them:** memory is **learned and retrieved** (inside attention) rather than authored (MultiGen) or absent (MIRA). Strongest *spatial* claim; MIRA has the strongest *stability* claim; MultiGen the strongest *persistence/editability* claim. **Different axes — none implies the others.**
- **LingBot-World** (Robbyant/Ant Group) — the largest fully open one: **Wan2.2-based DiT trained with FSDP**, MoBA attention, 720p/60fps, hour-long rollouts, multi-user, plus Pilot/Director agents. **Code *and* weights released.** Verdict: **read + run, don't reproduce.** It's the best way to *feel* a state-of-the-art world model (MIRA ships no weights), and it's a concrete reference for the FSDP scale-out of Phase 10.
- **Genie / Genie 2 / Genie 3** (DeepMind) — the other frontier line, and the only one that learns **actions themselves** rather than assuming them. Genie 3 does navigable 3D worlds at 24 fps in real time, general-purpose. **All closed** — reference points only. The mechanism is reproducible at small scale though, which is Phase 17.
- **Natural capstone:** graft MultiGen's Memory/Observation/Dynamics decomposition onto a world model you already control. **Diamond is by far the cheapest host** — you already have the rollout loop *and* an RL agent. `minWM` is explicitly built as a pluggable framework for this kind of surgery.

**How to evaluate any of this — the two axes need two different tests:**
- **Drift** → the **drift curve** from Phase 9 (quality vs rollout position).
- **Spatial memory** → the **return-to-viewpoint (loop-closure) test**: drive a closed loop — turn 360°, or leave a room and come back — and compare the frame at the *returned* pose against the frame originally captured at that pose (PSNR/LPIPS, plus "is it the same room at all?"). Report **loop length vs similarity**. A model with no memory scores fine at loop length 0 and collapses immediately after.
- Running only one of these is how papers claim "long-horizon" while failing the other. **MIRA is a live example:** it has a strong drift result and *no memory mechanism*. It gets away with it because **Rocket League barely needs spatial memory** — one fixed symmetric arena, almost entirely in frame at all times, very little offscreen geometry to forget. The *task* hides the problem. Put the same architecture in a world where you walk into a building and back out and the gap appears immediately. **Always ask what the benchmark's environment lets the model skip.**

### Phase 17 — Latent actions: where do actions come from when nothing is labelled?

*Placed last on purpose: this is orthogonal to everything above. It does not block Phases 12–16 and nothing in them depends on it — it answers a question the rest of the plan quietly assumes away.*

**The assumption this phase attacks.** Every world model in this plan is handed its actions:

| Phase | Where its actions come from |
|---|---|
| 8 Diamond | the RL agent generated them |
| 11 MIRA | Rocket Science ships labelled action streams at 15 Hz |
| 12 Matrix-Game | action-annotated Unreal/GTA5 capture |
| — minWM | camera-control fine-tune on labelled trajectories |

So the plan's implicit answer to *"where do actions come from?"* is **"somebody labelled them"** — which caps you at curated datasets and rules out the entire internet. **Latent action models are the only route past that.**

**1. Read Genie, then diff it against an implementation.** [Genie](https://arxiv.org/html/2402.15391v1) (Bruce et al., DeepMind, ICML 2024 best paper) has three parts:

1. a **spatiotemporal video tokenizer** (ST-ViViT / MagViT-style)
2. a **Latent Action Model** — a **VQ-VAE over frame pairs** producing a *deliberately small* discrete codebook, so codes become interpretable (`MOVE_RIGHT`)
3. an **autoregressive dynamics model** (MaskGIT) predicting the next frame from video tokens + latent action

Trained **entirely unsupervised on unlabelled video** — no ground-truth actions anywhere — and you can still act frame-by-frame in the result.

Then read [`myscience/open-genie`](https://github.com/myscience/open-genie) (MIT) **as a companion to the paper**. Its three modules map 1:1 onto the paper's three components, which makes it well suited to paper↔code reading — the same technique used in Phase 9 (Diamond's loop vs Self-Forcing's) and Phase 7 (your DiT vs `diffusers`').
> ⚠️ **Read it, don't run it.** Its own roadmap still lists "add functioning training script" and "show some results" as open TODOs. No datasets, no weights, no documented hardware requirements. Trust the **architecture**; treat the **training details** as unverified.

**2. Do NOT reproduce Genie.** The numbers, so the decision is obvious: **11B parameters**, **30,000 hours** of 2D-platformer video, filtered from **6.8 million videos** using a learned quality classifier that itself needed **10k human-labelled videos**. The scaling study alone ran 40M → 2.7B before the final 11B. The data is not provided and curating it would be a bigger project than the model. This is an industrial TPU-pod run.

**3. Actually run LAPO instead** — [`schmidtdominik/LAPO`](https://github.com/schmidtdominik/LAPO), *"Learning to Act without Actions"* (Schmidt & Jiang, ICLR 2024 spotlight). It isolates the same idea at roughly a thousandth of the compute:

```
IDM (inverse dynamics):  (frame_t, frame_t+1)  →  latent action a
FDM (forward dynamics):  (frame_t, a)          →  predicted frame_t+1
```

Train jointly for **predictive consistency** — the IDM must emit an `a` that lets the FDM reconstruct the true next frame, so `a` becomes *whatever information explains the transition*.

> **The whole trick is the information bottleneck on `a`.** Without it the IDM cheats: it copies frame_t+1 straight through and "the action" becomes the answer. Constrain `a` to be tiny and it is forced to encode only the **cause** of the change. Genie's small VQ codebook implements the same constraint by different means — recognising that these are the same idea is the insight of this phase.

Output: latent-action policies, a world model, and an IDM — all from video. Then fine-tune to real controls either offline with a **small** labelled set or online with rewards.

**Practicalities (verified):** 16 Procgen tasks, expert `.npz` data provided via Google Drive, **~1 hour per stage × 3 stages** per task, GPU required, **~40 GB host RAM** (it loads ~2.5M frames — a machine-shape constraint, not a GPU one). **No pretrained checkpoints.** The download script can hit Drive bandwidth limits; there's a manual fallback. **≈$5 on an L4.**

**The family, sorted** — so you pick the right prior for a domain:

| Method | Latent action type | Scale | Domain fit |
|---|---|---|---|
| **Genie** | small **discrete** VQ codebook | internet-scale | discrete controls (platformers: left/right/jump) |
| **LAPO** | bottlenecked latent | RL-benchmark, **tractable** | discrete-ish control, Procgen |
| **CLAM** | **continuous** | robotics | continuous control — joint torques, manipulation. A codebook of 8 actions is the wrong prior here |

**Deliverable:** LAPO trained on 2–3 Procgen tasks, with the recovered latent action space compared against the true action space (does code 3 reliably mean "jump"?), plus a written account of how Genie's codebook and LAPO's bottleneck are the same mechanism.


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
- **torchrun** — raw launcher: spawns 1 process/GPU, sets `RANK`/`LOCAL_RANK`/`WORLD_SIZE`, sets up rendezvous. You write the DDP/FSDP wrapping. *Assumes machines already exist.* Most transparent — use it by hand in Phase 10a. (This is what MIRA uses.)
- **Accelerate** — a convenience wrapper *over* torchrun: one config switches your script single-GPU ↔ DDP ↔ FSDP ↔ DeepSpeed with no code change. *Still assumes machines exist; does not sweep.*
- **Ray Train** — launcher **+ its own cluster manager**: coordinates workers across a Ray cluster with fault tolerance/elasticity, integrated with Ray Data/Tune/Serve. *Can also provision the machines.*

### VM ↔ Vertex dev loop
- **VM** = interactive: edit, `python train.py`, watch `nvitop` live, re-run in seconds. The workshop.
- **Vertex** = submit a job → cold-provision (minutes) → runs elsewhere → watch logs/metrics live but **can't interactively poke** (no persistent shell; "rerun" = cancel + resubmit). The factory.
- Standard workflow: **prototype on VM → submit the same code as a Vertex job to scale.**

### K8s, Ray, Kubeflow — who sits where
- **K8s = infrastructure manager:** "run these containers, restart crashed ones, share machines across teams, quotas/RBAC/networking." Knows nothing about your Python.
- **Ray = distributed-application engine:** "run this Python program across the cluster" — actors, elastic fault-tolerant training, Train/Tune/Data/Serve. K8s alone can't do elastic training or coordinate a Python compute graph — **that's what Ray adds.**
- **Ray-on-VMs** (`ray up`) vs **Ray-on-K8s** (KubeRay): same Ray, different provisioner. Ray-on-VMs when you have no platform (simplest, our choice); Ray-on-K8s when the org already standardized on K8s.
- **Kubeflow = an ML toolkit *for* K8s** — a *suite* of components: Training Operator (`PyTorchJob`), Katib (tuning), KServe (serving), Pipelines (DAGs). Contrast Ray, which offers Train/Tune/Data/Serve as *one unified Python library*.

### Accelerate — no in Phase 2, yes in Phase 10b
Raw PyTorch first (see the mechanics). Adopt Accelerate at multi-node: same script runs single-GPU → DDP → FSDP → DeepSpeed by changing **one config**. It parallelizes **one run**; it does *not* sweep.

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
- **Training** = runs to completion, resources known up front → **GCE VM, no K8s.** (K8s only at 100s-of-GPUs / shared clusters, for fault-tolerance & scheduling — not traffic autoscaling.)
- **Inference** = long-running service, variable traffic, must stay up → K8s's reason to exist. But you can also serve on **Cloud Run / Vertex Endpoint** without K8s. Inference = a **custom container** (model + serving code).

### Slurm vs K8s vs Ray
- **Slurm** and **K8s** are **peer** cluster schedulers from different worlds (HPC/academia vs cloud). Pick one — you don't stack Slurm on K8s.
- You're in **K8s-world (GCP)** → skip Slurm unless you head to HPC/academia.
- **Ray** is the thing that *does* run "on top of" — Ray-on-K8s (KubeRay), Ray-on-Slurm, or Ray-on-plain-VMs.

---

## Next step
**Phase 1 (local).** On your Mac, stand up the production repo layout, write the diffusion core + unit tests, **overfit one batch to ~0 loss on CPU/MPS**, and **cut the `build_model(cfg)` seam** so Phase 6's DiT is a config change rather than a refactor. Then **Phase 2**: move that same repo to a cheap L4/T4 **GCE VM**, wire up `nvitop` + the profiler, and learn to name your bottleneck. Everything else builds on those two muscles — correct code, then systems literacy.
