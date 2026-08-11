# Cost & Resource Analysis: VibeThinker-3B-Agent

**Project timeline:** 3–5 weeks  
**Total cost:** $0–800 USD  
**Minimum viable compute:** 1× GPU with 12GB+ VRAM

---

## 1. Timeline Breakdown

### Phase 0 — Diagnostics (3–5 days)
- Reproduce baseline scores: 2 days
- Cold-start format probe: 1 day
- Setup & data prep: 1–2 days
- **GPU hours:** ~2 (inference only, no training)
- **Cost:** $0 (free Kaggle/Colab)

### Phase A — Single-Step Tool-Calling RL (7–10 days)
- Data preparation (xLAM + ToolACE): 1–2 days
- Training: 5–7 days (depends on compute, see below)
- Evaluation & regression suite: 2–3 days
- **GPU hours:** 40–80 (varies by GPU type)
- **Typical run:** 1× T4 for 6 days = ~144 hours, or 1× A100 for 12 hours

### Phase B — Multi-Turn Agentic RL (10–14 days, conditional on Phase A)
- Checkpoint conversion & setup: 1 day
- Training: 7–10 days
- Evaluation: 2–3 days
- **GPU hours:** 80–160 (on-policy RL is costlier than Phase A)
- **Typical run:** 2× A100 for 4 days, or 1× A100 for 8 days

### **Total timeline: 3–5 weeks** (24–30 days elapsed; parallelizable portions reduce this)

---

## 2. GPU Tier Matrix

| Tier | Hardware | Phase A | Phase B | Notes |
|---|---|---|---|---|
| **Free (Course)** | Kaggle T4 (15GB) | ✓ (1 week, LoRA) | ❌ (OOM or too slow) | Use for Phase A only |
| **Free (Academic)** | Google Colab A100 | ✓ (3 days) | ⚠️ (unstable, 15-min runtime limit) | Good for prototyping |
| **Mid-tier** | 1× RTX 4090 (24GB) | ✓ (4–5 days) | ⚠️ (slow, 2–3 weeks) | Good if you own it |
| **Enterprise** | 1× A100 80GB | ✓ (12 hours) | ✓ (3–4 days) | Industry standard |
| **Enterprise** | 2× A100 80GB | ✓ (6 hours) | ✓ (2 days) | **Recommended for Phase B** |
| **Cloud alt.** | Google TPU Research Cloud | ✓ (free academic) | ✓ (free academic) | Apply at research.google.com; 1 month approval |
| **LoRA variant** | 1× RTX 3090 (24GB) | ✓ (1 week) | ✓ (2–3 weeks, slow) | Parameter-efficient; trades speed for memory |

---

## 3. Detailed Resource Requirements

### **Base Model Download & Setup**
- **VibeThinker-3B:** ~6GB (model weights + tokenizer)
- **Qwen2.5-Coder-3B base:** ~6GB (download once)
- **Training data (xLAM + ToolACE):** ~2GB (compressed); ~5GB unpacked
- **Intermediate checkpoints:** ~20GB per phase (keep 2–3 best checkpoints)
- **Total disk:** ~40–50GB SSD recommended

### **Software Stack (all free/open)**
```
Python 3.10+
PyTorch 2.0+
vLLM (inference engine)
verl (unified RL training framework)
transformers, peft (for LoRA), datasets, numpy, pandas
```

### **Memory Profile**

| Component | VRAM needed | Notes |
|---|---|---|
| Model (3B FP16) | 6–8 GB | Full precision training needs ~16GB |
| Optimizer state (AdamW) | 2–4 GB | ~2× model size for LoRA, ~6× for full param |
| Batch data & activations | 2–4 GB | Depends on batch size & seq length |
| **Total (Phase A, LoRA)** | **10–14 GB** | Fits on 1× RTX 4090 or 1× A100 40GB |
| **Total (Phase A, full)** | **16–24 GB** | Needs 1× A100 80GB |
| **Total (Phase B, full)** | **20–28 GB** | Phase B has long contexts; ~64K tokens |

**Recommendation:** If you have access to 1× A100 80GB, you can run both phases sequentially. If limited to RTX 4090 or T4, use **LoRA** (Low-Rank Adaptation) for both phases — trades 10–20% speed for ~40% memory savings.

---

## 4. Cost Breakdown (if using cloud)

### **Google Colab (per-session credits)**
- **A100 GPU:** $0.31/hr (with Colab Pro)
- **Phase A (8 A100-hours):** ~$2.50
- **Phase B (20 A100-hours):** ~$6.20
- **Total:** ~$10–15 (very cheap; limited by 15-min runtime)

### **Lambda Labs (dedicated cloud)**
- **8× A100 cluster:** $24/hr
- **Phase A (1 day on 2× A100):** ~$48
- **Phase B (3 days on 2× A100):** ~$144
- **Total:** ~$200–250 (reasonable for enterprise)

### **Google Cloud TPU Research Cloud (recommended for students)**
- **Cost:** $0 (free academic allocation; apply at [research.google.com](https://research.google.com/tpu/research-cloud/))
- **Timeline:** 1 month approval; 100 hours/month TPU-v4 equivalent
- **Phase A (6 TPU-hours):** Free
- **Phase B (12 TPU-hours):** Free
- **Total:** $0

### **AWS or Azure (if you have credits)**
- **p3.8xlarge (4× V100 GPU):** $12/hr
- **Phase A (1 day):** ~$96–120
- **Phase B (3 days):** ~$288–360
- **Total:** ~$400–500

### **University HPC (most likely option for CSE465)**
- **Cost:** $0 (if available; check with your department)
- **Example clusters:** XSEDE, NERSC, campus systems
- **Typical allocation:** 10,000–50,000 GPU-hours for a semester

---

## 5. Recommended Setup for CSE465

### **Scenario 1: Use university HPC (best)**
- **Cost:** $0
- **Timeline:** 3–5 weeks (no waiting for cloud approval)
- **Action:** Contact your department; request GPU allocation for a course project
- **GPUs:** 2× A100 or equivalent; ask for 100+ GPU-hours

### **Scenario 2: Use Google TPU Research Cloud (free, requires patience)**
- **Cost:** $0
- **Timeline:** 4–6 weeks (1 month for approval, then 3–5 weeks for training)
- **Action:** Apply at research.google.com; mention it's a course project
- **Allocation:** Request 100 hours TPU-v4 (enough for both phases)

### **Scenario 3: Use Kaggle + LoRA (lowest friction, slower)**
- **Cost:** $0 (free Kaggle T4)
- **Timeline:** 6–8 weeks (Phase A: 1 week on T4; Phase B: skip or use Colab A100)
- **Setup:** 
  - Phase A: Full LoRA on Kaggle T4 (15GB VRAM)
  - Phase B: Either skip, or prototype on Colab A100 (free, 15-min limit means multiple sessions)
- **Trade-off:** No Phase B results, but Phase A alone is a complete project

### **Scenario 4: Use your own GPU (if you have RTX 4090+)**
- **Cost:** Electricity only (~$20–50 for the project duration)
- **Timeline:** 4–6 weeks (Phase A: 4–5 days; Phase B: 2–3 weeks)
- **Setup:** Full LoRA on RTX 4090 or 4080
- **Trade-off:** Slower than A100, but fully yours; no cloud waiting

---

## 6. Compute Budget Checklist

### **Minimum viable (Phase A only)**
- [ ] 1× GPU with 12GB+ VRAM
- [ ] 50GB disk space
- [ ] 2–3 weeks
- [ ] $0–20 (if using free tier)

### **Recommended (both phases)**
- [ ] 1–2× GPU with 24GB+ VRAM (or 1× A100 80GB)
- [ ] 50GB disk space
- [ ] 4–5 weeks
- [ ] $0–200 (university HPC or TPU research cloud)

### **Optimal (enterprise-grade)**
- [ ] 2–4× A100 80GB GPUs
- [ ] 100GB disk space
- [ ] 3 weeks (parallelizable)
- [ ] $200–500 (cloud) or $0 (university)

---

## 7. Throughput & Efficiency Estimates

### **Phase A Training (xLAM + ToolACE, ~68K samples)**

| GPU | Batch size | Seq length | Throughput | Phase A time | Notes |
|---|---|---|---|---|---|
| 1× T4 (15GB) | 8, LoRA | 2K | 10 samples/min | 7 days | Kaggle free tier |
| 1× RTX 4090 (24GB) | 16, LoRA | 2K | 30 samples/min | 2–3 days | Home GPU |
| 1× A100 80GB | 64, full | 2K | 200 samples/min | 6–8 hours | Cloud ($24/hr) |
| 2× A100 80GB | 128, full | 2K | 400 samples/min | 3–4 hours | Ideal; $48/hr |

### **Phase B Training (multi-turn RL with rollouts, ~5K prompts × 8 rollouts per prompt)**

| GPU | Rollout budget | Tool calls | Phase B time | Notes |
|---|---|---|---|---|
| 1× T4 (15GB) | 2 parallel | ~5K | 3–4 weeks | Very slow; not recommended |
| 1× RTX 4090 (24GB) | 4 parallel | ~40K | 2–3 weeks | Affordable but slow |
| 1× A100 80GB | 8 parallel | ~40K | 3–4 days | Good single-GPU speed |
| 2× A100 80GB | 16 parallel + ARPO | ~20K | 2 days | Recommended; efficient |

---

## 8. LoRA vs. Full Parameter Training

### **LoRA (Low-Rank Adaptation)**
- **VRAM saved:** ~40–50% (12GB vs. 24GB)
- **Speed trade-off:** ~10–20% slower (fewer parallel processes)
- **Quality trade-off:** Negligible if rank=16–32
- **Use case:** Kaggle T4, home GPUs, budget-constrained projects
- **Recommendation:** **Use LoRA for Phase A** (single-step tool calling is stable); consider for Phase B if compute is tight

### **Full Parameter Training**
- **VRAM required:** 20–28GB (Phase B)
- **Speed:** 10–20% faster than LoRA
- **Quality:** Marginally better (full expressiveness)
- **Use case:** Enterprise, university HPC, well-funded labs
- **Recommendation:** **Use full training for Phase B** if possible (multi-turn RL is sensitive to expressiveness)

---

## 9. Hidden Costs & Gotchas

| Item | Cost | Mitigation |
|---|---|---|
| **Data transfers (if on cloud)** | ~$5–20 | Pre-cache datasets in cloud storage (Hugging Face, S3) |
| **Interrupted training (cloud OOM)** | Wasted GPU time | Use checkpointing every 100 steps; resume from checkpoint |
| **Baseline reproduction (bugfinding)** | 1–2 extra GPU days | Do Phase 0 diagnostics first; verify environment |
| **Ablation studies** | +50% training time | Run in parallel if possible; prioritize top 2 ablations |
| **Manual verification (Phase B)** | ~10 hrs human time | Budget this for result validation & paper writing |

---

## 10. Total Project Budget Summary

| Scenario | Compute | Cost | Timeline | Viability |
|---|---|---|---|---|
| **Kaggle T4 (Phase A only)** | Free tier | $0 | 7–10 days | ✓ Good for course |
| **Kaggle T4 + LoRA** | Free tier | $0 | 5–6 weeks | ✓ Good for both phases (slow) |
| **University HPC** | 100+ GPU-hrs | $0 | 3–5 weeks | ✓✓ Ideal if available |
| **Google TPU Research** | 100 TPU-hrs | $0 | 4–6 weeks | ✓✓ Ideal; free; just slow approval |
| **Home GPU (RTX 4090)** | Electricity | ~$30 | 4–6 weeks | ✓ Good if you own it |
| **Cloud (Colab + Lambda)** | Pay-per-use | $20–50 | 3–4 weeks | ✓ Flexible; pricey |

---

## 11. Recommendation for Your Team

**Assuming CSE465 is a university course at North South University:**

1. **First choice:** Contact your department's HPC admin or Professor
   - Ask for GPU allocation (100+ hours on any NVIDIA GPU)
   - This is standard for capstone/project courses
   - Cost: $0; Timeline: 3–5 weeks

2. **Second choice:** Apply to Google TPU Research Cloud
   - Free academic allocation (up to 100 hours/month)
   - Takes 1 month to approve; plan ahead
   - Cost: $0; Timeline: 4–6 weeks

3. **Fallback:** Use Kaggle T4 for Phase A + Colab A100 for Phase B
   - Phase A: 1 week on Kaggle T4 (free)
   - Phase B: Prototype on Colab A100 (multiple sessions, 15-min limit each)
   - Cost: $0–10; Timeline: 5–7 weeks; **won't yield production results for Phase B**

4. **If you have budget:** Use Lambda Labs or AWS for 1–2 days of A100
   - Accelerate both phases to 4–5 days total
   - Cost: $200–300; Timeline: 1–2 weeks

---

**Bottom line:** Phase A is doable on free tier in 1 week; Phase B requires better compute (A100 or better). **Budget 4–5 weeks and $0–200, or ask your university for HPC access.**
