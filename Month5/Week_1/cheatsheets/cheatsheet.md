# Month 5, Week 1 Cheatsheet: Fine-Tuning, Parameter-Efficient Adaptation (PEFT) & QLoRA

---

## 1. Parameter-Efficient Fine-Tuning: LoRA Mathematics

Standard full fine-tuning modifies $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, updating all $d_{\text{in}} \times d_{\text{out}}$ weights. LoRA freezes $W_0$ and decomposes updates into low-rank matrices:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A) x$$
where:
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$: Initialized with Gaussian noise $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$: Initialized to **exact zero** (ensures $\Delta W = 0$ on step 0).
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$: Intrinsic adapter rank (typically $8 - 64$).
- $\frac{\alpha}{r}$: Constant scaling factor ensuring stable gradient steps when varying rank $r$.

---

## 2. QLoRA: NormalFloat-4 (NF4) & Double Dequantization

QLoRA (Dettmers et al. 2023) enables 4-bit fine-tuning of 70B models on a single GPU:

```
                      QLoRA Memory Architecture
 ┌────────────────────────────────────────────────────────┐
 │ 4-Bit NormalFloat Base Weights (NF4)                   │  Frozen base weights
 ├────────────────────────────────────────────────────────┤  (Saves 75% VRAM)
 ├───────────────────┬────────────────────────────────────┤
 │ LoRA Adapter A    │ LoRA Adapter B                     │  Trainable FP16 / BF16
 │ (rank r=16)       │ (rank r=16)                        │  (< 1% of total params)
 └───────────────────┴────────────────────────────────────┘
```

### NormalFloat-4 (NF4) Information Density
- Weights follow a normal distribution $\mathcal{N}(0, \sigma^2)$.
- NF4 quantiles are derived such that each discrete 4-bit bin contains identical probability mass $\frac{1}{16}$, maximizing information entropy compared to uniform INT4.
- **Double Dequantization**: Quantizes the 32-bit quantization constants down to 8-bit, saving an additional 0.37 bits per parameter.

---

## 3. Production Weight Merging for Zero-Overhead Inference

During serving, eliminate adapter matrix latency by folding weights into base parameters:
$$W_{\text{serving}} = W_0 + \frac{\alpha}{r} (B A)$$

```python
from peft import AutoPeftModelForCausalLM

# Load trained adapter
model = AutoPeftModelForCausalLM.from_pretrained("/checkpoints/lora_adapter")
# Merge adapter weights directly into base FP16 weights
merged_model = model.merge_and_unload()
merged_model.save_pretrained("/models/fused_production_model")
```

---

## 4. Supervised Fine-Tuning (SFT) Masked Loss Labeling

Only calculate cross-entropy loss on the assistant's generated completion tokens:
```python
# System + User prompt tokens set to label -100 (ignored by PyTorch CrossEntropyLoss)
labels = [-100] * len(prompt_tokens) + completion_tokens
```
