# Week 1 Production Engineering Assignment: End-to-End QLoRA Fine-Tuning & Adapter Deployment Pipeline

## Executive Overview & Architectural Motivation
Full fine-tuning of open-source Large Language Models (7B to 70B parameters) demands clusters of multi-GPU servers (8x A100/H100) running ZeRO-3 distributed optimizers. **QLoRA (Quantized Low-Rank Adaptation)** democratizes production LLM customization by freezing 4-bit NormalFloat base weights while backpropagating gradients strictly through low-rank adapter matrices, enabling high-performance domain adaptation on a single commodity GPU.

In this assignment, you will engineer a complete **QLoRA Fine-Tuning & Serving Pipeline** for domain-specific instruction alignment.

Your pipeline will prepare a structured instruction dataset, configure 4-bit base weight quantization, inject trainable LoRA adapters into multi-head attention projections, train the adapter weights using mixed precision, and fuse the final weights into an optimized production checkpoint ready for vLLM deployment.

---

## Technical Specifications & Architecture Diagram

```
                 Raw Domain Instruction Corpus (JSONL)
                 [{"instruction": "...", "input": "...", "output": "..."}]
                                    │
                                    ▼
                 ┌───────────────────────────────────────────────────────┐
                 │          Instruction Formatting & Tokenization        │
                 │   - ChatML / Llama-3 Special Token Templates          │
                 │   - Masked Loss Labels (Only calculate loss on output)│
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │       4-Bit Quantized Base Model Ingestion            │
                 │   - NormalFloat-4 (NF4) Weights Frozen                │
                 │   - Double Quantization Enabled                       │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │          LoRA Adapter Injection Layer                 │
                 │   - Targets: q_proj, k_proj, v_proj, o_proj           │
                 │   - Rank: r=16, Alpha: 32.0, Dropout: 0.05            │
                 │   - Trainable Parameters: < 1.0% of Base Model        │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │          Fused Weight Merging & Checkpointing         │
                 │   - W_final = W_dequant + (alpha/r) * (B @ A)         │
                 │   - Export to GGUF / SafeTensors format               │
                 └───────────────────────────────────────────────────────┘
```

---

## Detailed Deliverable Requirements

### 1. Instruction Dataset Masking & Tokenization (`dataset.py`)
Implement `InstructionDatasetBuilder`:
- Formats prompts using the standard Llama-3 instruction format:
  `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n...<|eot_id|>`
- Applies **Label Masking**: sets target label tokens corresponding to the system prompt and user instruction to `-100` so that cross-entropy loss is computed strictly over generated assistant tokens.

### 2. PEFT LoRA Configuration & Injection Engine (`peft_model.py`)
- Inject LoRA adapters into transformer projection layers (`q_proj`, `v_proj`).
- Track and report trainable vs frozen parameter counts:
  $$\text{Trainable Ratio} = \frac{\sum \text{Params}_{\text{LoRA}}}{\sum \text{Params}_{\text{Base}} + \sum \text{Params}_{\text{LoRA}}} \times 100\%$$
- Assert that the trainable ratio is less than $1.5\%$.

### 3. Fused Adapter Merging & Benchmark Verification (`merge.py`)
- Implement `merge_and_save(model, adapter_weights, output_path)`:
  - Dequantizes base weights and folds adapter matrices directly: $W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B A)$.
  - Verifies that post-merge inference latency matches unadapted base inference.

---

## Starter Code Template & Submission Skeleton

```python
"""
qlora_finetuning_pipeline.py - Production PEFT & LoRA Suite
"""
import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple

class LoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False  # Freeze base weights
        
        self.rank = rank
        self.scaling = alpha / rank
        
        in_dim = base_layer.in_features
        out_dim = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) / math.sqrt(in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

class ToyTransformerBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        v = self.v_proj(x)
        return self.out_proj(q + v)

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    dim = 128
    block = ToyTransformerBlock(dim=dim)
    
    # Count original parameters
    orig_params = sum(p.numel() for p in block.parameters())
    
    # Inject LoRA into q_proj and v_proj
    block.q_proj = LoRALayer(block.q_proj, rank=4, alpha=8.0)
    block.v_proj = LoRALayer(block.v_proj, rank=4, alpha=8.0)
    
    trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in block.parameters())
    
    print(f"Total Model Parameters:     {total_params}")
    print(f"Trainable LoRA Parameters:  {trainable_params}")
    print(f"Trainable Parameter Ratio:  {(trainable_params / total_params) * 100:.2f}%")
    
    assert trainable_params < total_params * 0.10
    
    # Test forward and backward pass
    inputs = torch.randn(2, 10, dim)
    outputs = block(inputs)
    loss = outputs.sum()
    loss.backward()
    
    assert block.q_proj.lora_B.grad is not None
    assert block.q_proj.base_layer.weight.grad is None
    print("Fine-tuning gradient isolation verified successfully!")
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **LoRA Mathematical Formulations** | 30 | Exact low-rank matrix decomposition, scaling factor $\frac{\alpha}{r}$, zero initialization. |
| **NormalFloat-4 (NF4) Theory & Quant** | 25 | Information density derivation, quantile codebook generation, quantization noise checks. |
| **Masked Loss Label Preparation** | 20 | Strict masking of instruction prompt tokens (label = -100); cross-entropy on targets only. |
| **Weight Merging & Inference Fold** | 15 | Flawless weight folding eliminating adapter runtime overhead in serving mode. |
| **Verification & Reproducibility** | 10 | Complete runnable execution script asserting gradient isolation and parameter counts. |
