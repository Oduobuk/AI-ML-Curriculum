# Week 3 Production Engineering Assignment: Vision Transformer (ViT) from Scratch with Attention Rollout Interpretability

## Executive Overview & Architectural Motivation
Vision Transformers (ViT) have revolutionized computer vision, displacing classical convolutional inductive biases in autonomous driving, satellite imagery classification, medical diagnostics, and multimodal models (CLIP, GPT-4V). However, debugging what a ViT "sees" requires extracting the self-attention weights and computing Attention Rollout maps.

In this assignment, you will engineer a complete **Vision Transformer (ViT) Architecture from Scratch** in pure PyTorch 2.x.

Your implementation will feature convolutional patch projection, learnable 1D spatial embeddings, pre-LayerNorm Multi-Head Self-Attention blocks, and an automated Attention Rollout extraction engine that outputs 2D saliency heatmaps overlaid on input images.

---

## Technical Specifications & Architecture Diagram

```
                 Input RGB Image $X \in \mathbb{R}^{B \times 3 \times H \times W}$
                                        │
                                        ▼
                 ┌───────────────────────────────────────────┐
                 │       Patch Embedding Layer               │
                 │   - Conv2d(k=P, s=P) Projection           │
                 │   - Prepend [CLS] Token                   │
                 │   - Add 1D Learnable Positional Embeddings│
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │       Transformer Encoder Stack (L=6)     │
                 │   For each layer:                         │
                 │   - x = x + DropPath(MHSA(LayerNorm(x)))  │
                 │   - x = x + DropPath(MLP(LayerNorm(x)))   │
                 │   - Intercept raw attention matrix $A_l$  │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │       Classification Head / Rollout       │
                 │   - Extract [CLS] Token -> Linear Head    │
                 │   - Rollout: $\prod (0.5 A_l + 0.5 I)$    │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 Class Probabilities + 2D Interpretability Heatmap
```

---

## Detailed Deliverable Requirements

### 1. Vision Transformer Core Engine (`vit_engine.py`)
Implement `VisionTransformer(nn.Module)`:
- `PatchEmbedding`: Projects $(3, H, W)$ image into sequence of $N = (H/P) \times (W/P)$ tokens of dimension $D$.
- `TransformerEncoderBlock`:
  - Multi-Head Self-Attention with $h$ heads and scale factor $\frac{1}{\sqrt{d_k}}$.
  - Feedforward MLP with GeLU activation ($4 \times$ expansion).
  - Residual connections with Pre-LayerNorm configuration.
- Return both final class logits and all intermediate attention weight matrices for explainability.

### 2. Attention Rollout Interpretability Engine (`rollout.py`)
- Implement `extract_attention_rollout(attentions: List[torch.Tensor]) -> np.ndarray`:
  - Averages attention matrices across all attention heads.
  - Recursively accumulates residual attention flow: $R_{l} = (0.5 A_l + 0.5 I) R_{l-1}$.
  - Extracts the row corresponding to the `[CLS]` token across all spatial patches $1 \dots N$.
  - Reshapes the 1D attribution vector back to a 2D grid $(H/P, W/P)$ and upsamples to $(H, W)$ using bicubic interpolation.

### 3. Verification & Fine-Tuning Script (`train_eval.py`)
- Train or evaluate the model on an image classification dataset (e.g. CIFAR-10 or synthetic 4-class geometric dataset).
- Generate a side-by-side visualization: input RGB image alongside the computed Attention Rollout saliency overlay.

---

## Starter Code Template & Submission Skeleton

```python
"""
vision_transformer_rollout.py - Production ViT Suite
"""
import torch
import torch.nn as nn
import math
from typing import List, Tuple

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        # [B x N x 3*C] -> [B x N x 3 x Heads x HeadDim] -> permute
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B x Heads x N x HeadDim]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        out = torch.matmul(attn_probs, v)  # [B x Heads x N x HeadDim]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out, attn_probs

class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, attn_probs = self.attn(self.norm1(x))
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, attn_probs

class ProductionViT(nn.Module):
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_chans: int = 3, 
                 embed_dim: int = 128, depth: int = 4, num_heads: int = 4, num_classes: int = 10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        
        attentions = []
        for block in self.blocks:
            x, attn = block(x)
            attentions.append(attn)
            
        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits, attentions

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B = 2
    model = ProductionViT(img_size=64, patch_size=8, embed_dim=64, depth=3, num_heads=4, num_classes=5)
    img = torch.randn(B, 3, 64, 64)
    
    logits, attns = model(img)
    print(f"Logits Output Shape: {logits.shape}")
    print(f"Number of Attention Layers Extracted: {len(attns)}")
    print(f"Attention Layer 0 Tensor Shape: {attns[0].shape}")
    
    assert logits.shape == (B, 5)
    assert len(attns) == 3
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Patch Embedding & Tokenization** | 25 | Correct convolutional projection, class token prepending, positional embedding broadcasting. |
| **Multi-Head Self-Attention Mechanics** | 30 | Flawless projection shapes, scaling factor, and raw attention tensor extraction. |
| **Attention Rollout Derivation & Logic** | 25 | Correct recursive multiplication ($\prod 0.5 A_l + 0.5 I$) and spatial reshape to 2D. |
| **LayerNorm & Architectural Rigor** | 10 | Pre-LayerNorm placement, skip connection stability, and clean PyTorch modularity. |
| **Verification & Reproducibility** | 10 | Complete runnable execution script asserting correct output shapes and finite losses. |
