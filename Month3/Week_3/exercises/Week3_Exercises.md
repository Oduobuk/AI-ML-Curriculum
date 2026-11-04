# Week 3: Graduate-Level Problem Set — Modern Computer Vision, ConvNeXt & Vision Transformers (ViT)

Welcome to the Month 3 Week 3 Problem Set. This suite evaluates your mastery of state-of-the-art vision architectures: patch projection embedding, 2D spatial tokenization, learnable class tokens (`[CLS]`), multi-head self-attention on images, Depthwise Separable Convolutions, and Attention Rollout map visualization.

---

## Tier 1: Core Mechanics & Patch Tokenization Geometry

### Exercise 1: Image Patch Projection & 1D Sequence Unfolding
**Concepts**: 2D convolution as patch embedding, sequence length $N = \frac{H W}{P^2}$, learnable positional embeddings.

#### Problem Specification
Consider an RGB input image $X \in \mathbb{R}^{B \times C \times H \times W}$ where $C=3, H=224, W=224$, and patch size $P=16$:
1. Prove that the total number of flattened 1D spatial patches is $N = \frac{224 \times 224}{16 \times 16} = 196$.
2. Demonstrate how a single 2D Convolution `nn.Conv2d(in_channels=3, out_channels=D, kernel_size=16, stride=16)` is mathematically equivalent to dividing the image into non-overlapping $16 \times 16$ patches and projecting each $16 \times 16 \times 3 = 768$ vector into dimension $D$.
3. Write a PyTorch module `PatchEmbedding(nn.Module)` that prepends a learnable `[CLS]` token and adds a learnable 1D position embedding $E_{\text{pos}} \in \mathbb{R}^{1 \times (N+1) \times D}$.

#### Verification Solution
```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional projection
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token and Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # [B x C x H x W] -> [B x D x (H/P) x (W/P)] -> [B x D x N] -> [B x N x D]
        x_patches = self.proj(x).flatten(2).transpose(1, 2)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_tokens, x_patches), dim=1)
        
        # Add position embeddings
        x_out = x_tokens + self.pos_embed
        return x_out

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B = 4
    img = torch.randn(B, 3, 224, 224)
    patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=384)
    tokens = patch_embed(img)
    
    print(f"Input Image Shape:   {img.shape}")
    print(f"Token Output Shape:  {tokens.shape} (Expected: [{B}, 197, 384])")
    assert tokens.shape == (B, 197, 384)
```

---

## Tier 2: ViT Encoder & Attention Rollout Maps

### Exercise 2: Multi-Head Self-Attention with Attention Rollout Extraction
**Concepts**: Attention Rollout (Abnar & Zuidema 2020), identity matrix residual accumulation, visual interpretability.

#### Problem Specification
In deep Vision Transformers, information flows across layers via residual connections. To trace the influence of the input patches onto the final `[CLS]` token:
1. At layer $l$, let raw attention matrix be $A^{(l)} \in \mathbb{R}^{(N+1) \times (N+1)}$.
2. Account for residual connections by adding identity: $\tilde{A}^{(l)} = 0.5 A^{(l)} + 0.5 I$.
3. Normalize rows: $\bar{A}^{(l)} = \frac{\tilde{A}^{(l)}}{\sum_j \tilde{A}^{(l)}_{i, j}}$.
4. Recursively compute the cumulative rollout matrix:
   $$R^{(L)} = \bar{A}^{(L)} \bar{A}^{(L-1)} \cdots \bar{A}^{(1)}$$
5. Implement `compute_attention_rollout(all_layer_attentions)` and extract the attention attribution map of the `[CLS]` token over the $14 \times 14$ spatial patches.

#### Verification Solution
```python
import torch

def compute_attention_rollout(attentions: list[torch.Tensor]) -> torch.Tensor:
    """
    attentions: list of L tensors, each [B, Heads, N, N]
    Returns: Rollout matrix [B, N, N]
    """
    B, Heads, N, _ = attentions[0].shape
    rollout = torch.eye(N).unsqueeze(0).repeat(B, 1, 1).to(attentions[0].device)
    
    for A in attentions:
        # Average across heads
        A_mean = torch.mean(A, dim=1)  # [B, N, N]
        
        # Add residual skip connection and re-normalize
        I = torch.eye(N).unsqueeze(0).to(A.device)
        A_tilde = 0.5 * A_mean + 0.5 * I
        A_bar = A_tilde / A_tilde.sum(dim=-1, keepdim=True)
        
        rollout = torch.matmul(A_bar, rollout)
        
    return rollout

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, N = 2, 4, 197
    dummy_attentions = [torch.softmax(torch.randn(B, H, N, N), dim=-1) for _ in range(6)]
    
    rollout = compute_attention_rollout(dummy_attentions)
    cls_attribution = rollout[:, 0, 1:]  # [B, 196]
    
    print(f"Rollout Matrix Shape:     {rollout.shape}")
    print(f"CLS Attribution Map Shape: {cls_attribution.shape}")
    assert cls_attribution.shape == (B, 196)
```

---

## Tier 3: Modern ConvNeXt 7x7 Depthwise Block

### Exercise 3: Modernizing ResNet to ConvNeXt Block Mechanics
**Concepts**: Inverted bottleneck, $7 \times 7$ Depthwise Convolution, LayerNorm in vision, GELU activations.

#### Problem Specification
ConvNeXt (Liu et al. 2022) modernizes standard CNNs to mirror Vision Transformer design choices:
1. Implement a complete `ConvNeXtBlock(nn.Module)`:
   - $7 \times 7$ Depthwise Convolution (`groups=dim`).
   - Channels-last permutation for LayerNorm.
   - Pointwise Linear expansion ($4 \times \text{dim}$) with GeLU.
   - Pointwise Linear contraction back to $\text{dim}$.
   - Layer scale scaling parameter $\gamma \in \mathbb{R}^{\text{dim}}$ initialized to $10^{-6}$.
   - Residual identity addition.

#### Verification Solution
```python
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        # 7x7 Depthwise Conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Inverted bottleneck (4x expansion)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        # Permute to channels-last for LayerNorm: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        # Permute back: [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return residual + x

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B, C, H, W = 2, 64, 32, 32
    block = ConvNeXtBlock(dim=C)
    inputs = torch.randn(B, C, H, W)
    out = block(inputs)
    
    print(f"ConvNeXt Input Shape:  {inputs.shape}")
    print(f"ConvNeXt Output Shape: {out.shape}")
    assert out.shape == inputs.shape
```
