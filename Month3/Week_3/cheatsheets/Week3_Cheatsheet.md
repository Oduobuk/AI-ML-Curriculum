# Month 3, Week 3 Cheatsheet: Modern Computer Vision, ConvNeXt & Vision Transformers (ViT)

---

## 1. Vision Transformer (ViT) Patch Tokenization Geometry

Decomposes an RGB image $X \in \mathbb{R}^{H \times W \times C}$ into a sequence of non-overlapping patches:

```
               Image Patch Projection Pipeline
 ┌─────────────────────────┐
 │ 224x224 RGB Image (C=3) │
 └───────────┬─────────────┘
             │
   Conv2d(k=16, s=16) -> Patch Size P=16
             │
             ▼
 ┌─────────────────────────┐
 │ 196 Patches of Dim D    │  N = (224/16) * (224/16) = 196
 └───────────┬─────────────┘
             │ Prepend [CLS] Token (index 0)
             ▼
 ┌─────────────────────────┐
 │ 197 Sequence Tokens     │  Shape: [B, 197, D]
 └───────────┬─────────────┘
             │ Add 1D Learnable Positional Embeddings
             ▼
      Transformer Encoder Stack
```

---

## 2. Multi-Head Self-Attention on Visual Tokens

For token sequence $Z \in \mathbb{R}^{N \times D}$:
$$Q = Z W_Q, \quad K = Z W_K, \quad V = Z W_V$$
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- **Pre-LayerNorm Residual Block**:
  $$z'_l = \text{MHSA}(\text{LayerNorm}(z_{l-1})) + z_{l-1}$$
  $$z_l = \text{MLP}(\text{LayerNorm}(z'_l)) + z'_l$$

---

## 3. Attention Rollout Interpretability

Traces information flow from input patches to the final `[CLS]` token across $L$ layers:
$$R^{(L)} = \bar{A}^{(L)} \bar{A}^{(L-1)} \cdots \bar{A}^{(1)} \quad \text{where } \bar{A}^{(l)} = 0.5 A^{(l)} + 0.5 I$$
- Row 0 of $R^{(L)}$ (excluding `[CLS]` self-token) yields the 196-element attention heatmap.
- Reshaped to $14 \times 14$ and bicubic upsampled to $224 \times 224$ to overlay over input pixels.

---

## 4. Modern ConvNeXt Architecture (Liu et al. 2022)

Adopts Transformer design principles into convolutional networks:
- **Large Kernel Depthwise Conv**: $7 \times 7$ depthwise convolution (`groups=C`).
- **Inverted Bottleneck**: Pointwise $1 \times 1$ conv expanding channels $4 \times$.
- **Fewer Normalizations & Activations**: Single LayerNorm and single GeLU activation per block.
- **Layer Scale**: Per-channel scaling factor $\gamma$ initialized to $10^{-6}$.

```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = (self.gamma * x).permute(0, 3, 1, 2)
        return res + x
```
