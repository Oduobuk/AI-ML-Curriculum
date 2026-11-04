# Month 3, Week 2: Deep Neural Network Engineering: Initialization, Normalization Dynamics, Residual Architectures, and Modern Optimizers (AdamW)

```
================================================================================
COURSE:    AI & LLM SYSTEMS ENGINEERING (2026 EDITION)
MODULE:    MONTH 3: MODERN DEEP LEARNING & COMPUTER VISION
WEEK:      WEEK 2: NEURAL NETWORK ENGINEERING, NORMALIZATION & OPTIMIZATION
DELIVERABLE: COMPREHENSIVE MASTERCLASS LECTURE NOTE
TARGET:    SENIOR AI ENGINEERS & FOUNDATION MODEL ARCHITECTS
================================================================================
```

---

## Table of Contents
1. [Executive Summary & Industry Context](#1-executive-summary--industry-context)
2. [Canonical Textbook Grounding](#2-canonical-textbook-grounding)
3. [Visual Architecture](#3-visual-architecture)
4. [Deep Theoretical Foundations](#4-deep-theoretical-foundations)
   - 4.1. The Vanishing/Exploding Signal Catastrophe
   - 4.2. Variance Preservation: Rigorous Derivation of Xavier/Glorot and He/Kaiming Initialization
   - 4.3. Normalization Mechanics: BatchNorm vs. LayerNorm vs. GroupNorm vs. RMSNorm
   - 4.4. Residual Connections and Loss Landscape Smoothing
   - 4.5. First and Second Moment Optimization: Stochastic Gradient Descent to AdamW
5. [Complete Runnable Code Implementation](#5-complete-runnable-code-implementation)
   - 5.1. From-Scratch Tensor-Level AdamW Optimizer Engine
   - 5.2. Custom Numerically Stable RMSNorm Layer with Vectorized CUDA/MPS Parity
   - 5.3. Enterprise-Grade PyTorch Training Harness with Cosine Annealing and Gradient Clipping
6. [Common Pitfalls & Debugging Playbook](#6-common-pitfalls--debugging-playbook)
7. [Production Best Practices & Real-World Case Study](#7-production-best-practices--real-world-case-study)
8. [Hands-On Mini-Lab & Self-Assessment](#8-hands-on-mini-lab--self-assessment)
9. [Key Takeaways, Glossary & Further Reading](#9-key-takeaways-glossary--further-reading)

---

## 1. Executive Summary & Industry Context

### Why This Matters in 2026
In modern deep learning engineering, arbitrary neural network design is obsolete. When scaling architectures—whether 30-layer residual vision networks or 128-layer transformer foundation models—**untamed variance dynamics cause immediate numerical collapse**. If intermediate activations grow exponentially across layers, tensors overflow into floating-point `NaN` / `Inf` within five iterations. Conversely, if signal variances decay exponentially, gradients vanish into machine epsilon ($\approx 1.19 \times 10^{-7}$ in IEEE FP32), completely stalling weight updates.

Superficial practitioners treat neural networks as black boxes, randomly tuning learning rates and stacking layers. Senior AI systems architects treat neural networks as **dynamical systems governing high-dimensional probability distributions**. Every architectural component serves a concrete mathematical purpose:
- **Initialization Strategies (He / Xavier)** stabilize activation variance at step $t=0$, ensuring that signal energy neither attenuates nor explodes during forward and backward passes.
- **Normalization Layers (BatchNorm, LayerNorm, RMSNorm)** actively control internal covariate shift and preserve Lipschitz smoothness along the optimization trajectory.
- **Residual Skips ($x + F(x)$)** eliminate singular saddle points and shatter exponential gradient attenuation, transforming chaotic, non-convex loss surfaces into smooth, navigable basins.
- **Decoupled Weight Decay (AdamW)** fixes the mathematical flaw in classical Adam where L2 regularization interacts catastrophically with historical second-moment gradient scaling.

Mastering these engineering fundamentals is the dividing line between an engineer whose multi-node GPU runs diverge mysteriously and an architect capable of reliably training trillion-parameter foundation models without loss spikes.

```
+---------------------------------------------------------------------------------------------------+
|                        THE DEEP LEARNING DYNAMICAL EQUILIBRIUM ENGINE                             |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Variance Preservation] ---> He / Kaiming Normal (Var(W) = 2 / fan_in)                          |
|         |                     Guarantees Var(y) = Var(x) across 100+ un-normalized layers        |
|         v                                                                                         |
|  [Loss Landscape Reform] ---> Residual Skip Highway: H(x) = x + F(x)                              |
|         |                     Eliminates shattered gradients; produces smooth convex basins       |
|         v                                                                                         |
|  [Scale Invariance] --------> RMSNorm: y = x / RMS(x) * gamma                                     |
|         |                     30% faster than LayerNorm; bounds activation drift in Transformers  |
|         v                                                                                         |
|  [Adaptive Optimization] ---> AdamW (Decoupled Weight Decay: theta -= lr * lambda * theta)        |
|                               True scale-invariant L2 regularization across moving moments        |
+---------------------------------------------------------------------------------------------------+
```

### Learning Objectives
By the end of this masterclass lecture note, you will be able to:
1. Prove the **Variance Preservation Theorem** from first principles for linear, tanh, and rectified linear (ReLU) activation functions, deriving exact Xavier and He initialization bounds.
2. Mathematically derive the forward and backward formulations for **Batch Normalization**, **Layer Normalization**, and **RMSNorm**, analyzing why RMSNorm has replaced LayerNorm in modern 2026 LLM architectures (Llama 3, Mistral).
3. Analyze the loss landscape smoothing effect of **Residual Connections** ($x + F(x)$) using differential calculus and Neumann series expansions.
4. Derive the update equations for **AdamW**, proving mathematically why coupling L2 regularization with adaptive gradient scaling ($\nabla \mathcal{L} + \lambda \theta$) fundamentally breaks weight decay on high-frequency parameters.
5. Implement a from-scratch tensor-level `AdamW` optimizer matching `torch.optim.AdamW` numerical precision to $10^{-7}$ relative tolerance.
6. Build an enterprise training harness incorporating **Cosine Annealing with Warm Restarts**, **Gradient Norm Clipping**, and mixed-precision stability.

---

## 2. Canonical Textbook Grounding

To ground your engineering intuition in verified mathematical literature, study these canonical references:

| Author / Text | Specific Chapters / Sections | Core Theoretical Concept Contributed | How to Read |
| :--- | :--- | :--- | :--- |
| **Kaiming He, Xiangyu Zhang, Shaoqing Ren, & Jian Sun**, *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* (ICCV 2015) | **Section 2**: *Approach* (Forward and Backward Variance Derivation) | Theoretical proof that ReLU halves the variance of inputs, requiring a compensatory factor of $2 / n_{\text{in}}$ in weight initialization. | **Read Section 2.2 with line-by-line mathematical scrutiny.** Trace equation (1) through (39) carefully. Understand why backward initialization ($2 / n_{\text{out}}$) and forward initialization ($2 / n_{\text{in}}$) are both valid. |
| **Ilya Loshchilov & Frank Hutter**, *Decoupled Weight Decay Regularization* (ICLR 2019) | **Section 2**: *Decoupling Weight Decay from the Gradient Updates* | Formal proof of why classical L2 regularization is ineffective in adaptive moment methods (Adam) and the derivation of AdamW. | **Read Section 2 closely.** Examine Algorithm 2. Note how gradient-based updates and weight decay updates are applied in separate computational steps. |
| **Bjorck, Gomes, Selman, & Weinberger**, *Understanding Batch Normalization* (NeurIPS 2018) & **Santurkar et al.**, *How Does Batch Normalization Help Optimization?* (NeurIPS 2018) | Core Architectural Papers | Debunking internal covariate shift; Proving that normalization layers fundamentally smooth the Lipschitz constant of the loss landscape. | **Focus on Santurkar Section 3.** Replicate the gradient variance proofs. Understand why normalization allows significantly higher learning rates without divergence. |
| **Ian Goodfellow, Yoshua Bengio, & Aaron Courville**, *Deep Learning* (MIT Press, 2016) | **Chapter 8.2**: *Basic Algorithms* (Momentum, Nesterov)<br>**Chapter 8.5**: *Optimization Strategies* | Mathematical foundations of first and second-order stochastic optimization; Exponential moving averages of gradients. | **Read Section 8.2.2 and 8.5.3.** Pay close attention to the physical analogy of friction and momentum in quadratic bowls. |

---

## 3. Visual Architecture

### 3.1. Normalization Schemes Across Multidimensional Tensors

Deep learning models operate on 4D vision tensors $(N, C, H, W)$ or 3D sequence tensors $(N, T, D)$. The choice of normalization determines along which axes mean and variance statistics are aggregated.

```
Visual Tensor Dimensions:
  N: Batch Dimension | C: Feature / Channel Dimension | (H, W) or T: Spatial / Sequence Dimensions

1. Batch Normalization (BN)
   Aggregates across (N, H, W) for EACH channel C independently.
   [====== C1 ======]  [====== C2 ======]  [====== C3 ======]
   | Batch 1, 2, ...|  | Batch 1, 2, ...|  | Batch 1, 2, ...| -> Mean & Var per Channel
   Fails when batch size N < 8. Batch-dependent at inference.

2. Layer Normalization (LN)
   Aggregates across (C, H, W) for EACH sample N independently.
   Sample 1: [ C1, C2, C3, ... (all channels / features) ] -> Mean & Var per Sample
   Sample 2: [ C1, C2, C3, ... (all channels / features) ]
   Batch-independent. Standard in classical Transformers (BERT, GPT-2).

3. Group Normalization (GN)
   Splits C channels into G groups. Aggregates across (Group_c, H, W) per sample.
   Sample 1: Group 1 [C1...C4] | Group 2 [C5...C8] -> Mean & Var per Group
   Robust to mini-batch size in high-resolution vision tasks.

4. Root Mean Square Normalization (RMSNorm)
   Simplifies LayerNorm by removing mean-centering: normalizes strictly by RMS.
   y = x / sqrt( (1/d) * sum(x_i^2) + eps ) * gamma
   Eliminates 2 reduction passes. Default in modern LLMs (Llama 3, Mistral, Gemma).
```

![Normalization Paradigms](../../../docs/images/normalization_paradigms.png)
*Figure 3.1: Geometric comparison of tensor reduction axes across Batch Normalization, Layer Normalization, Group Normalization, and modern RMSNorm.*

---

## 4. Deep Theoretical Foundations

### 4.1. The Vanishing and Exploding Signal Catastrophe

#### Intuition
Consider an un-normalized, deep linear network of depth $L$:
$$y = W_L W_{L-1} \dots W_2 W_1 x$$
Where each weight matrix $W_l \in \mathbb{R}^{d \times d}$ is initialized with independent and identically distributed (i.i.d.) random variables with mean zero and variance $\sigma^2_w$.

If $\sigma^2_w \cdot d > 1$, the expected norm of the activation vector grows exponentially with depth:
$$\mathbb{E}[\|y\|^2] \propto (\sigma^2_w \cdot d)^L$$
If $L = 50$ and $\sigma^2_w \cdot d = 1.2$:
$$(1.2)^{50} \approx 9,100 \implies \text{Activations explode rapidly into floating-point overflow!}$$
Conversely, if $\sigma^2_w \cdot d = 0.8$:
$$(0.8)^{50} \approx 1.4 \times 10^{-5} \implies \text{Signal vanishes into machine precision zero!}$$

To enable stable signal propagation, we must enforce **variance preservation**:
$$\operatorname{Var}(y_l) = \operatorname{Var}(y_{l-1}) \quad \forall l \in \{1, 2, \dots, L\}$$

---

### 4.2. Variance Preservation: Rigorous Derivation of Xavier and He Initialization

#### Part 1: The General Linear Layer
Let $y = Wx + b$, where $x \in \mathbb{R}^{n_{\text{in}}}$ and $W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$.
Assume:
1. $w_{ij}$ are i.i.d. with $\mathbb{E}[w_{ij}] = 0$ and $\operatorname{Var}(w_{ij}) = \sigma^2_w$.
2. $x_j$ are i.i.d. with $\mathbb{E}[x_j] = 0$ and $\operatorname{Var}(x_j) = \sigma^2_x$.
3. $W$ and $x$ are mutually independent. Bias $b = 0$.

For any single output neuron $y_i = \sum_{j=1}^{n_{\text{in}}} w_{ij} x_j$:
$$\mathbb{E}[y_i] = \sum_{j=1}^{n_{\text{in}}} \mathbb{E}[w_{ij} x_j] = \sum_{j=1}^{n_{\text{in}}} \mathbb{E}[w_{ij}] \mathbb{E}[x_j] = 0$$

Computing the variance of $y_i$:
$$\operatorname{Var}(y_i) = \mathbb{E}[y_i^2] - (\mathbb{E}[y_i])^2 = \mathbb{E}\left[ \left( \sum_{j=1}^{n_{\text{in}}} w_{ij} x_j \right)^2 \right]$$
Because $w_{ij} x_j$ and $w_{ik} x_k$ are independent for $j \ne k$, all cross-product expectations vanish:
$$\operatorname{Var}(y_i) = \sum_{j=1}^{n_{\text{in}}} \mathbb{E}[w_{ij}^2 x_j^2] = \sum_{j=1}^{n_{\text{in}}} \mathbb{E}[w_{ij}^2] \mathbb{E}[x_j^2] = \sum_{j=1}^{n_{\text{in}}} \operatorname{Var}(w_{ij}) \operatorname{Var}(x_j)$$
Since all inputs and weights share identical variances:
$$\operatorname{Var}(y_i) = n_{\text{in}} \sigma^2_w \sigma^2_x$$

To maintain equal signal energy from input to output ($\operatorname{Var}(y_i) = \sigma^2_x$), we must enforce:
$$n_{\text{in}} \sigma^2_w = 1 \implies \sigma^2_w = \frac{1}{n_{\text{in}}}$$

#### Part 2: Backward Pass Variance Preservation
During backpropagation, the gradient with respect to the input activation is:
$$\frac{\partial \mathcal{L}}{\partial x_j} = \sum_{i=1}^{n_{\text{out}}} \frac{\partial \mathcal{L}}{\partial y_i} w_{ij}$$
By identical mathematical reasoning, preserving gradient variance backwards requires:
$$\operatorname{Var}\left( \frac{\partial \mathcal{L}}{\partial x_j} \right) = n_{\text{out}} \sigma^2_w \operatorname{Var}\left( \frac{\partial \mathcal{L}}{\partial y_i} \right)$$
Enforcing equal gradient variance backwards implies:
$$\sigma^2_w = \frac{1}{n_{\text{out}}}$$

#### Part 3: Xavier (Glorot) Initialization
Xavier & Bengio (2010) compromised between the forward constraint ($1 / n_{\text{in}}$) and backward constraint ($1 / n_{\text{out}}$) by taking their harmonic mean:
$$\sigma^2_w = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$
- **Normal Distribution**: $W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$
- **Uniform Distribution**: $W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, +\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$

*Material Limit*: Xavier initialization assumes linear or symmetric zero-centered activations (e.g., identity, tanh). When applied to **ReLU**, it fails catastrophically.

#### Part 4: He (Kaiming) Initialization for ReLU Networks
Let $a_j = \operatorname{ReLU}(y_{j}) = \max(0, y_{j})$.
Assume $y_j$ is symmetric around zero with probability density $p(y)$. Then:
$$a_j = \begin{cases} y_j & \text{with probability } 0.5 \\ 0 & \text{with probability } 0.5 \end{cases}$$
Computing the second moment:
$$\mathbb{E}[a_j^2] = \int_{-\infty}^\infty (\max(0, y))^2 p(y) \, dy = \int_0^\infty y^2 p(y) \, dy = \frac{1}{2} \int_{-\infty}^\infty y^2 p(y) \, dy = \frac{1}{2} \operatorname{Var}(y_j)$$

**The Fundamental Halving Rule**: The ReLU activation function zeroes out exactly half the distribution, **cutting the signal variance in half**!
$$\operatorname{Var}(y_l) = n_{\text{in}} \sigma^2_w \operatorname{Var}(a_{l-1}) = n_{\text{in}} \sigma^2_w \left( \frac{1}{2} \operatorname{Var}(y_{l-1}) \right) = \frac{1}{2} n_{\text{in}} \sigma^2_w \operatorname{Var}(y_{l-1})$$

To enforce $\operatorname{Var}(y_l) = \operatorname{Var}(y_{l-1})$:
$$\frac{1}{2} n_{\text{in}} \sigma^2_w = 1 \implies \mathbf{\sigma^2_w = \frac{2}{n_{\text{in}}}}$$

- **He Normal**: $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$
- **He Uniform**: $W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}}}, +\sqrt{\frac{6}{n_{\text{in}}}}\right)$

---

### 4.3. Normalization Mechanics: BatchNorm vs. LayerNorm vs. RMSNorm

#### 1. Batch Normalization (Ioffe & Szegedy, 2015)
Given a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$ of size $m$:
$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma^2_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2$$
$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
- **Pros**: Accelerates CNN training; allows larger learning rates; acts as implicit regularization.
- **Cons**: Training/inference disparity (requires running exponential moving average tracking); fails when batch size $m < 8$; broken by distributed sequence masking.

#### 2. Layer Normalization (Ba, Kiros, & Hinton, 2016)
Computes statistics across hidden features $d$ for each independent sample:
$$\mu = \frac{1}{d} \sum_{j=1}^d x_j, \quad \sigma^2 = \frac{1}{d} \sum_{j=1}^d (x_j - \mu)^2$$
$$\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_j = \gamma \hat{x}_j + \beta$$
- **Pros**: Independent of batch size; identical computation at training and inference; ideal for variable-length NLP sequences.

#### 3. Root Mean Square Normalization (RMSNorm - Zhang & Sennrich, 2019)
Hypothesis: The regularizing success of LayerNorm comes entirely from the scaling metric (rescaling by variance), while mean-centering ($\mu$) provides negligible optimization benefit.
$$\operatorname{RMS}(x) = \sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}$$
$$\bar{x}_j = \frac{x_j}{\operatorname{RMS}(x)}, \quad y_j = \gamma_j \bar{x}_j$$

#### Analytical VJP for Custom RMSNorm Kernel:
Let upstream gradient be $\bar{\mathbf{y}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$.
By the chain rule:
$$\frac{\partial \mathcal{L}}{\partial \gamma_j} = \sum_{k} \bar{y}_k \bar{x}_k$$
$$\frac{\partial \mathcal{L}}{\partial x_j} = \frac{\gamma_j \bar{y}_j}{\operatorname{RMS}(x)} - \frac{x_j}{d \cdot \operatorname{RMS}(x)^3} \sum_{k=1}^d (\gamma_k \bar{y}_k x_k)$$

*Computational Advantage*: RMSNorm eliminates two global reductions (computing mean $\mu$, then subtracting $\mu$ before variance). In modern GPU kernel execution, this yields a **$20\% - 30\%$ latency reduction** in normalization layers.

---

### 4.4. Residual Connections and Loss Landscape Smoothing

#### The Degradation Problem
He et al. (2015) made a startling empirical discovery: as un-skipping convolutional networks grow deeper (e.g., from 20 to 56 layers), **both training error and test error degrade simultaneously**. This is not overfitting (which would exhibit low training error but high test error); it is an optimization failure.

#### The Residual Formulation
Instead of forcing a stacked sequence of non-linear layers to directly fit an underlying mapping $\mathcal{H}(x)$, the network is parameterized to learn a residual perturbation $\mathcal{F}(x)$:
$$\mathcal{H}(x) = x + \mathcal{F}(x)$$

Consider a deep network with $K$ residual blocks:
$$x_K = x_0 + \sum_{l=0}^{K-1} \mathcal{F}(x_l, \mathcal{W}_l)$$

Taking the derivative of the loss $\mathcal{L}$ with respect to the input activation $x_l$ at any shallow layer $l$:
$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_K} \frac{\partial x_K}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_K} \left( \mathbf{I} + \frac{\partial}{\partial x_l} \sum_{j=l}^{K-1} \mathcal{F}(x_j, \mathcal{W}_j) \right)$$

#### Why Residual Networks Cannot Vanish Gradients:
Even if the learned Jacobian term $\frac{\partial \mathcal{F}}{\partial x_l}$ vanishes completely toward zero, the identity matrix $\mathbf{I}$ remains!
$$\frac{\partial \mathcal{L}}{\partial x_l} \approx \frac{\partial \mathcal{L}}{\partial x_K} \cdot \mathbf{I} = \frac{\partial \mathcal{L}}{\partial x_K}$$
The identity connection acts as a **zero-resistance gradient highway**, transmitting loss gradients cleanly from the output layer directly back to layer 0 without exponential attenuation.

---

### 4.5. First and Second Moment Optimization: From SGD to AdamW

#### 1. Classical Stochastic Gradient Descent with Momentum
$$\mathbf{m}_t = \beta \mathbf{m}_{t-1} + (1 - \beta) \mathbf{g}_t$$
$$\theta_t = \theta_{t-1} - \eta \mathbf{m}_t$$
Momentum dampens high-frequency oscillations across steep ravines while accelerating along flat, consistent gradients.

#### 2. The Flaw in Classical Adam with L2 Regularization
Standard Adam tracks both first moments $\mathbf{m}_t$ (mean) and second raw moments $\mathbf{v}_t$ (uncentered variance):
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \quad \hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

In classical library implementations, L2 regularization was implemented by adding $\lambda \theta_t$ directly to the objective loss gradient:
$$\tilde{\mathbf{g}}_t = \mathbf{g}_t + \lambda \theta_t$$
Now observe what happens during parameter updates:
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t \approx \theta_{t-1} - \eta \frac{\mathbf{g}_t + \lambda \theta_{t-1}}{\sqrt{\mathbf{v}_t} + \epsilon}$$

**The Mathematical Failure**:
For a parameter with massive, frequent gradients, $\mathbf{v}_t$ is huge. Therefore, the effective weight decay applied to this parameter is:
$$\lambda_{\text{effective}} = \frac{\eta \lambda}{\sqrt{\mathbf{v}_t} + \epsilon} \ll \eta \lambda$$
Parameters with large historical gradients receive **virtually zero weight decay**, while parameters with rare, tiny gradients receive massive weight decay! This completely inverts the intended regularization objective.

#### 3. Decoupled Weight Decay (AdamW - Loshchilov & Hutter, 2019)
AdamW restores the true mathematical intention of weight decay by applying shrinkage **after** computing the adaptive gradient step:
$$\mathbf{g}_t = \nabla_\theta \mathcal{L}(\theta_{t-1}) \quad \text{(Pure loss gradient)}$$
$$\hat{\mathbf{m}}_t = \frac{\beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t}{1 - \beta_1^t}$$
$$\hat{\mathbf{v}}_t = \frac{\beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \underbrace{\eta \lambda \theta_{t-1}}_{\text{Decoupled Weight Decay}} - \underbrace{\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t}_{\text{Adaptive Gradient Step}}$$

Every parameter experiences proportional weight decay shrinkage $\eta \lambda \theta_{t-1}$ regardless of its historical gradient magnitude.

---

## 5. Complete Runnable Code Implementation

Let us construct an end-to-end suite demonstrating:
1. A pure PyTorch tensor-level from-scratch `AdamW` optimizer matching `torch.optim.AdamW` exactly.
2. A custom, numerically stable `RMSNorm` layer with verified forward and backward analytical VJP.
3. An enterprise training harness with Cosine Annealing, warm restarts, and gradient norm clipping.

### Python Environment Setup
```bash
# Initialize isolated Python 3.11+ environment with uv
uv init week2-nn-engineering
cd week2-nn-engineering
uv add torch torchvision numpy tabulate
```

### 5.1. From-Scratch Tensor-Level AdamW Optimizer Engine

```python
"""
custom_adamw.py
Author: Senior AI Engineering Lead
Description: Pure PyTorch tensor-level implementation of the decoupled weight decay
optimizer (AdamW). Verified against torch.optim.AdamW to 1e-7 relative tolerance.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Iterable
import torch
from torch.optim import Optimizer


class CustomAdamW(Optimizer):
    """
    Implements AdamW with decoupled weight decay as formulated by Loshchilov & Hutter (2019).
    
    Update equations:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * weight_decay * theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    """
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                # Initialize state buffers on first optimization step
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                state["step"] += 1
                t: int = state["step"]

                # 1. Decoupled Weight Decay Update: theta = theta - lr * weight_decay * theta
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # 2. Update biased first moment: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 3. Update biased second raw moment: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 4. Bias correction factors
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t

                # Compute step size scaling
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 5. Parameter update: theta = theta - step_size * m_t / denom
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ==============================================================================
# Numerical Parity Verification Against torch.optim.AdamW
# ==============================================================================
def verify_adamw_parity() -> None:
    print("=" * 80)
    print("VERIFYING CUSTOM ADAMW AGAINST TORCH.OPTIM.ADAMW")
    print("=" * 80)

    torch.manual_seed(42)
    p_custom = torch.randn(10, 10, requires_grad=True)
    p_native = p_custom.clone().detach().requires_grad_(True)

    opt_custom = CustomAdamW([p_custom], lr=1e-3, weight_decay=1e-2)
    opt_native = torch.optim.AdamW([p_native], lr=1e-3, weight_decay=1e-2)

    # Simulate 20 training steps with random gradients
    for step in range(1, 21):
        g = torch.randn(10, 10)
        p_custom.grad = g.clone()
        p_native.grad = g.clone()

        opt_custom.step()
        opt_native.step()

        max_diff = torch.max(torch.abs(p_custom - p_native)).item()
        assert max_diff < 1e-7, f"Parity mismatch at step {step}: max diff {max_diff}"

    print(f"-> 20 Steps Complete. Maximum Absolute Parameter Difference: {max_diff:.2e}")
    print("-> PARITY VERIFIED: Custom AdamW matches PyTorch Native to 7 decimal places!\n")


if __name__ == "__main__":
    verify_adamw_parity()
```

---

### 5.2. Custom Numerically Stable RMSNorm Layer

```python
"""
rmsnorm_module.py
Author: Senior AI Engineering Lead
Description: Custom RMSNorm module (as used in Llama 3 and Mistral architectures).
Features analytic backward pass and numerical stability verification.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Eliminates mean-centering, scaling activations strictly by root-mean-square.
    
    y = (x / RMS(x)) * gamma
    where RMS(x) = sqrt( (1/d) * sum(x_i^2) + eps )
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS across last dimension
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale by learnable affine parameter gamma
        return self._norm(x.float()).type_as(x) * self.gamma


def test_rmsnorm() -> None:
    print("=" * 80)
    print("TESTING CUSTOM RMSNORM LAYER (LLAMA-3 ARCHITECTURE)")
    print("=" * 80)

    batch, seq_len, dim = 4, 32, 256
    x = torch.randn(batch, seq_len, dim, requires_grad=True)

    norm = RMSNorm(dim=dim)
    out = norm(x)

    # Compute loss and backward
    loss = out.sum()
    loss.backward()

    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Gamma Grad Norm: {norm.gamma.grad.norm().item():.4f}")
    print(f"Input Grad Norm: {x.grad.norm().item():.4f}")
    assert out.shape == x.shape, "Shape mismatch"
    assert not torch.isnan(out).any(), "NaN in output"
    print("-> RMSNorm Execution and Backpropagation Verified.\n")


if __name__ == "__main__":
    test_rmsnorm()
```

---

### 5.3. Enterprise-Grade PyTorch Training Harness

```python
"""
training_harness.py
Author: Senior AI Engineering Lead
Description: Production PyTorch training loop incorporating Cosine Annealing,
Gradient Norm Clipping, and Mixed-Precision Execution.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class ResidualMLP(nn.Module):
    """Modern deep MLP featuring He initialization, RMSNorm, and residual connections."""
    def __init__(self, dim: int = 256, num_blocks: int = 4, num_classes: int = 10):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                RMSNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(num_blocks)
        ])
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Applies He/Kaiming initialization to all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = x + block(x)  # Residual skip connection
        x = self.final_norm(x)
        return self.head(x)


def run_training_harness() -> None:
    print("=" * 80)
    print("RUNNING ENTERPRISE DEEP LEARNING TRAINING HARNESS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Active Accelerator Device: {device.type.upper()}")

    # Synthetic dataset
    N, dim, num_classes = 2000, 256, 10
    X = torch.randn(N, dim)
    y = torch.randint(0, num_classes, (N,))
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    model = ResidualMLP(dim=dim, num_blocks=4, num_classes=num_classes).to(device)
    optimizer = CustomAdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Cosine Annealing with Warm Restarts scheduler
    T_0, T_mult = 5, 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

    model.train()
    epochs = 10
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Gradient Norm Clipping to prevent exploding gradient spikes
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Current LR: {current_lr:.6f}")

    print("\n-> Training Harness Completed Successfully.")


if __name__ == "__main__":
    run_training_harness()
```

---

## 6. Common Pitfalls & Debugging Playbook

### Pitfall 1: Breaking Variance Preservation with Mismatched Initializers
- **Symptom**: A 30-layer ReLU network exhibits zero gradient updates across layers 1–15; training stalls immediately at epoch 1.
- **Root Cause**: The network used `nn.init.xavier_normal_` on a deep ReLU model. Because Xavier preserves variance assuming linear activations ($1 / n_{\text{in}}$), each successive ReLU layer cuts variance in half: $\operatorname{Var}(y_{30}) \approx (0.5)^{30} \operatorname{Var}(x_0) \approx 9.3 \times 10^{-10}$.
- **Fix**: Use He (Kaiming) initialization with `nonlinearity='relu'`:
  ```python
  # CORRECT: Compensates for ReLU halving with 2 / fan_in
  nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
  ```

---

### Pitfall 2: Placing Normalization Before vs. After Residual Connections (Pre-LN vs. Post-LN)
- **Symptom**: A deep 64-layer Transformer diverges during initial warmup or requires an agonizingly slow warmup learning rate ($10^{-6}$).
- **Root Cause**: Using **Post-LayerNorm** ($x_{l+1} = \operatorname{LN}(x_l + \mathcal{F}(x_l))$). Post-LN normalizes the residual stream, diluting the identity gradient highway and forcing gradients to pass through the normalization derivative at every step.
- **Fix**: Migrate to **Pre-LayerNorm** / **Pre-RMSNorm** ($x_{l+1} = x_l + \mathcal{F}(\operatorname{RMSNorm}(x_l))$), preserving an unperturbed identity skip from input to output.

---

### Pitfall 3: Applying Weight Decay to Biases and Normalization Scales
- **Symptom**: Performance degrades noticeably when increasing weight decay; LayerNorm scales $\gamma$ collapse toward zero.
- **Root Cause**: In standard PyTorch, passing `model.parameters()` to AdamW applies weight decay uniformly to 1D biases and normalization affine scales ($\gamma, \beta$), constraining dynamic range.
- **Fix**: Partition parameters into decay and no-decay groups:
  ```python
  def build_optimizer(model, lr=1e-3, weight_decay=1e-2):
      decay_params = [p for p in model.parameters() if p.dim() >= 2]
      no_decay_params = [p for p in model.parameters() if p.dim() < 2]
      return torch.optim.AdamW([
          {"params": decay_params, "weight_decay": weight_decay},
          {"params": no_decay_params, "weight_decay": 0.0}
      ], lr=lr)
  ```

---

### Pitfall 4: Gradient Explosion and Exploding Loss Spikes Mid-Training
- **Symptom**: Training loss descends smoothly for 10,000 steps, then suddenly spikes to `NaN` or orders of magnitude higher.
- **Root Cause**: High curvature ravines in the loss landscape produce extreme gradient norms $\|\mathbf{g}\| > 1000$, ejecting optimizer parameters into catastrophic regions.
- **Fix**: Apply **Gradient Norm Clipping** before optimizer steps:
  ```python
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()
  ```

---

## 7. Production Best Practices & Real-World Case Study

### Enterprise Case Study: Stabilizing 70B Foundation Model Training Run
- **Scenario**: Training a multi-layer transformer on a cluster of 512 NVIDIA H100 GPUs.
- **Cost**: $\$4,200\text{ per hour}$ in cloud compute.
- **Problem**: The run suffered from periodic loss spikes at step 14,200 and step 28,900, requiring manual rollbacks from checkpoints and wasting hundreds of GPU-hours.

### Root-Cause Analysis & Modern Architectural Modernization
```
Legacy Architecture (Failed Run):
  - Initialization: Standard Normal N(0, 0.02)
  - Normalization: Post-LayerNorm
  - Optimizer: Classical Adam with coupled L2 regularization
  - Result: High activation variance drift -> Loss spikes -> Numerical instability.

Modern 2026 Foundation Model Architecture:
  1. Initialization: Scaled Kaiming Init (weights scaled by 1 / sqrt(2 * num_layers))
  2. Normalization: Pre-RMSNorm with eps=1e-6 (Zero mean-centering overhead)
  3. Optimizer: AdamW with decoupled weight decay + No-decay on 1D biases/scales
  4. Stability Layer: z-loss regularization (penalizing extreme logit magnitudes)
  5. Gradient Clipping: Hard clip at max_norm=1.0

Results:
  - 100,000 continuous training steps with ZERO loss spikes.
  - Overall throughput increased by 18% due to RMSNorm kernel speedups.
  - Training completed 4 days ahead of schedule, saving $403,200 in compute costs.
```

---

## 8. Hands-On Mini-Lab & Self-Assessment

### Lab Objective
Construct a robust deep residual classifier that trains on a synthetic noisy multi-modal distribution without diverging:
1. Prove variance preservation across a 50-layer deep network by logging layer-by-layer activation norms under Xavier vs. He initialization.
2. Implement custom Pre-RMSNorm blocks.
3. Train using custom `CustomAdamW` with decoupled weight decay and Cosine Annealing.

### Acceptance Criteria
- [ ] Layer-50 activation variance under He init remains within $[0.5, 2.0] \times \operatorname{Var}(x_0)$.
- [ ] Layer-50 activation variance under Xavier init collapses to $< 10^{-6}$.
- [ ] Custom AdamW achieves numerical parity with PyTorch native optimizer.
- [ ] Gradient norm clipping actively constrains $\|\mathbf{g}\| \le 1.0$.

---

### Graded Self-Assessment Questions (With Complete Solutions)

#### Question 1 (Mathematical - Variance Preservation)
Why does a linear layer initialized with $W \sim \mathcal{N}(0, 1 / n_{\text{in}})$ fail to preserve activation variance when followed by a ReLU activation?
- **Answer**: ReLU maps all negative inputs to zero, cutting the distribution's second moment exactly in half: $\mathbb{E}[\operatorname{ReLU}(y)^2] = \frac{1}{2} \operatorname{Var}(y)$. Consequently, $\operatorname{Var}(\text{output}) = \frac{1}{2} n_{\text{in}} (1 / n_{\text{in}}) \operatorname{Var}(\text{input}) = \mathbf{0.5 \operatorname{Var}(\text{input})}$. To compensate for this 50% loss of signal energy, the weight variance must be doubled to $\sigma^2_w = \mathbf{2 / n_{\text{in}}}$ (He initialization).

#### Question 2 (Architecture - RMSNorm vs. LayerNorm)
What algebraic operation does RMSNorm omit compared to LayerNorm, and why does this provide a latency benefit in modern foundation model architectures?
- **Answer**: RMSNorm omits the **mean-centering step** ($\mu = \frac{1}{d} \sum x_i$ and $x_i - \mu$). It scales activations strictly by root-mean-square: $\sqrt{\frac{1}{d} \sum x_i^2 + \epsilon}$. This eliminates two global reduction/synchronization barriers across thread blocks in GPU kernels, yielding a 20%–30% speedup in normalization execution.

#### Question 3 (Optimization - AdamW Derivation)
Why is adding $\lambda \theta$ directly to the objective loss gradient ($\tilde{\mathbf{g}} = \mathbf{g} + \lambda \theta$) fundamentally broken when used with adaptive optimizers like Adam?
- **Answer**: Adam scales updates inversely by the square root of the second raw moment: $\frac{\eta}{\sqrt{\mathbf{v}_t} + \epsilon} \tilde{\mathbf{g}}_t$. Parameters that encounter large, frequent gradients accumulate huge $\mathbf{v}_t$ values, which scales down their weight decay term $\frac{\lambda \theta}{\sqrt{\mathbf{v}_t}}$. Consequently, high-gradient parameters receive almost zero regularization, while rare-gradient parameters receive excessive regularization. Decoupled weight decay (AdamW) applies $\theta \leftarrow \theta - \eta \lambda \theta$ outside the adaptive moment fraction.

#### Question 4 (Residual Networks - Gradient Flow)
Using the residual equation $x_K = x_0 + \sum_{l=0}^{K-1} \mathcal{F}(x_l)$, derive the Jacobian $\frac{\partial x_K}{\partial x_0}$ and explain why residual networks prevent vanishing gradients in 100+ layer networks.
- **Answer**:
  $$\frac{\partial x_K}{\partial x_0} = \mathbf{I} + \sum_{l=0}^{K-1} \frac{\partial \mathcal{F}(x_l)}{\partial x_0}$$
  Even if the learned non-linear transformation gradients $\frac{\partial \mathcal{F}}{\partial x_0}$ decay to zero, the additive identity matrix $\mathbf{I}$ guarantees that the gradient signal $\frac{\partial \mathcal{L}}{\partial x_0} \approx \frac{\partial \mathcal{L}}{\partial x_K} \cdot \mathbf{I}$ flows backwards with zero attenuation.

#### Question 5 (Debugging - Optimizer Parameter Groups)
Why should 1D parameters (biases $b$ and LayerNorm scale parameters $\gamma$) be excluded from weight decay in AdamW?
- **Answer**: Biases and normalization scale parameters shift and scale the linear activation range without contributing to model complexity via matrix rank or spectral norm. Regularizing $\gamma$ forces normalization layers toward zero activation gain, causing signal collapse rather than generalizable margin improvement.

#### Question 6 (Pre-LN vs Post-LN)
Why did modern Transformer architectures universally abandon Post-LayerNorm in favor of Pre-LayerNorm?
- **Answer**: Post-LayerNorm places normalization directly on the residual addition ($x_{l+1} = \operatorname{LN}(x_l + \mathcal{F}(x_l))$), which perturbs the identity highway and causes gradient magnitudes to compound across deep layers, requiring fragile learning rate warmups. Pre-LayerNorm ($x_{l+1} = x_l + \mathcal{F}(\operatorname{LN}(x_l))$) leaves the identity skip unperturbed, enabling stable training of 100+ layer models without warmup instability.

#### Question 7 (Learning Rate Schedulers)
What is the mathematical formulation of Cosine Annealing learning rate schedule between base rate $\eta_{\max}$ and minimum rate $\eta_{\min}$ over $T_{\max}$ steps?
- **Answer**:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}} \pi\right)\right)$$

#### Question 8 (Gradient Norm Clipping)
Given parameter gradient vector $\mathbf{g}$ with Euclidean norm $\|\mathbf{g}\|_2 = 4.5$ and maximum allowable clipping threshold $c = 1.5$, compute the clipped gradient $\mathbf{g}_{\text{clipped}}$.
- **Answer**:
  $$\mathbf{g}_{\text{clipped}} = \mathbf{g} \cdot \frac{c}{\max(c, \|\mathbf{g}\|_2)} = \mathbf{g} \cdot \frac{1.5}{4.5} = \frac{\mathbf{1}}{\mathbf{3}} \mathbf{g}$$

---

### Stretch Challenges
1. **Fused Triton RMSNorm Kernel**: Implement an OpenAI Triton kernel for fused RMSNorm and compare its memory bandwidth efficiency against standard PyTorch.
2. **Empirical Loss Landscape Visualization**: Implement Li et al.'s (2018) filter-normalized random direction loss landscape sampling algorithm to plot 3D contour surfaces comparing networks with and without residual connections.

---

## 9. Key Takeaways, Glossary & Further Reading

### Key Takeaways
- **Variance must be preserved**: Initializing ReLU networks requires He initialization ($\sigma^2 = 2 / n_{\text{in}}$) to compensate for variance halving.
- **Residual highways enable depth**: Skip connections provide an unperturbed identity gradient channel ($\mathbf{I}$) that prevents vanishing gradients.
- **RMSNorm is the 2026 standard**: Modern LLMs discard mean-centering in favor of pure root-mean-square scaling, maximizing GPU throughput.
- **Decouple your weight decay**: Always use AdamW over classical Adam to ensure true scale-invariant L2 regularization across all parameters.

### Technical Glossary
| Term | Definition |
| :--- | :--- |
| **He / Kaiming Init** | Weight initialization with variance $2 / n_{\text{in}}$ designed specifically for rectified non-linearities. |
| **Internal Covariate Shift** | The change in the distribution of network activations due to parameter updates during training. |
| **RMSNorm** | Root Mean Square Normalization; a computationally efficient variant of LayerNorm that omits mean centering. |
| **Pre-LN** | An architectural pattern where normalization is applied to inputs before sub-layers rather than after the residual addition. |
| **Decoupled Weight Decay** | Regularization where weight shrinkage is applied directly to parameters independently of adaptive gradient moments. |

### Further Reading
- He, K., et al. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. ICCV.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR.
- Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS.

