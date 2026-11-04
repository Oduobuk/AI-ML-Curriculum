# Week 4: Graduate-Level Problem Set — Real-Time Computer Vision, YOLOv8/v11 Architectures & Bounding Box Regression

Welcome to the Month 3 Week 4 Problem Set. This suite evaluates your mastery of real-time object detection systems: Anchor-Free detection heads, Intersection over Union metrics (IoU, GIoU, CIoU), Task-Aligned Assigners (TAL), vectorized Greedy Non-Maximum Suppression (NMS), and INT8 quantization for edge deployment.

---

## Tier 1: Core Bounding Box Metrics & Loss Formulations

### Exercise 1: Generalized (GIoU) vs Complete (CIoU) Loss Derivation
**Concepts**: Bounding box coordinate parametrization $(x_1, y_1, x_2, y_2)$, aspect ratio consistency $v$, trade-off parameter $\alpha$.

#### Problem Specification
Standard IoU $\frac{|A \cap B|}{|A \cup B|}$ yields zero gradient whenever two bounding boxes do not overlap, causing gradient descent to stall:
1. Formulate Generalized IoU (GIoU):
   $$\mathcal{L}_{\text{GIoU}} = 1 - \text{IoU} + \frac{|C \setminus (A \cup B)|}{|C|}$$
   where $C$ is the smallest enclosing convex bounding box.
2. Complete IoU (CIoU, Zheng et al. 2020) incorporates overlapping area, central point distance, and aspect ratio:
   $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$
   where $\rho$ is Euclidean distance between box centroids, $c$ is the diagonal length of the smallest enclosing box, and:
   $$v = \frac{4}{\pi^2} \left(\arctan\frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan\frac{w}{h}\right)^2, \quad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$
3. Implement `compute_ciou_loss(pred_boxes, target_boxes)` in vectorized PyTorch and prove that $\mathcal{L}_{\text{CIoU}} \to 0$ when predictions converge to ground truth.

#### Verification Solution
```python
import torch
import math

def compute_ciou_loss(b1: torch.Tensor, b2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    b1, b2: [N, 4] formatted as (x1, y1, x2, y2)
    """
    # 1. Intersection coordinates
    x1_i = torch.max(b1[:, 0], b2[:, 0])
    y1_i = torch.max(b1[:, 1], b2[:, 1])
    x2_i = torch.min(b1[:, 2], b2[:, 2])
    y2_i = torch.min(b1[:, 3], b2[:, 3])
    
    inter_area = torch.clamp(x2_i - x1_i, min=0) * torch.clamp(y2_i - y1_i, min=0)
    
    # 2. Areas
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union_area = area1 + area2 - inter_area + eps
    iou = inter_area / union_area
    
    # 3. Smallest enclosing box
    x1_c = torch.min(b1[:, 0], b2[:, 0])
    y1_c = torch.min(b1[:, 1], b2[:, 1])
    x2_c = torch.max(b1[:, 2], b2[:, 2])
    y2_c = torch.max(b1[:, 3], b2[:, 3])
    c2 = (x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2 + eps
    
    # 4. Centroid distance
    cx1 = (b1[:, 0] + b1[:, 2]) / 2.0
    cy1 = (b1[:, 1] + b1[:, 3]) / 2.0
    cx2 = (b2[:, 0] + b2[:, 2]) / 2.0
    cy2 = (b2[:, 1] + b2[:, 3]) / 2.0
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    
    # 5. Aspect ratio consistency v and alpha
    w1, h1 = b1[:, 2] - b1[:, 0], b1[:, 3] - b1[:, 1]
    w2, h2 = b2[:, 2] - b2[:, 0], b2[:, 3] - b2[:, 1]
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + eps)
        
    ciou = iou - (rho2 / c2) - alpha * v
    return 1.0 - ciou

# Verification Run
if __name__ == "__main__":
    b_true = torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32)
    b_pred = torch.tensor([[12.0, 14.0, 48.0, 52.0]], dtype=torch.float32)
    
    ciou_loss = compute_ciou_loss(b_pred, b_true)
    print(f"Computed CIoU Loss: {ciou_loss.item():.4f}")
    assert 0.0 < ciou_loss.item() < 0.5
    
    # Perfect box match
    assert torch.isclose(compute_ciou_loss(b_true, b_true), torch.tensor([0.0]), atol=1e-5)
```

---

## Tier 2: Post-Processing & Non-Maximum Suppression (NMS)

### Exercise 2: Vectorized Greedy Non-Maximum Suppression (NMS)
**Concepts**: Duplicate suppression, IoU thresholding, score ordering, vectorized masking.

#### Problem Specification
Raw object detection networks emit hundreds of overlapping bounding boxes per anchor/grid cell:
1. Implement `vectorized_greedy_nms(boxes, scores, iou_threshold=0.5)` in pure PyTorch without using `torchvision.ops.nms`.
2. Sort candidates descending by confidence score.
3. Iteratively retain the highest scoring box and discard all remaining boxes with $\text{IoU} > \text{threshold}$.
4. Verify that duplicate overlapping detections on synthetic boxes collapse to a single clean detection.

#### Verification Solution
```python
import torch

def compute_pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # boxes: [N, 4] -> (x1, y1, x2, y2)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)

def vectorized_greedy_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> list[int]:
    order = scores.argsort(descending=True)
    keep = []
    
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
            
        cur_box = boxes[i].unsqueeze(0)
        remaining_boxes = boxes[order[1:]]
        
        ious = compute_pairwise_iou(cur_box, remaining_boxes).squeeze(0)
        mask = ious <= iou_threshold
        order = order[1:][mask]
        
    return keep

# Verification Run
if __name__ == "__main__":
    boxes = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],  # Box 0: High score
        [12.0, 11.0, 49.0, 51.0],  # Box 1: Overlaps heavily with 0
        [100.0, 100.0, 140.0, 140.0]  # Box 2: Distinct object
    ])
    scores = torch.tensor([0.95, 0.88, 0.90])
    
    kept_indices = vectorized_greedy_nms(boxes, scores, iou_threshold=0.5)
    print(f"Kept box indices after NMS: {kept_indices}")
    assert kept_indices == [0, 2]
```

---

## Tier 3: Edge Quantization & Post-Training INT8 Conversion

### Exercise 3: Post-Training Quantization (PTQ) INT8 Scale & Zero-Point Math
**Concepts**: Affine quantization $q = \text{round}(x / S) + Z$, calibration min-max observers, INT8 GEMM throughput.

#### Problem Specification
1. In affine uniform quantization, continuous real values $x \in [\alpha, \beta]$ are mapped to 8-bit unsigned integers $q \in [0, 255]$ via:
   $$S = \frac{\beta - \alpha}{255}, \quad Z = \text{round}\left(-\frac{\alpha}{S}\right)$$
2. Implement `quantize_tensor(x)` and `dequantize_tensor(q, S, Z)`.
3. Compute the quantization error $\|\hat{x} - x\|_2$ on normal weights.

#### Verification Solution
```python
import torch

def calculate_quant_parameters(x: torch.Tensor) -> tuple[float, int]:
    alpha = float(x.min())
    beta = float(x.max())
    S = (beta - alpha) / 255.0
    Z = int(round(-alpha / S))
    Z = max(0, min(255, Z))
    return S, Z

def quantize_tensor(x: torch.Tensor, S: float, Z: int) -> torch.Tensor:
    q = torch.round(x / S) + Z
    return torch.clamp(q, 0, 255).to(torch.uint8)

def dequantize_tensor(q: torch.Tensor, S: float, Z: int) -> torch.Tensor:
    return (q.to(torch.float32) - Z) * S

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    weights = torch.randn(100, 100) * 0.5
    S, Z = calculate_quant_parameters(weights)
    
    q_weights = quantize_tensor(weights, S, Z)
    rec_weights = dequantize_tensor(q_weights, S, Z)
    
    max_err = torch.max(torch.abs(weights - rec_weights)).item()
    print(f"Scale S: {S:.5f} | Zero-Point Z: {Z}")
    print(f"Maximum PTQ INT8 Reconstruction Error: {max_err:.5f}")
    assert max_err < S
```
