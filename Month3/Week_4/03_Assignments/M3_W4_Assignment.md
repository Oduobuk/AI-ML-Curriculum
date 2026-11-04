# Week 4 Production Engineering Assignment: Real-Time Edge Vision Pipeline & YOLO Detection Engine

## Executive Overview & Architectural Motivation
Real-time computer vision in autonomous driving, industrial robotics, and edge camera analytics requires detecting multi-scale objects within stringent real-time budgets ($< 30\text{ ms}$ latency). Achieving this performance demands anchor-free feature pyramid heads, Complete IoU (CIoU) loss optimization, vectorized Non-Maximum Suppression (NMS), and INT8 deployment engines.

In this assignment, you will engineer a **Production-Grade Real-Time Object Detection Pipeline** inspired by YOLOv8/v11.

You will construct an anchor-free detection head that emits bounding box regression offsets and class logits across multi-scale feature strides ($8, 16, 32$), implement Complete IoU (CIoU) loss and Distribution Focal Loss (DFL), vectorize Greedy NMS, and benchmark post-training INT8 quantization.

---

## Technical Specifications & Architecture Diagram

```
                 Multi-Scale Feature Pyramid [P3 (stride 8), P4 (stride 16), P5 (stride 32)]
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │         Anchor-Free Decoupled Detection Head          │
                 │   - Classification Branch: [B, C, H_i, W_i] (BCE Loss)│
                 │   - Box Regression Branch: [B, 4, H_i, W_i] (CIoU/DFL)│
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │             Target Assigner & Losses                  │
                 │   - Task-Aligned Assigner (TAL): $t = s^\alpha \times \text{IoU}^\beta$ │
                 │   - Complete IoU Loss ($\mathcal{L}_{\text{CIoU}}$)    │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │         Vectorized Non-Maximum Suppression (NMS)      │
                 │   - Confidence Score Filter (> 0.25)                  │
                 │   - Pairwise IoU Masking & Duplicate Pruning          │
                 └──────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                 Real-Time Bounding Box Detections at 60+ FPS
```

---

## Detailed Deliverable Requirements

### 1. Anchor-Free Multi-Stride Detection Head (`head.py`)
Implement `AnchorFreeDetectionHead(nn.Module)`:
- Accepts multi-scale feature maps from strides 8, 16, and 32.
- Decouples classification and bounding box regression into separate parallel convolutional branches.
- Decodes box coordinates from grid cell centers:
  $$\hat{x}_1 = (c_x - l) \times s, \quad \hat{y}_1 = (c_y - t) \times s, \quad \hat{x}_2 = (c_x + r) \times s, \quad \hat{y}_2 = (c_y + b) \times s$$
  where $(l, t, r, b)$ are predicted distance offsets to the box edges.

### 2. Complete IoU (CIoU) Regression Loss Engine (`losses.py`)
- Implement `CIoULoss`:
  - Penalizes overlapping area, central point distance, and aspect ratio discrepancies.
  - Returns scalar mean loss over positive matches.

### 3. Vectorized NMS & INT8 Benchmarking Suite (`engine.py`)
- Implement `vectorized_nms(boxes, scores, iou_threshold=0.5, conf_threshold=0.25)`:
  - Vectorized pairwise IoU computation.
  - Zero Python list-loop overhead on candidate box pruning.
- Quantize the convolutional head using PyTorch Dynamic/Static INT8 quantization and compare latency and FLOPs.

---

## Starter Code Template & Submission Skeleton

```python
"""
realtime_yolo_detector.py - Production Real-Time Vision Suite
"""
import torch
import torch.nn as nn
import math
from typing import List, Tuple

class DecoupledDetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 4, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_branch(x)  # [B x Num_Classes x H x W]
        reg_dist = torch.relu(self.reg_branch(x))  # [B x 4 x H x W]
        return cls_logits, reg_dist

def vectorized_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    order = scores.argsort(descending=True)
    keep = []
    
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
            
        cur = boxes[i].unsqueeze(0)
        rem = boxes[order[1:]]
        
        # Pairwise IoU
        area_cur = (cur[:, 2] - cur[:, 0]) * (cur[:, 3] - cur[:, 1])
        area_rem = (rem[:, 2] - rem[:, 0]) * (rem[:, 3] - rem[:, 1])
        
        lt = torch.max(cur[:, None, :2], rem[:, :2])
        rb = torch.min(cur[:, None, 2:], rem[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area_cur[:, None] + area_rem - inter
        iou = (inter / union.clamp(min=1e-6)).squeeze(0)
        
        order = order[1:][iou <= iou_thresh]
        
    return torch.tensor(keep, dtype=torch.long)

# Verification Run
if __name__ == "__main__":
    torch.manual_seed(42)
    B, C, H, W = 2, 64, 20, 20
    head = DecoupledDetectionHead(in_channels=C, num_classes=3)
    feat = torch.randn(B, C, H, W)
    
    cls_logits, reg_dist = head(feat)
    print(f"Classification Logits Shape: {cls_logits.shape} (Expected: [{B}, 3, 20, 20])")
    print(f"Regression Distance Shape:   {reg_dist.shape} (Expected: [{B}, 4, 20, 20])")
    
    assert cls_logits.shape == (B, 3, H, W)
    assert reg_dist.shape == (B, 4, H, W)
    
    # Test NMS
    dummy_boxes = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 11.0, 48.0, 52.0],
        [100.0, 100.0, 150.0, 150.0]
    ])
    dummy_scores = torch.tensor([0.9, 0.7, 0.85])
    kept = vectorized_nms(dummy_boxes, dummy_scores, iou_thresh=0.4)
    print(f"Kept indices after NMS: {kept.tolist()}")
    assert kept.tolist() == [0, 2]
```

---

## Grading Rubric (100 Points Total)

| Component | Points | Criteria |
| :--- | :--- | :--- |
| **Decoupled Detection Head** | 30 | Correct convolutional channels, grid-cell center offset mapping, activation functions. |
| **Complete IoU (CIoU) Loss Formulation** | 30 | Accurate aspect ratio consistency $v$ and trade-off parameter $\alpha$ derivation. |
| **Vectorized Non-Maximum Suppression** | 20 | Pure PyTorch tensor operations with zero Python loops over candidate matches. |
| **Architectural Rigor & Clean Typing** | 10 | Clear parameter naming, docstrings, and strict tensor shape assertions. |
| **Verification & Latency Benchmarks** | 10 | End-to-end execution script achieving sub-20ms forward+NMS latency on standard images. |
