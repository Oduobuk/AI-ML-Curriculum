# Month 3, Week 4 Cheatsheet: Real-Time Edge Vision, YOLO Detection & CIoU Loss

---

## 1. Object Detection Bounding Box Formulations & Coordinate Systems

```
                     Bounding Box Parameterizations
        Corner Format (x1, y1, x2, y2)        Center Format (cx, cy, w, h)
     (x1, y1) ┌──────────────┐                          ┌──────────────┐
              │              │                          │   (cx, cy)   │
              │              │                          │      •       │ h
              │              │                          │              │
              └──────────────┘ (x2, y2)                 └──────────────┘
                                                               w
```

### Complete IoU (CIoU) Loss Formulation
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$
where:
- $\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$.
- $\rho^2$: Squared Euclidean distance between box center points.
- $c^2$: Squared diagonal distance of the smallest enclosing bounding box.
- $v = \frac{4}{\pi^2} \left(\arctan\frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan\frac{w}{h}\right)^2$ (aspect ratio consistency).
- $\alpha = \frac{v}{(1 - \text{IoU}) + v}$ (adaptive dynamic trade-off parameter).

---

## 2. Anchor-Free Decoupled Detection Head (YOLOv8/v11 Style)

Separates class prediction from geometric coordinate regression:
- **Classification Branch**: Predicts class logits using Binary Cross-Entropy (BCE) Loss.
- **Bounding Box Regression Branch**: Predicts distance offsets $(l, t, r, b)$ from grid cell anchor points $(c_x, c_y)$ to box edges:
  $$\hat{x}_1 = c_x - l, \quad \hat{y}_1 = c_y - t, \quad \hat{x}_2 = c_x + r, \quad \hat{y}_2 = c_y + b$$

---

## 3. Vectorized Greedy Non-Maximum Suppression (NMS)

Prunes redundant overlapping detections:
```python
def vectorized_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> list[int]:
    order = scores.argsort(descending=True)
    keep = []
    
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
            
        cur_box = boxes[i].unsqueeze(0)
        remaining = boxes[order[1:]]
        
        # Vectorized pairwise IoU
        ious = compute_pairwise_iou(cur_box, remaining).squeeze(0)
        mask = ious <= iou_threshold
        order = order[1:][mask]
        
    return keep
```

---

## 4. Post-Training Quantization (PTQ) INT8 Math

Maps floating-point weights $x \in [\alpha, \beta]$ to 8-bit unsigned integers $q \in [0, 255]$:
$$S = \frac{\beta - \alpha}{255}, \qquad Z = \text{round}\left(-\frac{\alpha}{S}\right)$$
$$\text{Quantize: } q = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right) + Z, 0, 255\right)$$
$$\text{Dequantize: } \hat{x} = (q - Z) \times S$$
- **Latency Gain**: INT8 Tensor Cores yield up to $3\times$ higher inference throughput and $75\%$ reduction in memory bandwidth.
