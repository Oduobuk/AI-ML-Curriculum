# Month 4, Week 1 Cheatsheet: CNN Fundamentals

## Core CNN Layers & Operations

| Operation         | Purpose                                                              | Key Parameters                                                              |
| ----------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Convolution**   | Extracts features by sliding a filter over the input.                | `Number of Filters`, `Filter Size` (e.g., 3x3), `Stride`, `Padding`         |
| **Activation (ReLU)** | Introduces non-linearity. `ReLU(x) = max(0, x)`.                     | None. Applied element-wise after convolution.                               |
| **Pooling (Max)** | Downsamples feature maps to reduce computation and add invariance.   | `Pool Size` (e.g., 2x2), `Stride`                                           |
| **Flatten**       | Converts the final 3D feature map volume into a 1D vector.           | None.                                                                       |
| **Dense (Fully Connected)** | Performs classification based on the high-level features from flatten. | `Number of Neurons`                                                         |

---

## The CNN Architecture Flow

A typical CNN for classification follows this pattern:

**Input Image** -> `[CONV -> ReLU -> POOL]` -> `[CONV -> ReLU -> POOL]` -> ... -> **Flatten** -> **Dense** -> **Dense (Softmax)** -> **Output Probabilities**

1.  **Feature Learning:** The `[CONV -> ReLU -> POOL]` blocks are repeated. Early blocks learn simple features (edges, colors). Deeper blocks learn complex features (eyes, textures, shapes).
2.  **Classification:** The `Flatten` and `Dense` layers take the learned features and make a final prediction.

---

## Calculating Output Dimensions

This formula is essential for designing your network architecture.

Given:
-   `W_in`: Input Width/Height
-   `F`: Filter Size (e.g., 3 for a 3x3 filter)
-   `P`: Padding (the number of pixels added to each side of the input)
-   `S`: Stride

The output width/height `W_out` of a **Convolutional** or **Pooling** layer is:

`W_out = (W_in - F + 2*P) / S + 1`

**Example:**
-   Input: `32x32`
-   Filter: `5x5`
-   Padding: `2`
-   Stride: `1`
-   `W_out = (32 - 5 + 2*2) / 1 + 1 = (32 - 5 + 4) / 1 + 1 = 31 + 1 = 32`
    - The output size is `32x32`. This is a common setup to preserve the spatial dimensions.

---

## Key Terminology

*   **Filter / Kernel:** A small matrix of weights that acts as a feature detector. The values in this matrix are what the network learns during training.
*   **Feature Map / Activation Map:** The output of a convolution operation. It shows where a specific feature was detected in the input.
*   **Stride:** The number of pixels the filter moves at a time as it slides across the input. A stride of `1` moves one pixel at a time. A stride of `2` skips every other pixel.
*   **Padding:** Adding a border of zeros around the input image. This is primarily used to control the output size and ensure the filter can be applied to the edges of the image.
*   **Receptive Field:** The area of the original input image that influences the activation of a single neuron in a given layer.
*   **Channels:** The depth of an image or feature map. An RGB image has 3 channels. A convolutional layer with 64 filters produces a feature map with 64 channels.
