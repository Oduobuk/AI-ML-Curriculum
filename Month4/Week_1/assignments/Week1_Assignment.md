# Month 4, Week 1 Assignment: Design Your First CNN

## Objective

The goal of this assignment is to solidify your understanding of the fundamental building blocks of a Convolutional Neural Network. You will design a simple CNN architecture on paper for a standard image classification task without writing any code. This exercise focuses on understanding the flow of data and the change in dimensions through the network.

## Scenario

You are tasked with designing a CNN to classify images from the **CIFAR-10 dataset**.

**CIFAR-10 Dataset Specifications:**
*   **Input Image Size:** 32x32 pixels
*   **Color Channels:** 3 (RGB)
*   **Number of Classes:** 10 (e.g., airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)

So, the input volume to your network will be **32x32x3**.

## Instructions

Describe a simple CNN architecture layer by layer. For each layer, you must specify its parameters and calculate the dimensions of its output volume.

**Use the following formula for calculating the output size of a convolutional or pooling layer:**

`Output_Size = (Input_Size - Filter_Size + 2 * Padding) / Stride + 1`

---

### Your Proposed Architecture

Fill out the details for each layer below.

#### Layer 1: Convolutional Layer

*   **Operation:** Convolution + ReLU Activation
*   **Parameters:**
    *   Number of Filters: `16`
    *   Filter Size (Kernel Size): `3x3`
    *   Stride: `1`
    *   Padding: `1`
*   **Calculation:**
    *   `Output_Width = (32 - 3 + 2*1) / 1 + 1 = ?`
    *   `Output_Height = (32 - 3 + 2*1) / 1 + 1 = ?`
*   **Output Volume Dimensions:** `? x ? x 16`

---

#### Layer 2: Pooling Layer

*   **Operation:** Max Pooling
*   **Parameters:**
    *   Filter Size (Window Size): `2x2`
    *   Stride: `2`
    *   Padding: `0`
*   **Calculation (using the output size from Layer 1 as input):**
    *   `Output_Width = (Input_Width - 2 + 2*0) / 2 + 1 = ?`
    *   `Output_Height = (Input_Height - 2 + 2*0) / 2 + 1 = ?`
*   **Output Volume Dimensions:** `? x ? x 16`

---

#### Layer 3: Convolutional Layer

*   **Operation:** Convolution + ReLU Activation
*   **Parameters:**
    *   Number of Filters: `32`
    *   Filter Size: `3x3`
    *   Stride: `1`
    *   Padding: `1`
*   **Calculation (using the output size from Layer 2 as input):**
    *   `Output_Width = ?`
    *   `Output_Height = ?`
*   **Output Volume Dimensions:** `? x ? x 32`

---

#### Layer 4: Pooling Layer

*   **Operation:** Max Pooling
*   **Parameters:**
    *   Filter Size: `2x2`
    *   Stride: `2`
    *   Padding: `0`
*   **Calculation (using the output size from Layer 3 as input):**
    *   `Output_Width = ?`
    *   `Output_Height = ?`
*   **Output Volume Dimensions:** `? x ? x 32`

---

#### Layer 5: Flatten Layer

*   **Operation:** Flatten the 3D volume into a 1D vector.
*   **Calculation:**
    *   `Vector_Length = Output_Width * Output_Height * Depth` (from Layer 4)
    *   `Vector_Length = ? * ? * 32 = ?`
*   **Output Dimensions:** `? x 1`

---

#### Layer 6: Fully Connected (Dense) Layer

*   **Operation:** Dense Layer + ReLU Activation
*   **Parameters:**
    *   Number of Neurons: `128`
*   **Output Dimensions:** `128 x 1`

---

#### Layer 7: Output Layer

*   **Operation:** Fully Connected (Dense) Layer + Softmax Activation
*   **Parameters:**
    *   Number of Neurons: `?` (Hint: How many classes are you trying to predict?)
*   **Output Dimensions:** `? x 1`

## Submission

*   Complete the calculations for each layer, filling in the `?` marks.
*   Save your answers in a Markdown file. There is no need to implement this in code, but be prepared to discuss your reasoning.
