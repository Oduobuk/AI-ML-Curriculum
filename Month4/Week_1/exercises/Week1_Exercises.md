# Month 4, Week 1 Exercises: CNN Concepts

## Objective

These exercises are designed to test your understanding of the core concepts and mechanics of a Convolutional Neural Network, particularly how data dimensions change as they flow through the network.

**Use the following formula for your calculations:**
`Output_Size = (Input_Size - Filter_Size + 2 * Padding) / Stride + 1`

---

### Exercise 1: Calculate Convolutional Layer Output

You have an input volume with dimensions **64x64x3**. You apply a single convolutional layer with the following parameters:

*   Number of Filters: `8`
*   Filter Size: `5x5`
*   Stride: `1`
*   Padding: `0`

**Questions:**
1.  What will be the width and height of the output feature maps?
2.  What will be the depth (number of channels) of the output volume?
3.  What are the full dimensions of the output volume?

---

### Exercise 2: The Effect of Padding

You have the same input volume: **64x64x3**. You apply a convolutional layer with the same parameters as Exercise 1, but this time you add padding.

*   Number of Filters: `8`
*   Filter Size: `5x5`
*   Stride: `1`
*   **Padding: `2`**

**Questions:**
1.  What are the new width and height of the output feature maps?
2.  What is the primary benefit of adding padding in this specific case?

---

### Exercise 3: The Effect of Stride

You have the same input volume: **64x64x3**. You apply a convolutional layer, but this time with a larger stride.

*   Number of Filters: `8`
*   Filter Size: `3x3`
*   **Stride: `2`**
*   Padding: `1`

**Questions:**
1.  What are the width and height of the output feature maps?
2.  What is the main effect of using a stride of 2 compared to a stride of 1?

---

### Exercise 4: Calculate Pooling Layer Output

Imagine the output from Exercise 2 (`64x64x8`) is passed into a Max Pooling layer with the following parameters:

*   Pool Size (Filter Size): `2x2`
*   Stride: `2`
*   Padding: `0`

**Questions:**
1.  What will be the width and height of the volume after pooling?
2.  What will be the depth (number of channels) of the volume after pooling?
3.  What is the primary purpose of this pooling layer?

---

### Exercise 5: Conceptual Questions

Answer the following questions in 1-2 sentences.

1.  Why is applying a ReLU activation function after a convolution important?
2.  What is the difference between a "filter" and a "feature map"?
3.  Why do we use fully connected (dense) layers at the *end* of a CNN classifier and not at the beginning?
