Cheatsheet: Advanced Computer Vision

 Transfer Learning

A technique where a model pre-trained on a large dataset (e.g., ImageNet) is reused for a new, related task.

| Method | Description | When to Use |
| :--- | :--- | :--- |
| Feature Extraction | Use the convolutional base of the pre-trained model to extract features. Freeze the base and train only a new, custom classifier on top. | Smaller dataset, computationally limited, dataset is similar to the original. |
| Fine-Tuning | In addition to training a new classifier, unfreeze the top few layers of the pre-trained base and train them with a very low learning rate. | Larger dataset, more computational power, dataset is somewhat different from the original. |

Keras Implementation:
- Load a model from `tf.keras.applications` with `include_top=False`.
- Freeze layers by setting `layer.trainable = False`.
- Add your own `Flatten`, `Dense`, and output layers.
- For fine-tuning, unfreeze some layers (`layer.trainable = True`) and re-compile with a very low learning rate optimizer like `SGD`.

---

 Object Detection

Identifies and localizes multiple objects in an image with bounding boxes.

| Family | Approach | Key Characteristics | Famous Models |
| :--- | :--- | :--- | :--- |
| Two-Stage | 1. Propose Regions of Interest (RoIs). 2. Classify each region. | High accuracy, but slower. | R-CNN, Fast R-CNN, Faster R-CNN |
| One-Stage | Treats detection as a single regression problem (predicts boxes and classes directly from the image). | Extremely fast (real-time), but can be less accurate for small objects. | YOLO, SSD |

---

 Semantic Segmentation

Classifies every single pixel in an image to a specific class, creating a pixel-level mask.

 U-Net Architecture
A popular U-shaped architecture for segmentation.
- Contracting Path (Encoder): A standard CNN that downsamples the image to extract context features. Consists of `Conv2D` and `MaxPooling2D` layers.
- Expansive Path (Decoder): Symmetrically upsamples the feature map to recover spatial resolution for localization. Uses "up-convolution" or `Conv2DTranspose`.
- Skip Connections: The key innovation. Feature maps from the encoder are concatenated with feature maps in the decoder. This allows the decoder to use high-resolution features from the encoder to produce more precise segmentations.

---

 Famous Architectures

| Architecture | Key Innovation | Best For |
| :--- | :--- | :--- |
| Inception (GoogLeNet) | Inception Module: Performs convolutions of different sizes (1x1, 3x3, 5x5) in parallel and concatenates the results. Captures features at multiple scales efficiently. | High accuracy with computational efficiency. |
| MobileNet | Depthwise Separable Convolutions: Factors a standard convolution into a depthwise convolution and a 1x1 pointwise convolution. Drastically reduces computation. | Mobile and embedded devices where resources are limited. |