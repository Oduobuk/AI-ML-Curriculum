# Month 4, Week 2 Assignment: Transfer Learning Strategy Design

## Objective

This assignment requires you to design a transfer learning strategy for a specific computer vision task. You will outline your approach, including the choice of pre-trained model, the transfer learning technique, and data augmentation strategies, without writing any code. This exercise focuses on applying theoretical knowledge to a practical problem.

## Scenario

You are working on a project to build a model that can classify **120 different dog breeds** from images. You have a dataset of approximately **10,000 images**, with roughly 80 images per breed. This is a relatively small dataset for training a deep CNN from scratch.

## Instructions

In a Markdown file, describe your transfer learning strategy by addressing the following points:

### 1. Problem Analysis

*   **Task Type:** What kind of computer vision task is this (e.g., image classification, object detection, segmentation)?
*   **Dataset Characteristics:**
    *   Size: Is it large, medium, or small?
    *   Similarity to ImageNet: Do you expect the features learned by ImageNet-trained models to be highly relevant to dog breeds? Justify your answer.

### 2. Choice of Pre-trained Model

*   **Which pre-trained model would you choose?** (e.g., ResNet50, VGG16, InceptionV3, MobileNetV2).
*   **Justify your choice:** Consider factors like model complexity, typical performance, and computational efficiency.

### 3. Transfer Learning Strategy

*   **Which transfer learning strategy would you primarily use?** (Feature Extraction or Fine-tuning).
*   **Explain your choice:** Based on the dataset characteristics and the nature of the task, why is this strategy more appropriate?
*   **If Fine-tuning:**
    *   Would you fine-tune all layers or only a subset? If a subset, which layers (early, middle, late) would you unfreeze and why?
    *   What considerations would you have for the learning rate during fine-tuning?

### 4. Data Augmentation Strategy

*   **Why is data augmentation important for this task?**
*   **List at least 5 specific data augmentation techniques** you would apply to your dataset (e.g., random horizontal flip, rotation, zoom, brightness adjustment).
*   **Explain why each chosen technique is relevant** for improving the model's performance on dog breed classification.

### 5. Evaluation Metrics

*   What primary metric(s) would you use to evaluate the performance of your model? (e.g., Accuracy, F1-score, Precision, Recall).

## Submission

*   Save your detailed strategy in a Markdown file named `Week2_Transfer_Learning_Strategy.md`.
*   Be prepared to discuss your design choices.
