# Month 5, Week 4: Generative Models (AE, VAE, GANs)

## Overview
This week explores the fascinating world of generative models, which are a class of AI models capable of generating new data instances that resemble the training data. We will focus on Autoencoders (AEs), Variational Autoencoders (VAEs), and Generative Adversarial Networks (GANs), understanding their architectures, working principles, and diverse applications.

## Key Concepts
*   **Generative vs. Discriminative Models:** Differentiating between models that generate data and models that classify or predict.
*   **Latent Space:** Understanding the compressed, meaningful representation of data learned by generative models.
*   **Autoencoders (AEs):**
    *   **Encoder-Decoder Architecture:** Compressing input into a latent space and reconstructing it.
    *   **Reconstruction Loss:** Measuring the difference between original and reconstructed data.
    *   **Applications:** Dimensionality reduction, denoising, anomaly detection.
*   **Variational Autoencoders (VAEs):**
    *   **Probabilistic Approach:** Learning a distribution over the latent space rather than a fixed representation.
    *   **Reparameterization Trick:** Enabling backpropagation through the sampling process.
    *   **Loss Function:** Combining reconstruction loss with a KL divergence term to ensure latent space regularity.
    *   **Applications:** Image generation, data imputation.
*   **Generative Adversarial Networks (GANs):**
    *   **Generator and Discriminator:** The adversarial training process between two neural networks.
    *   **Minimax Game:** The objective function that drives the training of GANs.
    *   **Challenges:** Mode collapse, training instability.
    *   **Applications:** Realistic image generation, style transfer, super-resolution.
*   **Other Generative Models:** Briefly touching upon other models like Diffusion Models (DMs) and Normalizing Flows.

## Recommended Reading
*   **Deep Learning** — Ian Goodfellow et al. (The GANs chapter is written by Goodfellow himself, a co-creator of GANs)
*   **Generative Deep Learning** — David Foster (Practical guide to implementing various generative models)
*   **Hands-On Generative Adversarial Networks** — Rafael Valle (Focuses specifically on GANs with practical examples)
*   **Autoencoder, VAE & GAN sections in Dive Into Deep Learning** (FREE online book, provides good theoretical and practical insights)

## Note on Transcripts
Due to an issue with the transcript API (402 Client Error: Payment Required), the YouTube video transcripts for this week could not be automatically extracted. The lecture notes have been synthesized based on the provided topics and recommended reading materials.
