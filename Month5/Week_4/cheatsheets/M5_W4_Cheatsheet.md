# Month 5, Week 4: Generative Models Cheatsheet

## Generative vs. Discriminative Models

| Feature | Generative Models | Discriminative Models |
| :--- | :--- | :--- |
| **Goal** | Learn the distribution of data (P(X)) or joint distribution (P(X,Y)) to generate new samples. | Learn to distinguish between different classes (P(Y|X)) or predict an output. |
| **Examples** | AEs, VAEs, GANs, Diffusion Models | Logistic Regression, SVM, Decision Trees, most CNNs for classification. |

## Autoencoders (AEs)

*   **Purpose:** Learn an efficient, compressed representation (latent space) of input data for reconstruction.
*   **Architecture:**
    *   **Encoder:** Maps input `x` to latent representation `z` (`z = f(x)`).
    *   **Decoder:** Maps latent representation `z` back to reconstructed input `x'` (`x' = g(z)`).
*   **Loss Function:** Primarily **Reconstruction Loss** (e.g., Mean Squared Error for continuous data, Binary Cross-Entropy for binary data).
    *   `L_AE = ||x - x'||^2` (MSE) or `L_AE = BCE(x, x')`
*   **Applications:** Dimensionality reduction, denoising, anomaly detection, feature learning.

## Variational Autoencoders (VAEs)

*   **Purpose:** Learn a *probabilistic* mapping from input to latent space, allowing for continuous and smooth generation.
*   **Architecture:** Similar encoder-decoder, but the encoder outputs parameters (mean `μ` and variance `σ`) of a probability distribution (typically Gaussian) in the latent space.
*   **Reparameterization Trick:** Allows backpropagation through the sampling process by sampling `z = μ + σ * ε`, where `ε ~ N(0, 1)`.
*   **Loss Function:** Combines **Reconstruction Loss** and **KL Divergence Loss**.
    *   `L_VAE = Reconstruction_Loss(x, x') + KL_Divergence(N(μ, σ^2) || N(0, 1))`
    *   KL Divergence term encourages the latent distribution to be close to a standard normal distribution, ensuring a well-structured latent space.
*   **Applications:** Image generation, data imputation, semi-supervised learning.

## Generative Adversarial Networks (GANs)

*   **Purpose:** Generate realistic data samples that are indistinguishable from real data.
*   **Architecture:** Two competing neural networks:
    *   **Generator (G):** Takes random noise `z` as input and generates synthetic data `G(z)`.
    *   **Discriminator (D):** Takes either real data `x` or generated data `G(z)` and tries to distinguish between them.
*   **Training Process (Minimax Game):**
    *   **Discriminator's Goal:** Maximize `log(D(x)) + log(1 - D(G(z)))` (correctly classify real as real, fake as fake).
    *   **Generator's Goal:** Minimize `log(1 - D(G(z)))` (fool the discriminator into classifying fake as real).
*   **Loss Function:**
    *   `min_G max_D V(D, G) = E_{x~p_{data}(x)}[log D(x)] + E_{z~p_z(z)}[log(1 - D(G(z)))]`
*   **Challenges:** Mode collapse, training instability, difficulty in evaluating generated samples.
*   **Applications:** Realistic image generation (e.g., faces, landscapes), style transfer, super-resolution, data augmentation.

## Other Generative Models

*   **Diffusion Models (DMs):** A class of generative models that learn to reverse a gradual diffusion process (adding noise) to generate data. Known for high-quality image generation.
*   **Normalizing Flows:** Models that learn a sequence of invertible transformations to map a simple distribution (e.g., Gaussian) to a complex data distribution.
