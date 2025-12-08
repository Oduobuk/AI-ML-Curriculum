# Month 5, Week 4: Generative Models Assignment

## Instructions
This assignment will guide you through implementing a simple Autoencoder for image reconstruction.

## Task: Implement and Train a Simple Autoencoder

**Objective:** To build and train a basic Autoencoder using TensorFlow/Keras or PyTorch to reconstruct images from a dataset.

**Dataset:**
*   Use the **Fashion MNIST** dataset. This dataset consists of 28x28 grayscale images of clothing items. It's a good dataset for getting started with image-based generative models.

**Implementation Steps:**

1.  **Setup:**
    *   Choose your preferred deep learning framework (TensorFlow/Keras or PyTorch).
    *   Load the Fashion MNIST dataset. Normalize the pixel values to be between 0 and 1.
    *   Split the data into training and testing sets.

2.  **Autoencoder Architecture:**
    *   Design a simple Autoencoder architecture. It should consist of:
        *   **Encoder:** A neural network that takes an input image and compresses it into a lower-dimensional latent space representation. You can use dense layers or convolutional layers.
        *   **Decoder:** A neural network that takes the latent space representation and reconstructs the original image.
    *   **Example (Keras with Dense Layers):**
        ```python
        from tensorflow import keras
        from tensorflow.keras import layers

        # Encoder
        encoder_input = keras.Input(shape=(28 * 28,))
        x = layers.Dense(128, activation="relu")(encoder_input)
        latent_output = layers.Dense(64, activation="relu")(x) # Latent space

        # Decoder
        decoder_input = keras.Input(shape=(64,))
        x = layers.Dense(128, activation="relu")(decoder_input)
        decoder_output = layers.Dense(28 * 28, activation="sigmoid")(x) # Output image

        encoder = keras.Model(encoder_input, latent_output, name="encoder")
        decoder = keras.Model(decoder_input, decoder_output, name="decoder")

        autoencoder_input = keras.Input(shape=(28 * 28,))
        encoded_img = encoder(autoencoder_input)
        reconstructed_img = decoder(encoded_img)
        autoencoder = keras.Model(autoencoder_input, reconstructed_img, name="autoencoder")
        ```

3.  **Training:**
    *   Compile the Autoencoder with an appropriate optimizer (e.g., `adam`) and a loss function suitable for image reconstruction (e.g., `binary_crossentropy` for pixel values between 0 and 1, or `mse`).
    *   Train the Autoencoder on the training data. The input and target for the Autoencoder will be the same (the original images).
    *   Train for a reasonable number of epochs (e.g., 10-50).

4.  **Evaluation and Visualization:**
    *   After training, use the trained Autoencoder to reconstruct some images from the test set.
    *   Visualize a few original images alongside their reconstructed counterparts. This will help you assess the quality of the reconstruction.
    *   Optionally, visualize the latent space if you used a 2D latent space.

## Submission
*   A single Jupyter Notebook or Python script containing all your code.
*   Include visualizations of original vs. reconstructed images.
*   Briefly discuss your observations about the reconstruction quality and any challenges you faced.
