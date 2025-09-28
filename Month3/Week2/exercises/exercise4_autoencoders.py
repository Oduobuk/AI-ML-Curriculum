"""
Exercise 4: Autoencoders for Dimensionality Reduction
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

def load_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    return x_train.reshape(-1, 784), x_test.reshape(-1, 784)

def build_autoencoder():
    # Simple Autoencoder
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder

def plot_results(encoder, decoder, x_test):
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        
        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
    plt.show()

def main():
    x_train, x_test = load_data()
    
    # Build and train
    autoencoder, encoder = build_autoencoder()
    autoencoder.fit(x_train, x_train,
                   epochs=50,
                   batch_size=256,
                   shuffle=True,
                   validation_data=(x_test, x_test))
    
    # Visualize results
    plot_results(encoder, autoencoder, x_test[:10])

if __name__ == "__main__":
    main()
