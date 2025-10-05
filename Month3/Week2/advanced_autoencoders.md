# Advanced Topic: Autoencoders for Dimensionality Reduction

## Introduction
Autoencoders are neural networks that learn efficient data encodings by compressing input into a latent space and reconstructing it.

## Basic Implementation
```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def build_autoencoder(encoding_dim=32, input_shape=(784,)):
    input_img = Input(shape=input_shape)
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(input_shape[0], activation='sigmoid')(encoded)
    
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    
    return autoencoder, encoder
```

## Convolutional Autoencoder
```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

def build_conv_autoencoder():
    input_img = Input(shape=(28, 28, 1))
    
    # Encoder
    x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2,2), padding='same')(x)
    encoded = Conv2D(8, (3,3), activation='relu', padding='same')(x)
    
    # Decoder
    x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2,2))(x)
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
    
    return Model(input_img, decoded)
```

## Variational Autoencoder (VAE)
```python
from tensorflow.keras import backend as K

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Build VAE
inputs = Input(shape=(784,))
x = Dense(256, activation='relu')(inputs)
z_mean = Dense(32)(x)
z_log_var = Dense(32)(x)
z = Lambda(sampling)([z_mean, z_log_var])

# Instantiate VAE
vae = Model(inputs, decoder(z), name='vae')
```

## When to Use Autoencoders
- Non-linear dimensionality reduction
- Anomaly detection
- Image denoising
- Feature extraction
- Data generation (with VAEs)

## Comparison with PCA
- **Autoencoders**:
  - Learn non-linear transformations
  - Better for complex data
  - More computationally expensive
  
- **PCA**:
  - Linear transformation
  - Faster and simpler
  - More interpretable components
