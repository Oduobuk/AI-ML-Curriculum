Transfer Learning with Pre-trained Models

Transfer learning is a powerful technique in computer vision that allows you to leverage pre-trained models to achieve high performance on your own tasks, even with limited data. Instead of training a model from scratch, you can use a model that has already been trained on a large dataset like ImageNet, which contains millions of images across thousands of categories.

Keras provides easy access to a variety of pre-trained models, including VGG16, ResNet, Inception, and MobileNet.

 Using Pre-trained Models

There are two main ways to use a pre-trained model:

1.  Feature Extraction: In this approach, you use the convolutional base of a pre-trained model to extract features from your new dataset. The convolutional base is the part of the model that contains the convolutional and pooling layers, which are responsible for learning spatial hierarchies of features. You then feed these features into a new classifier that you train from scratch.

2.  Fine-Tuning: In addition to feature extraction, you can also "unfreeze" some of the top layers of the pre-trained model's convolutional base and train them on your new data with a very low learning rate. This allows the model to adapt its feature recognition to the new dataset, potentially leading to better performance.

 Example: Feature Extraction with VGG16

Here's an example of how to use the VGG16 model for feature extraction in Keras:

```python
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

 Load the VGG16 model, excluding the top fully connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

 Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

 Create a new model on top
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  10 classes

model = Model(inputs=base_model.input, outputs=predictions)

 Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

 Create data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
        'data/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

 Train the model
model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=50)
```

 Example: Fine-Tuning with ResNet

Here's an example of how to fine-tune the ResNet50 model in Keras:

```python
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD

 Load the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False)

 Create a new model on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  10 classes

model = Model(inputs=base_model.input, outputs=predictions)

 First, train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 model.fit(...)  Train the top layers

 At this point, a top layers are well trained and we can start fine-tuning
 convolutional layers from ResNet50. We will freeze the bottom N layers
 and train the remaining top layers.

 Let's say we want to fine-tune from the 143rd layer onwards
for layer in model.layers[:143]:
   layer.trainable = False
for layer in model.layers[143:]:
   layer.trainable = True

 We need to recompile the model for these modifications to take effect
 We use a very low learning rate with SGD to avoid overfitting
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

 model.fit(...)  Train the fine-tuned model
```