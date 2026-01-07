Exercises: Advanced Computer Vision Concepts

 Exercise 1: Understanding Transfer Learning

1.  Load a Pre-trained Model: Using `tensorflow.keras.applications`, load the `MobileNetV2` model with `weights='imagenet'` and `include_top=False`.
2.  Inspect the Architecture: Print the model's summary using `model.summary()`.
    How many layers does the model have?
    How many trainable parameters are there initially?
3.  Freeze the Base: Iterate through the model's layers and set `layer.trainable = False` for each one.
4.  Re-check Parameters: Print the model summary again.
    How many "Trainable params" are there now?
    How many "Non-trainable params"?
    Why is the number of trainable parameters not zero? (Hint: Think about the nature of some layers).

 Exercise 2: Conceptual Questions

Answer the following questions based on the lecture notes for this week.

1.  You are building an app for a smartphone that needs to identify different types of fruit in real-time from the phone's camera. Accuracy is important, but the app must run smoothly without lagging. Would you choose a YOLO-based model or a Faster R-CNN-based model? Explain your reasoning.
2.  You are tasked with creating a system to automatically measure the area of different crop fields from high-resolution satellite images. This requires creating a precise pixel-level map of each field. What kind of computer vision task is this? Which architecture mentioned in the notes is specifically designed for this kind of task?
3.  What is the primary purpose of the "skip connections" in a U-Net architecture? What problem do they help solve?

 Exercise 3: Exploring a Pre-trained Model's Predictions

This exercise shows you how a powerful pre-trained model "sees" the world.

1.  Load a Full Model: Load the `ResNet50` model, but this time, keep the top classification layer by setting `include_top=True`.
2.  Load and Preprocess an Image: Find an image of a common object (e.g., a cat, a dog, a car, a coffee mug) and load it into your Python script. You will need to resize it to `(224, 224)` pixels, which is the input size for ResNet50. You'll also need to use `tf.keras.applications.resnet50.preprocess_input` to prepare the image correctly.
3.  Make a Prediction: Use the `model.predict()` method on your preprocessed image.
4.  Decode the Prediction: The model will output a vector of probabilities for 1000 different classes. Use `tf.keras.applications.resnet50.decode_predictions` to see the top predicted classes for your image.

```python
 Example code snippet
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

 1. Load model
model = ResNet50(weights='imagenet', include_top=True)

 2. Load and preprocess image
img_path = 'path/to/your/image.jpg'  Change this to your image path
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

 3. Make prediction
preds = model.predict(x)

 4. Decode and print results
print('Predicted:', decode_predictions(preds, top=3)[0])
```
What did the model predict for your image? Were the predictions accurate?