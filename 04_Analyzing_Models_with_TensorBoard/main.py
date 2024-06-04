import tensorflow as tf
from tensorflow import keras
from keras.activations import relu, softmax, sigmoid
from tensorflow.keras.callbacks import TensorBoard
import time
import numpy as np
import pickle
import sys


NAME = "Cats-vs-dog-cnn-64x2-{}".format(int(time.time()))

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Terminal Command:
# tensorboard --logdir logs
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
y = np.array(y)


# Normalize the pixel values to the range [0, 1]
X = X / 255.0


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=X.shape[1:], activation=relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(), # this converts our 3D feature maps to 1D feature vectors
    tf.keras.layers.Dense(64, activation=relu),
    tf.keras.layers.Dense(1, activation=sigmoid)
])


# Compile the model
model.compile(optimizer="adam", # Good default optimizer.
            loss="binary_crossentropy", # How we'll calculate our "error". Neural Network aims to minimize loss.
            metrics=["accuracy"]) # What to track

model.fit(X, y, batch_size=32, epochs=50, validation_split=0.3, callbacks=[tensorboard])

model.summary()

# Print out the activation functions for each layer
for layer in model.layers:
    if hasattr(layer, 'activation'):
        print(f"Layer: {layer.name}, Activation Function: {layer.activation.__name__}")
print("\n")
