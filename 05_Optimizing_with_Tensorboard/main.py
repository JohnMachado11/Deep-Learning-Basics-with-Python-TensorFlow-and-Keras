import tensorflow as tf
from tensorflow import keras
from keras.activations import relu, softmax, sigmoid
from tensorflow.keras.callbacks import TensorBoard
import time
import numpy as np
import pickle
import sys


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
y = np.array(y)

# Normalize the pixel values to the range [0, 1]
X = X / 255.0

# Best Model
dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

# All these models were created, after checking with TensorBoard,
# 3 Convolutional Layers and 0 Dense Layers = the best
# dense_layers = [0, 1, 2]
# layer_sizes = [32, 64, 128]
# conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = tf.keras.Sequential()

            model.add(tf.keras.layers.Conv2D(filters=layer_size, kernel_size=(3, 3), input_shape=X.shape[1:]))
            model.add(tf.keras.layers.Activation("relu"))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer - 1):
                model.add(tf.keras.layers.Conv2D(filters=layer_size, kernel_size=(3, 3)))
                model.add(tf.keras.layers.Activation("relu"))
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(tf.keras.layers.Flatten())

            for l in range(dense_layer):
                model.add(tf.keras.layers.Dense(layer_size))
                model.add(tf.keras.layers.Activation("relu"))

            model.add(tf.keras.layers.Dense(1))
            model.add(tf.keras.layers.Activation("sigmoid"))

            # tensorboard --logdir logs
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            # Compile the model
            model.compile(optimizer="adam", # Good default optimizer.
                        loss="binary_crossentropy", # How we'll calculate our "error". Neural Network aims to minimize loss.
                        metrics=["accuracy"]) # What to track

            model.fit(X, y, batch_size=32, epochs=20, validation_split=0.3, callbacks=[tensorboard])

            model.summary()

# # Print out the activation functions for each layer
# for layer in model.layers:
#     if hasattr(layer, 'activation'):
#         print(f"Layer: {layer.name}, Activation Function: {layer.activation.__name__}")
# print("\n")
