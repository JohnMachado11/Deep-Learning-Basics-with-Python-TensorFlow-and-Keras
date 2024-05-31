import tensorflow as tf
from tensorflow import keras
from keras.activations import relu, softmax
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np


# Load and normalize the MNIST dataset
mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# print(x_train[0])

# Build the model
#  --- Method 1 for building Neural Net Architecture ---
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Input Layer
    tf.keras.layers.Dense(128, activation=relu), # Hidden Layer #1 - 128 Neurons in this layer
    tf.keras.layers.Dense(128, activation=relu), # Hidden Layer #2 - 128 Neurons in this layer
    tf.keras.layers.Dense(10, activation=softmax) # Output Layer
])
# ---------------------------------------------------------

# Compile the model
model.compile(optimizer="adam", # Good default optimizer.
            loss="sparse_categorical_crossentropy", # How we'll calculate our "error". Neural Network aims to minimize loss.
            metrics=["accuracy"]) # What to track

# model.fit(x_train, y_train, epochs=3)
model.fit(x_train, y_train, epochs=50)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

model.summary()

# Print out the activation functions for each layer
for layer in model.layers:
    if hasattr(layer, 'activation'):
        print(f"Layer: {layer.name}, Activation Function: {layer.activation.__name__}")
print("\n")

model.save(filepath="mnist_reader.h5")
predictions = model.predict([x_test])

# new_model = tf.keras.models.load_model("mnist_reader.h5")
# predictions = new_model.predict([x_test])

print(np.argmax(predictions[0]))

# plt.imshow(x_test[0], cmap=plt.cm.binary)
# plt.title(f"Predicted Label: {np.argmax(predictions[0])}")
# plt.show()

# Plotting the first 10 predictions
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.axis('off')

plt.tight_layout()
plt.show()


#  --- Method 2 for building Neural Net Architecture ---
# model = tf.keras.models.Sequential()
# # Input Layer
# model.add(tf.keras.layers.Flatten())
# # Hidden Layer #1 - 128 Neurons in this layer
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# # Hidden Layer #2 - 128 Neurons in this layer
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# # Output Layer
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# ---------------------------------------------------------