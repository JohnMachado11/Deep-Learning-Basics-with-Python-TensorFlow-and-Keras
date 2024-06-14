import tensorflow as tf
from tensorflow import keras
from keras.activations import relu, softmax


# Load and normalize the MNIST dataset
mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# print(x_train.shape)
# print(x_train[0])
print(x_train.shape[1:])

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=128, input_shape=(28, 28), activation=relu, return_sequences=True), # Input Layer
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.LSTM(units=128, activation=relu),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=32, activation=relu),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=10, activation=softmax)
])

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)

model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])


model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))